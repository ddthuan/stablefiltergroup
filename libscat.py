import torch
from torch import nn
import torch.optim as optim
from torch.nn import functional as F

import torchvision
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
from torchvision.transforms import functional as Fv

from matplotlib import pyplot as plt
from PIL import Image
import seaborn as sns

import numpy as np
import itertools
from numpy.fft import fft2, ifft2
import io
import pywt 

from collections import OrderedDict

from colorsys import hls_to_rgb

import pandas as pd
import datetime
import timeit
from time import process_time
import os

from pytorch_wavelets import ScatLayer

from functools import partial

from torchsummary import summary


# # https://stackoverflow.com/questions/34768717/matplotlib-unable-to-save-image-in-same-resolution-as-original-image/34769840
def imshow_actual_size(im_data, img_name):
    dpi = 80
    
    height, width =  im_data.shape
    
    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)
    
    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    
    # Hide spines, ticks, etc.
    ax.axis('off')
    
    # Display the image.
    ax.imshow(im_data, interpolation='nearest')
    
    # Add something...
    #ax.annotate('Look at This!', xy=(590, 650), xytext=(500, 500),
# =============================================================================
#     ax.annotate('Look at This!', xy=(100, 100), xytext=(70, 70),
#                 color='cyan', size=24, ha='right',
#                 arrowprops=dict(arrowstyle='fancy', fc='cyan', ec='none'))
# =============================================================================
    
    # Ensure we're displaying with square pixels and the right extent.
    # This is optional if you haven't called `plot` or anything else that might
    # change the limits/aspect.  We don't need this step in this case.
    ax.set(xlim=[0, width], ylim=[height, 0], aspect=1)
    
    fig.savefig(img_name, dpi=dpi, transparent=True)
    plt.show()   

# Imshow complex images
# https://stackoverflow.com/questions/17044052/mathplotlib-imshow-complex-2d-array
def colorize(z):
    n, m = z.shape
    c = np.zeros((n, m, 3))
    c[np.isinf(z)] = (1.0, 1.0, 1.0)
    c[np.isnan(z)] = (0.5, 0.5, 0.5)

    idx = ~(np.isinf(z) + np.isnan(z))
    A = (np.angle(z[idx]) + np.pi) / (2*np.pi)
    A = (A + 0.5) % 1.0
    B = 1.0/(1.0 + abs(z[idx])**0.3)
    c[idx] = [hls_to_rgb(a, b, 0.8) for a, b in zip(A, B)]
    return c
    
# load image to test
img = Image.open('chris.jpg')
img_np = np.array(img)
img_tensor = Fv.to_tensor(img)[0:3, :, :][None]


# https://github.com/kuangliu/pytorch-cifar/blob/master/utils.py
def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std



def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                nn.init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant(m.weight, 1)
            nn.init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal(m.weight, std=1e-3)
            if m.bias:
                nn.init.constant(m.bias, 0)


def show_filter(data):
    fig, (ax) = plt.subplots(1, 1, figsize=[8, 8])
    im = ax.imshow(data,cmap='Spectral',vmin=0,vmax=1)
    plt.subplots_adjust(hspace=0.275)
    #plt.savefig('inset_cbar.png',dpi=100)
    plt.show()    
    
    
