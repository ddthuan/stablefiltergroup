import torch
from torch import nn

C = 3 # input channel
L = 8 # number of angles
K = 7 # size of kernel
P = 3 # pading
S = 2 # stride 2 or [2,2], [1,2], ...

# the equivalent output structure with level 1 scattering transform with L=8
class ModelLevel1(nn.Module):
    def __init__(self):
        super(ModelLevel1, self).__init__()
        self.conv1 = nn.Conv2d(1,9,7, padding=(3,3))
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2))
        
    def forward(self, x):
        out = x.view(-1, 1, x.shape[2], x.shape[3])
        out = self.pool1(self.relu1(self.conv1(out)))
        out = out.view(-1,9*C,out.shape[2], out.shape[3])
        return out
   


# the equivalent output structure with level 2 scattering transform with L=8
class ModelLevel2(nn.Module):
    def __init__(self):
       super(ModelLevel2, self).__init__()
       self.conv1 = nn.Conv2d(1,9,7,padding=3)
       self.relu1 = nn.ReLU(inplace=True)
       self.pool1 = nn.MaxPool2d(kernel_size=(2,2))
       
       self.conv2 = nn.Conv2d(1,9,7,padding=3)
       self.relu2 = nn.ReLU(inplace=True)
       self.pool2 = nn.MaxPool2d(kernel_size=(2,2))
       
    def forward(self, x):
        x = x.view(-1, 1, x.shape[2], x.shape[3])
        out = self.pool1(self.relu1(self.conv1(x)))
        
        out = out.view(-1, 1, out.shape[2], out.shape[3])
        out = self.pool2(self.relu2(self.conv2(out)))
        
        out = out.view(-1, 81*C, out.shape[2], out.shape[3])
        return out
