import sys
sys.path.append("../model/")


from model_test import Scatt_OneOrder, Scatt_TwoOrder

from torch.nn import functional as F
from torchvision.transforms import functional as Fv
from PIL import Image
import pandas as pd
import torchvision

from matplotlib import pyplot as plt
import time

t0 = time.time()

#img = Image.open('../trump.jpg')
img = Image.open('../chris.jpeg')
img_tensor = Fv.to_tensor(img)
x = img_tensor[None]

lp = torch.zeros([1, 7, 7])
hp_real = torch.zeros(8, 7, 7)
hp_imag = torch.zeros(8, 7, 7)

# =============================================================================
# lp[0] = torch.tensor(pd.read_csv('csv/filter_random_phi.csv').values).data
# for i in range(8):
#     hp_real[i] = torch.tensor(pd.read_csv('csv/filter_random_psi_real_{}.csv'.format(i)).values).data
#     hp_imag[i] = torch.tensor(pd.read_csv('csv/filter_random_psi_imag_{}.csv'.format(i)).values).data
# =============================================================================
    
    
lp[0] = torch.tensor(pd.read_csv('../results/csv/filter_cifar_phi.csv').values).data
for i in range(8):
    hp_real[i] = torch.tensor(pd.read_csv('../results/csv/filter_cifar_psi_real_{}.csv'.format(i)).values).data
    hp_imag[i] = torch.tensor(pd.read_csv('../results/csv/filter_cifar_psi_imag_{}.csv'.format(i)).values).data
from model_test import Scatt_OneOrder, Scatt_TwoOrder

from torch.nn import functional as F
from torchvision.transforms import functional as Fv
from PIL import Image
import pandas as pd
import torchvision

from matplotlib import pyplot as plt
import time

t0 = time.time()

#img = Image.open('../trump.jpg')
img = Image.open('../chris.jpeg')
img_tensor = Fv.to_tensor(img)
x = img_tensor[None]

lp = torch.zeros([1, 7, 7])
hp_real = torch.zeros(8, 7, 7)
hp_imag = torch.zeros(8, 7, 7)

# =============================================================================
# lp[0] = torch.tensor(pd.read_csv('csv/filter_random_phi.csv').values).data
# for i in range(8):
#     hp_real[i] = torch.tensor(pd.read_csv('csv/filter_random_psi_real_{}.csv'.format(i)).values).data
#     hp_imag[i] = torch.tensor(pd.read_csv('csv/filter_random_psi_imag_{}.csv'.format(i)).values).data
# =============================================================================
    
    
lp[0] = torch.tensor(pd.read_csv('../results/csv/filter_cifar_phi.csv').values).data
for i in range(8):
    hp_real[i] = torch.tensor(pd.read_csv('../results/csv/filter_cifar_psi_real_{}.csv'.format(i)).values).data
    hp_imag[i] = torch.tensor(pd.read_csv('../results/csv/filter_cifar_psi_imag_{}.csv'.format(i)).values).data

C = 3
    
phi = lp[:, None].repeat(C,1,1,1)
psi_real = hp_real[:, None].repeat(C,1,1,1)
psi_imag = hp_imag[:, None].repeat(C,1,1,1)

# =============================================================================
# print(phi.shape)
# print(psi_real.shape)
# print(psi_imag.shape)
# =============================================================================

#model = Scatt_OneOrder(8,3,2)
model = Scatt_TwoOrder(8,3,[2,2])
model.phi.weight.data = phi
model.psi_real.weight.data = psi_real
model.psi_imag.weight.data = psi_imag
print("=====================================")

# =============================================================================
# y = model(x)
# for i in range(27):
#     name = "out/plt/chris_cifar_{}.png".format(i)
#     #torchvision.utils.save_image(y[0,i:i+1], name)
#     plt.imshow(y[0,i].detach().cpu().numpy())
#     plt.savefig(name)
#     plt.show()
# =============================================================================

# Order 2
y = model(x)
for i in range(243):
    name = "out/order2/cifar/chris_cifar_{}.png".format(i)
    #torchvision.utils.save_image(y[0,i:i+1], name)
    plt.imshow(y[0,i].detach().cpu().numpy())
    plt.savefig(name)
    plt.show()  
    
tFinnish = time.time() - t0
print("time: ", tFinnish)
C = 3
    
phi = lp[:, None].repeat(C,1,1,1)
psi_real = hp_real[:, None].repeat(C,1,1,1)
psi_imag = hp_imag[:, None].repeat(C,1,1,1)

# =============================================================================
# print(phi.shape)
# print(psi_real.shape)
# print(psi_imag.shape)
# =============================================================================

#model = Scatt_OneOrder(8,3,2)
model = Scatt_TwoOrder(8,3,[2,2])
model.phi.weight.data = phi
model.psi_real.weight.data = psi_real
model.psi_imag.weight.data = psi_imag
print("=====================================")

# =============================================================================
# y = model(x)
# for i in range(27):
#     name = "out/plt/chris_cifar_{}.png".format(i)
#     #torchvision.utils.save_image(y[0,i:i+1], name)
#     plt.imshow(y[0,i].detach().cpu().numpy())
#     plt.savefig(name)
#     plt.show()
# =============================================================================

# Order 2
y = model(x)
for i in range(243):
    name = "out/order2/cifar/chris_cifar_{}.png".format(i)
    #torchvision.utils.save_image(y[0,i:i+1], name)
    plt.imshow(y[0,i].detach().cpu().numpy())
    plt.savefig(name)
    plt.show()  
    
tFinnish = time.time() - t0
print("time: ", tFinnish)
