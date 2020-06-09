import torch

import sys
sys.path.append("model/")
from model_test import Scatt_OneOrder

#model = Scatt_OneOrder(8,1,2)
model = Scatt_OneOrder(8,1,[2,2])
model.to("cuda:0")
model_name = model.__class__.__name__

# =============================================================================
# dsName = 'random'
# PATH = './model/Scatt_OneOrder_random_level1_Smooth_epsConstant.pth'
# =============================================================================

# =============================================================================
# dsName = 'cifar10'
# PATH = './model/Scatt_OneOrder_cifar10_level1_Smooth_best.pth'
# =============================================================================

# =============================================================================
# dsName = 'imagenet'
# PATH = './model/Scatt_OneOrder_restnet_level1_Smooth.pth'
# =============================================================================

# =============================================================================
# dsName = 'imagenet_smooth'
# PATH = './model/Scatt_TwoOrder_imagenet_level2_Smooth_200.pth'
# =============================================================================

dsName = 'imagenet_nonesmooth'
PATH = './model/Scatt_TwoOrder_imagenet_level2_None_Smooth_200.pth'


model.load_state_dict(torch.load(PATH))
model.eval()

# =============================================================================
# path_phi = './filters/order1_{}_phi.pt'.format(dsName)
# path_psi_real = './filters/order1_{}_psi_real.pt'.format(dsName)
# path_psi_imag = './filters/order1_{}_psi_imag.pt'.format(dsName)
# =============================================================================

path_phi = './filters/order2_{}_phi.pt'.format(dsName)
path_psi_real = './filters/order2_{}_psi_real.pt'.format(dsName)
path_psi_imag = './filters/order2_{}_psi_imag.pt'.format(dsName)

filter_phi = model.phi.weight.detach().cpu().data
filter_psi_real = model.psi_real.weight.detach().cpu().data
filter_psi_imag = model.psi_imag.weight.detach().cpu().data

torch.save(filter_phi, path_phi)
torch.save(filter_psi_real, path_psi_real)
torch.save(filter_psi_imag, path_psi_imag)

