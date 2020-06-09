from torch.nn import functional as F
from torchvision.transforms import functional as Fv
from PIL import Image
import pandas as pd
import torchvision

from matplotlib import pyplot as plt
import time

from kymatio.torch import Scattering2D
device = "cuda:0"

t0 = time.time()

#img = Image.open('../trump.jpg')
img = Image.open('../chris.jpeg')
img_tensor = Fv.to_tensor(img)
x = img_tensor[None]
x = x.cuda()

B,C,W,H = x.shape
model = Scattering2D(J=2, shape=(W,H), L=8)
model.to(device)
model.to(device)
y = model(x)
y = y.view(y.size(0), -1, y.size(3), y.size(4))

for i in range(243):
    name = "out/order2/scatter/chris_scatt_{}.png".format(i)
    #torchvision.utils.save_image(y[0,i:i+1].detach().cpu(), name)
    plt.imshow(y[0,i].detach().cpu().numpy())
    plt.savefig(name)
    plt.show()

tFinnish = time.time() - t0
print("time: ", tFinnish)

# =============================================================================
# R, G, B = torch.chunk(y, 3, dim=1)    
# for i in range(9):
#     name = "out/s_color_{}.png".format(i)        
#     
#     out = torch.stack([R[0,i], G[0,i], B[0,i]], dim=0)    
#     torchvision.utils.save_image(out.detach().cpu(), name)
#     
#     plt.imshow(Fv.to_pil_image(out.detach().cpu())) 
#     plt.show()
# =============================================================================
