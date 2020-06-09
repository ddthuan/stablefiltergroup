from libscat import *
from kymatio import Scattering2D

import time

from p2_test_data import img_tensor, W1, W2

# =============================================================================
# W1, W2 = 1536, 2048
# img = Image.open('japan.jpeg')
# img_np = np.array(img)
# img_tensor = Fv.to_tensor(img)[1:2, :, :][None]
# =============================================================================

#print(img_tensor.shape)

device = torch.device('cuda:0')

class ScatNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scat = Scattering2D(J=2, shape=(W1, W2), L=8).cuda()
        
    def forward(self, x):
        x = self.scat(x)
        return x.view(x.shape[0], -1, x.shape[3], x.shape[4])

#x = torch.nn.functional.interpolate(img_tensor, (W1, W2))
#x = img_tensor.detach()

# =============================================================================
# x_img = Image.open('h2.jpg')
# x_tensor = Fv.to_tensor(x_img)[None]
# x = torch.nn.functional.interpolate(x_tensor,(512,768))
# =============================================================================

x = torch.ones(1,1,512,768)
B,C,W1,W2 = x.shape
x = x.cuda()

time_start = time.time()
model_scat = ScatNet()
model_scat.to(device)
out = model_scat(x)
time_run = time.time() - time_start
print(time_run)
print(out.shape)

for i in range(81):
    img_name = 'hotel/energy_{}.jpg'.format(i)    
    torchvision.utils.save_image(out[0,i:i+1].detach(), img_name)

#plt.imshow(x[0,0].detach().cpu().numpy())
#plt.show()
