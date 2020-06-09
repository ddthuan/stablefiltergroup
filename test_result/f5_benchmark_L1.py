from libcore import *
import time

# for kymat version 0.2
#from kymatio.torch import Scattering2D

# version 0.1
from kymatio import Scattering2D

#device = "cuda:0"
device = "cpu"

# =============================================================================
# B, C, W, H = 2, 3, 1024, 1024
# x = torch.randn(B,C,W,H, device=device)
# =============================================================================

# =============================================================================
# img_name = "chris"
# img = Image.open("{}.jpeg".format(img_name))
# img_np = np.array(img)
# img_tensor = torchvision.transforms.functional.to_tensor(img)[0:3][None]
# B, C, W, H = img_tensor.shape
# x = img_tensor.detach().to(device)
# print(x.shape)
# =============================================================================

B, C, W, H = 256, 3, 32, 32

x = torch.randn(B,C,W,H, device=device)

# time for init Convolutional model 
t0 = time.time()
conv = torch.nn.Conv2d(C, 9*C, 3, padding=1, stride=2, bias=False)
conv.to(device)
t_model_conv = time.time()-t0

# time for initialing scattering model
t0 = time.time()
#scatter = Scattering2D(J=1, shape=(W,H)).to(device)
#scatter = Scattering2D(J=1, shape=(W,H)).cuda()
scatter = Scattering2D(J=1, shape=(W,H))
t_model_scatter =time.time() - t0

print("init time, scatter: {}, conv: {} ".format(t_model_scatter, t_model_conv))

# Load data      
normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])

train_loader = torch.utils.data.DataLoader(
  datasets.CIFAR10('root/', train=False, download=True,
                             transform=transforms.Compose([
                               transforms.ToTensor(),
                               normalize
                             ])),
  batch_size=B, shuffle=True, num_workers=6)


import pandas as pd
out_dir = 'metric/'
metric_name = 'benchmark_L1_{}.csv'.format(device)
benchmask_path=os.path.join(out_dir, metric_name)
    
col_epoch = []
col_scatter = []
col_our = []


epochs = 10
for epoch in range(epochs):
    t_sum_conv, t_sum_scatter = 0,0
    for i, (data, label) in enumerate(train_loader):
        
        data = data.to(device)
        
        # Convolution
        t0 = time.time()
        y_conv = conv(data)
        t_conv = time.time() - t0
        t_sum_conv += t_conv
        
        # Scattering
        t0 = time.time()
        y_scatter = scatter(data)
        t_scatter = time.time() - t0
        t_sum_scatter += t_scatter
        
    print("{}. epoch {}: scatter: {}, conv: {} ".format(device, epoch+1, t_sum_scatter, t_sum_conv))
    
    
df = pd.DataFrame()    
df["Epoch"] = col_epoch
df["Scatter"] = col_scatter
df["My model"] = col_our
df.to_csv(benchmask_path, index = False)

    
