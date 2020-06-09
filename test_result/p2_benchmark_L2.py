from libcore import *
import time

#from kymatio.torch import Scattering2D
from kymatio import Scattering2D
#device = "cuda:0"
device = "cpu"

lp = torch.zeros([1, 7, 7])
hp_real = torch.zeros(8, 7, 7)
hp_imag = torch.zeros(8, 7, 7)

lp[0] = torch.tensor(pd.read_csv('filters/filter_lowpass.csv').values).data

for i in range(8):
    hp_real[i] = torch.tensor(pd.read_csv('filters/real_{}.csv'.format(i)).values).data
    hp_imag[i] = torch.tensor(pd.read_csv('filters/imag_{}.csv'.format(i)).values).data
    
lp_filter = torch.cat([lp[:, None], lp[:, None], lp[:, None]], dim=0)
real_filter = torch.cat([hp_real[:, None], hp_real[:, None], hp_real[:, None]], dim=0)
imag_filter = torch.cat([hp_imag[:, None], hp_imag[:, None], hp_imag[:, None]], dim=0)


class DynConv_Order1(nn.Module):
    def __init__(self, J, K, S):
        # Cin : so kenh vao
        # C : tuong ung voi J trong scattering
        
        super(DynConv_Order1, self).__init__()
        
        # Dynamic Channel 3, 6, 8, .... ~ J in scattering
        self.J = J
        self.S = S
        
        self.kernel_size = K
        
        self.lp_conv = nn.Conv2d(1, 1, K, S, padding=(int)((self.kernel_size-1)/2), bias=False)
        
        self.real_conv = nn.Conv2d(1, self.J, K, S, padding=(int)((self.kernel_size-1)/2), bias=False)
        self.imag_conv = nn.Conv2d(1, self.J, K, S, padding=(int)((self.kernel_size-1)/2), bias=False)
    
    def forward(self, x):     
        B, Cin, W, H = x.shape
        x = x.view(-1, 1, W, H)
        
        lp = self.lp_conv(x)
        
        real = self.real_conv(x)
        imag = self.imag_conv(x)
        
        hp = torch.sqrt(real**2 + imag**2)
        
        x = torch.cat([lp, hp], dim=1)
        
        return x.view(-1, (1+self.J)*Cin, x.shape[2], x.shape[3])

J = 8    
K = 7
B, C, W, H = 256, 3, 32, 32

# Scatter order 2
t0 = time.time()
order = 2
model = nn.Sequential(OrderedDict([
        ('order1', DynConv_Order1(J, K, S=2)),
        ('order2', DynConv_Order1(J, K, S=2))
        ]))
model.to(device)
    
for i in range(order):
    model[i].lp_conv.weight.data = lp[:, None].to(device)
    model[i].real_conv.weight.data = hp_real[:, None].to(device)
    model[i].imag_conv.weight.data = hp_imag[:, None].to(device)
t_model_cnn  = time.time() - t0    


# time for initialing scattering model
t0 = time.time()
#scatter = Scattering2D(J=2, shape=(W,H)).to(device)
#scatter = Scattering2D(J=2, shape=(W,H)).cuda()
scatter = Scattering2D(J=2, shape=(W,H))
t_model_scatter =time.time() - t0

print("init time, scatter: {}, CNN: {} ".format(t_model_scatter, t_model_cnn))


# Load data      
normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])

train_loader = torch.utils.data.DataLoader(
  datasets.CIFAR10('root/', train=False, download=True,
                             transform=transforms.Compose([
                               transforms.ToTensor(),
                               normalize
                             ])),
  batch_size=B, shuffle=True, num_workers=2)


import pandas as pd
out_dir = 'metric/'
metric_name = 'benchmark_{}.csv'.format(device)
benchmask_path=os.path.join(out_dir, metric_name)
    
col_epoch = []
col_scatter = []
col_our = []
epochs = 2
for epoch in range(epochs):
    t_sum_cnn, t_sum_scatter = 0,0
    for i, (data, label) in enumerate(train_loader):
        
        data = data.to(device)
        
        # Convolution
        t0 = time.time()
        y_cnn = model(data)
        t_cnn = time.time() - t0
        t_sum_cnn += t_cnn
        #print('cnn: ', y_cnn.shape)
        
        # Scattering
        t0 = time.time()
        y_scatter = scatter(data)
        t_scatter = time.time() - t0
        t_sum_scatter += t_scatter
        #print('scatter: ', y_scatter.shape)
    
    col_epoch.append(epoch+1)
    col_scatter.append(t_sum_scatter)
    col_our.append(t_sum_cnn)
    
    print("{}. epoch {}: scatter: {}, conv: {} ".format(device, epoch+1, t_sum_scatter, t_sum_cnn))
    
df = pd.DataFrame()    
df["Epoch"] = col_epoch
df["Scatter"] = col_scatter
df["My model"] = col_our
df.to_csv(benchmask_path, index = False)
