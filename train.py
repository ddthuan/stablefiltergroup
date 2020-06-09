from libscat import *
import sys

sys.path.append("model/")
from model import Scatt_TwoOrder
from torch.optim import lr_scheduler

# apply for function torch.nn.function.conv2d() in case using GPU
torch.backends.cudnn.deterministic = True

device = torch.device('cuda:0')

# For loading dataset
#from dataloader_order1 import train_scatter_loader
from dataloader_order2 import train_scatter_loader
dataloader = train_scatter_loader

kernel_size = 7
#model = PriorNet_Complex_SizeKernel_Ex(kernel_size)
#model = PriorNet_Complex()
#model = Scatt_OneOrder(8,1,2)
model = Scatt_TwoOrder(8,1,[2,2])
#model = Model_Raw() 
#model = PriorNet_Complex_Im()
#model = PriorNet_Basic()

model.to(device)

# Loss function
criterion = torch.nn.MSELoss(reduction='sum')

# Optimizer
learning_rate = 1e-2
weight_delay = 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = weight_delay, eps=1e-7)
#optimizer = optim.Adam(to_optimize, lr = args.learning_rate_adam, betas=args.betas, eps=args.eps, weight_decay = args.weight_decay)    
# for random
#epochs = 500
    
epochs = 60
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[20, 30, 40, 50], gamma=0.1)

out_dir = 'metric/'

metric_name = 'loss_kernel_cifar100_level2_None_Smooth_200_{}.csv'.format(str(kernel_size))
iter_time_path=os.path.join(out_dir, metric_name)

loss_csv = []
eps_csv = []
epoch_csv = []

for epoch in range(epochs):
    epoch_csv.append(epoch)
    for i, (data, target) in enumerate(dataloader):    
        
        optimizer.zero_grad()
        pred = model(data)
        loss = criterion(pred.view(-1), target.view(-1))
        loss.backward()
        optimizer.step()
    scheduler.step()
    loss_csv.append(loss.detach().cpu().item())
    #eps_csv.append(model.eps.data.detach().cpu().item())
# =============================================================================
#         if i % 100 == 0:
#             print('loss: ', loss.item())
# =============================================================================
            
    print('finnish epoch: {}, loss: {}'.format(epoch, loss.detach().cpu()))
    
model_name = model.__class__.__name__
PATH = './model/{}_cifar100_level2_None_Smooth_200.pth'.format(model_name)

torch.save(model.state_dict(), PATH)

df=pd.DataFrame()
df['epoch'] = epoch_csv
df['loss'] = loss_csv
#df['eps'] = eps_csv
df.to_csv(iter_time_path,index=False)

#### Can thuc thi them do thi ham loss function.

