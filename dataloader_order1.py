from libscat import *
from kymatio import Scattering2D


# =============================================================================
# # ==================== Create Dataset randomly ==============================
# W1, W2 = 512, 128
# 
# class ScatNet(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.scat = Scattering2D(J=1, shape=(W1, W2), L=8).cuda()
#         
#     def forward(self, x):
#         x = self.scat(x)
#         return x.view(x.shape[0], -1, x.shape[3], x.shape[4])
#     
# 
# # use the scattering-model for creating the test-data
# model_scat = ScatNet()
# 
# # Create data for training
# # random data --> invariance with data
# # custome size (w, h)  --> invariance with size
# class CreateData(torch.utils.data.Dataset):
#     def __init__(self, n_sample, W1, W2):
#         self.inputs = torch.randn(n_sample, 1, W1, W2).cuda()
#         self.target = model_scat(self.inputs)
#         
#     def __len__(self):
#         return len(self.inputs)
#     
#     def __getitem__(self, idx):
#         return self.inputs[idx], self.target[idx]
# =============================================================================




# =============================================================================
# ############################################### CIFAR10 ##################
# random_seed = 1
# torch.manual_seed(random_seed)
# 
# normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
#                                      std=[0.2023, 0.1994, 0.2010])
# 
# train_loader = torch.utils.data.DataLoader(
#   datasets.CIFAR10('root/', train=False, download=True,
#                              transform=transforms.Compose([
#                                transforms.ToTensor(),
#                                normalize
#                              ])),
#   batch_size=128, shuffle=True)
# 
# 
# data_train = []
# for i, x in enumerate(train_loader):
#     data_train.append(x[0])
#     
# data_train = torch.cat(data_train, dim=0)
# 
# len_train = len(data_train)
# Bt = 100
# step_train = int(len_train/Bt)
# 
# class ScatNet_MNIST(torch.nn.Module):
#     def __init__(self):
#         super(ScatNet_MNIST, self).__init__()
#         self.scat = Scattering2D(J=1, shape=(32, 32), L=8).cuda()
#         
#     def forward(self, x):
#         x = self.scat(x)
#         return x.view(x.shape[0], -1, x.shape[3], x.shape[4])
# 
# model_catter = ScatNet_MNIST()
# model_catter.cuda()
#     
# class CreateDataScatter(torch.utils.data.Dataset):
#     def __init__(self, data, step):
#         self.inputs = data[0:10000].cuda()
#         self.target = model_catter(self.inputs)
#         
#     def __len__(self):
#         return len(self.inputs)
#     
#     def __getitem__(self, idx):
#         return self.inputs[idx], self.target[idx]
#     
# train_dataset  = CreateDataScatter(data_train[:, 1:2], step_train)
# train_scatter_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1)
# =============================================================================





# =============================================================================
# model = ScatNet_MNIST()
# x = torch.randn(1,1,32,32)
# y = model(x.cuda())
# print(y.shape)
# =============================================================================


# =============================================================================
# ############################################### MNIST ##################
# train_loader = torch.utils.data.DataLoader(
#   datasets.MNIST('root/', train=False, download=True,
#                              transform=transforms.Compose([
#                                transforms.ToTensor(),
#                                transforms.Normalize(
#                                  (0.1307,), (0.3081,))
#                              ])),
#   batch_size=128, shuffle=True)
# 
# data_train = []
# for i, x in enumerate(train_loader):
#     data_train.append(x[0])
#     
# data_train = torch.cat(data_train, dim=0)
# 
# len_train = len(data_train)
# Bt = 100
# step_train = int(len_train/Bt)
# 
# class ScatNet_MNIST(torch.nn.Module):
#     def __init__(self):
#         super(ScatNet_MNIST, self).__init__()
#         self.scat = Scattering2D(J=1, shape=(28, 28), L=8).cuda()
#         
#     def forward(self, x):
#         x = self.scat(x)
#         return x.view(x.shape[0], -1, x.shape[3], x.shape[4])
# 
# model_catter = ScatNet_MNIST()
# model_catter.cuda()
#     
# class CreateDataScatter(torch.utils.data.Dataset):
#     def __init__(self, data, step):
#         self.inputs = data[0:10000].cuda()
#         self.target = model_catter(self.inputs)
#         
#     def __len__(self):
#         return len(self.inputs)
#     
#     def __getitem__(self, idx):
#         return self.inputs[idx], self.target[idx]
#     
# train_dataset  = CreateDataScatter(data_train, step_train)
# train_scatter_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1)
# =============================================================================







# ######################## Tiny Imagetnet  ###################################
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomCrop(64, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]),
    'val': transforms.Compose([
        #transforms.CenterCrop(64),
        transforms.ToTensor(),
        normalize
    ]),
}
    
data_dir = 'tiny-imagenet-200'

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=256,
                                             shuffle=True, num_workers=6)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

data_train = []
data_train_lb = []
for i, x in enumerate(dataloaders['train']):
    data_train.append(x[0])
    data_train_lb.append(x[1])    
    
data_train = torch.cat(data_train, dim=0)
data_train_lb = torch.cat(data_train_lb, dim=0)

data_val = []
data_val_lb = []
for i, x in enumerate(dataloaders['val']):
    data_val.append(x[0])
    data_val_lb.append(x[1])
    
data_val = torch.cat(data_val, dim=0)
data_val_lb = torch.cat(data_val_lb, dim=0)


len_train = len(data_train)
len_val = len(data_val)
Bt = 2000
step_train = int(len_train/Bt)
step_val = int(len_val/Bt)
#print('train_len: {}, val_len: {}, step: {}'.format(len_train, len_val, step_train))


class ScatNet_TinyImage(torch.nn.Module):
    def __init__(self):
        super(ScatNet_TinyImage, self).__init__()
        self.scat = Scattering2D(J=1, shape=(64, 64), L=8).cuda()
        
    def forward(self, x):
        x = self.scat(x)
        return x.view(x.shape[0], -1, x.shape[3], x.shape[4])

model_catter = ScatNet_TinyImage()
model_catter.cuda()

class CreateDataScatter(torch.utils.data.Dataset):
    def __init__(self, data):
        self.inputs = data[0:10000].cuda()
        self.target = model_catter(self.inputs)
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.target[idx]
    
train_dataset  = CreateDataScatter(data_val[:, 1:2])
train_scatter_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1)





# =============================================================================
# out = invariant_L2(model_new, data_val[0:3000].cuda())
# print(out.shape)
# =============================================================================

# =============================================================================
# class CreateDataScatter(torch.utils.data.Dataset):
#     def __init__(self, data, label, step):
#         data_scatter = []
#         for i in range(step):
#             data_scatter.append(invariant_L2(model_new, data[i*Bt:(i+1)*Bt].cuda()).detach().cpu())
#         data_scatter = torch.cat(data_scatter, dim=0)
#         self.inputs = data_scatter
#         self.target = label
#         
#     def __len__(self):
#         return len(self.inputs)
#     
#     def __getitem__(self, idx):
#         return self.inputs[idx], self.target[idx]
#     
# train_dataset  = CreateDataScatter(data_train, data_train_lb, step_train)
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1000)
# for i, (data, target) in enumerate(train_loader):
#     print('train - scatter: ', data.shape)
#     #print('label: ', target)
#     
# val_dataset  = CreateDataScatter(data_val, data_val_lb, step_val)
# val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1000)
# for i, (data, target) in enumerate(val_loader):
#     print('val - scatter: ', data.shape)
#     #print('label: ', target)    
# =============================================================================




# =============================================================================
# for i, (data, target) in enumerate(train_scatter_loader):    
#     print(i)
#     print(data.shape, '-->data')
#     print(target.shape, '-->target', )
# =============================================================================
