from libscat import *
from p2_model import PriorNet_Complex, PriorNet_Invariant
from p2_test_data import img_tensor, W1, W2

from kymatio import Scattering2D

from torchsummary import summary

def get_data(in_size, data_dir, val_only=False, batch_size=128,
             trainsize=-1, perturb=True, num_workers=0, iter_size=1,
             distributed=False, pin_memory=False):
    """ Provides a pytorch loader to load in imagenet
    Args:
        in_size (int): the input size - can be used to scale the spatial size
        data_dir (str): the directory where the data is stored
        val_only (bool): Whether to load only the validation set
        batch_size (int): batch size for train loader. the val loader batch
            size is always 100
        class_sz (int): size of the training set. can be used to subsample it
        seed (int): random seed for the loaders
        perturb (bool): whether to do data augmentation on the training set
        num_workers (int): how many workers to load data
        iter_size (int):
    """
    valdir = os.path.join(data_dir, 'val2')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if not os.path.exists(valdir):
        raise ValueError(
            'Could not find the val2 folder in the Tiny Imagenet directory.'
            'Have you run the prep_tinyimagenet.py script in '
            'scatnet_learn.data?')

    # Get the test loader
    transform_test = transforms.Compose([
        transforms.CenterCrop(in_size),
        transforms.ToTensor(),
        normalize
    ])
    testloader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transform_test),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory)

    if val_only:
        trainloader = None
    else:
        if 0 < trainsize < 100000:
            class_sz = trainsize // 200
            traindir = os.path.join(data_dir, 'train{}'.format(class_sz))
        else:
            traindir = os.path.join(data_dir, 'train')
        if not os.path.exists(traindir):
            subsample(data_dir, class_sz)
            assert os.path.exists(traindir)
        # Get the train loader
        if perturb:
            transform_train = transforms.Compose([
                transforms.RandomCrop(in_size, padding=8),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            transform_train = transforms.Compose([
                transforms.CenterCrop(in_size),
                transforms.ToTensor(),
                normalize
            ])

        trainset = datasets.ImageFolder(
            traindir, transform_train)

        if distributed:
            trainsampler = torch.utils.data.distributed.DistributedSampler(
                trainset)
        else:
            trainsampler = None

        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size // iter_size,
            shuffle=(trainsampler is None), num_workers=num_workers,
            pin_memory=pin_memory, sampler=trainsampler)

    sys.stdout.write("| loaded tiny imagenet")
    return trainloader, testloader


model = PriorNet_Complex()
model.to("cuda:0")
model_name = model.__class__.__name__
PATH = './model/{}.pth'.format(model_name)

model.load_state_dict(torch.load(PATH))
model.eval()

model_new = PriorNet_Invariant()
#model_new.to("cuda:0")

# dong bang
model_new.lowpass.weight.requires_grad_(False)
model_new.real.weight.requires_grad_(False)
model_new.imag.weight.requires_grad_(False)

# Gan trong so cho model    
model_new.lowpass.weight.data = model.lowpass.weight.detach().cpu()
model_new.real.weight.data = model.real.weight.detach().cpu()
model_new.imag.weight.data = model.imag.weight.detach().cpu()
model_name_new = model_new.__class__.__name__

#print(summary(model_new, (1, 255, 255)))

def invariant_L2(model_Invariant_L1, x):
    B, C = x.shape[0], x.shape[1]
    x = x.view(-1, 1, x.shape[2], x.shape[3])
    y= model_Invariant_L1(x)
    y = y.view(-1,1, y.shape[2], y.shape[3])
    y = model_Invariant_L1(y)
    y = y.view(B, -1, y.shape[2], y.shape[3])
    return y



class ScatNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scat = Scattering2D(J=2, shape=(64, 64), L=8).cuda()
        
    def forward(self, x):
        x = self.scat(x)
        return x.view(x.shape[0], -1, x.shape[3], x.shape[4])

scatter = ScatNet()


# Load Tiny Imagetnet 
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
data_transforms = {
    'train': transforms.Compose([
        #transforms.RandomCrop(64, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #normalize
    ]),
    'val': transforms.Compose([
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        normalize
    ]),
}
    
data_dir = 'tiny-imagenet-200'

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=256,
                                             shuffle=False, num_workers=6)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

data = []
for i, x in enumerate(dataloaders['val']):
    if i == 8:
        break
    data.append(x[0])
data = torch.cat(data, dim=0)
print(data.shape)
data_cuda = data.clone().cuda()



# =============================================================================
# # only load for testing
# data_dir = 'tiny-imagenet-200/val'
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])
# 
# transform_test = transforms.Compose([
#     transforms.CenterCrop(64),
#     transforms.ToTensor(),
#     normalize
# ])
#     
# testloader = torch.utils.data.DataLoader(
#     datasets.ImageFolder(data_dir, transform_test),
#     batch_size=256, shuffle=False
#     )
# 
# data = []
# for i, x in enumerate(testloader):
#     if i == 10:
#         break
#     data.append(x[0])
# data = torch.cat(data, dim=0)
# print(data.shape)
# data_cuda = data.clone().cuda()
# =============================================================================


# Data is created randomly
#data = torch.randn(1000, 3, 256, 256)
data_L2 = invariant_L2(model_new, data)
data_scatt = scatter(data_cuda)

# add Noise
NOISE = 0.7 * torch.randn_like(data)
data_noise = data + NOISE
data_noise_L2 = invariant_L2(model_new, data_noise)
d_noise = torch.mean((data_noise - data)**2)
d_noise_l2 = torch.mean((data_noise_L2 - data_L2)**2)

percent = ((NOISE**2).sum()/(data**2).sum())*100

data_noise_cuda = data_noise.clone().cuda()
data_noise_scatter = scatter(data_noise_cuda)
d_scatter = torch.mean((data_noise_scatter - data_scatt)**2)
#print('Distance: {}, noise(our) : {}'.format(d_noise, d_noise_l2))
print('noise size - INV {}: , scatter: P{}'.format(data_noise_L2.shape, data_noise_scatter.shape))
print('Distance: {}, percent: {}, noise(our) : {}, scatter: {}'.format(d_noise, percent, d_noise_l2, d_scatter.detach().cpu()))

# add shifts
data_shifts = torch.roll(data, (2, 2), dims=(2,3))
data_shifts_L2 = invariant_L2(model_new, data_shifts)
d_shifts = torch.mean((data_shifts - data)**2)
d_shifts_l2 = torch.mean((data_shifts_L2 - data_L2)**2)

data_shifts_cuda = data_shifts.clone().cuda()
data_shifts_scatter = scatter(data_shifts_cuda)
d_shifts_scatter = torch.mean((data_shifts_scatter - data_scatt)**2)
#print('Distance: {}, shifts std: {}'.format(d_shifts, d_shifts_l2))
print('shift size - INV {}: , scatter: P{}'.format(data_shifts_L2.shape, data_shifts_scatter.shape))
print('Distance: {}, shifts (our): {}, scatter: {}'.format(d_shifts, d_shifts_l2, d_shifts_scatter.detach().cpu()))

# add Deformation
import elasticdeform
σ = 3
data_deformation = torch.tensor(elasticdeform.deform_random_grid(data.numpy(), sigma=σ, axis=(2,3)))
data_deformation_L2 = invariant_L2(model_new, data_deformation)
d_deformation = torch.mean((data_deformation-data)[:, :, 2*σ:-2*σ]**2)
d_deformation_L2 = torch.mean((data_deformation_L2-data_L2)[:, :, 2*σ:-2*σ]**2)


data_deformation_cuda = data_deformation.clone().cuda()
data_deformation_scatter = scatter(data_deformation_cuda)
d_deformation_scatter = torch.mean((data_deformation_scatter - data_scatt)**2)
#print('Distance: {}, deformation std: {}'.format(d_deformation, d_deformation_L2))
print('deformation size - INV {}: , scatter: P{}'.format(data_deformation_L2.shape, data_deformation_scatter.shape))
print('Distance: {}, deformation (our): {}, scatter: {} '.format(d_deformation, d_deformation_L2, d_deformation_scatter.detach().cpu()))





















