from libscat import *
from p2_model import PriorNet_Basic, PriorNet_Complex
from p2_test_data import img_tensor, W1, W2


model = PriorNet_Complex()
model.to("cuda:0")
model_name = model.__class__.__name__

dsName = ['random', 'cifar10', 'tinyimage']

for ds in dsName:
    PATH = './model/{}_{}.pth'.format(model_name, ds)
    
    # Load pre-traing model
    model.load_state_dict(torch.load(PATH))
    model.eval()
    
    # Du lieu da gan trong so (weight)
    img_out = model(img_tensor.cuda())
    
    for i in range(9):
        img_name = './img/multi/{}_{}.jpg'.format(ds, i)
        imshow_actual_size(img_out[0, i].detach().cpu().numpy(), img_name)
