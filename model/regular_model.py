
# the equivalent output structure with level 2 scattering transform
class ModelLevel2(nn.Module):
    def __init__(self):
       super(ModelLevel2, self).__init__()
       self.conv1 = nn.Conv2d(1,9,5,padding=2)
       self.relu1 = nn.ReLU(inplace=True)
       self.pool1 = nn.MaxPool2d(kernel_size=(2,2))
       
       self.conv2 = nn.Conv2d(1,9,5,padding=2)
       self.relu2 = nn.ReLU(inplace=True)
       self.pool2 = nn.MaxPool2d(kernel_size=(2,2))
       
    def forward(self, x):
        x = x.view(-1, 1, x.shape[2], x.shape[3])
        out = self.pool1(self.relu1(self.conv1(x)))
        
        out = out.view(-1, 1, out.shape[2], out.shape[3])
        out = self.pool2(self.relu2(self.conv2(out)))
        
        out = out.view(-1, 81*C, out.shape[2], out.shape[3])
        return out
