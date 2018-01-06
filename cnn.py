import torch
from torchvision import transforms, datasets
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import _addindent
from ImagesFolder import TrainFolder




# def show_image(image):
#     """Show image"""
#     print(image)
#     plt.imshow(image)
#
# #4 is good
# def show_exampleImages(numImages,dataset):
#     images =np.random.choice(len(dataset),numImages)
#     images=[0,1,2,3]
#     for i in range(numImages):
#         sample,_= dataset[images[i]]
#
#         ax = plt.subplot(2, numImages, i + 1)
#         plt.tight_layout()
#         ax.set_title('Color #{}'.format(i))
#         show_image(sample[0])
#         ax = plt.subplot(2, numImages, numImages+i + 1)
#         show_image(sample[1])
#
#         ax.set_title('Gray #{}'.format(i))
#         ax.axis('off')
#         if i == numImages-1:
#             plt.show()
#             break

# show_exampleImages(4,train_set)

def torch_summarize(model, show_weights=True, show_parameters=True):
    """Summarizes torch model by showing trainable parameters and weights."""
    tmpstr = model.__class__.__name__ + ' (\n'
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params and weights
        if type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential
        ]:
            modstr = torch_summarize(module)
        else:
            modstr = module.__repr__()
        modstr = _addindent(modstr, 2)

        params = sum([np.prod(p.size()) for p in module.parameters()])
        weights = tuple([tuple(p.size()) for p in module.parameters()])

        tmpstr += '  (' + key + '): ' + modstr
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr +=  ', parameters={}'.format(params)
        tmpstr += '\n'

    tmpstr = tmpstr + ')'
    return tmpstr

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3,stride=1,padding=1)
        self.bn1 = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 10, kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(10)
        self.conv3 = nn.Conv2d(10, 10, kernel_size=3,stride=1,padding=1)
        self.bn3 = nn.BatchNorm2d(10)
        self.conv4 = nn.Conv2d(10, 2, kernel_size=3,stride=1,padding=1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        return x

class AutoNet(nn.Module):
    def __init__(self):
        super(AutoNet, self).__init__()
        

        ## ENCODER ##

        self.conv1 = nn.Conv2d(1, 8, kernel_size=3,stride=1,padding=1) # 8 x 224 x 224
        self.bn1 = nn.BatchNorm2d(8)
        self.mp1 = nn.MaxPool2d(2,2) # 8 x 112 x 112

        self.conv2 = nn.Conv2d(8, 16, kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.mp2 = nn.MaxPool2d(2,2) # 8 x 56 x 56


        self.conv3 = nn.Conv2d(16, 32, kernel_size=3,stride=1,padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.mp3 = nn.MaxPool2d(2,2) # 32 x 28 x 28

        self.conv4 = nn.Conv2d(32, 64, kernel_size=3,stride=1,padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.mp4 = nn.MaxPool2d(2,2) # 64 x 14 x 14

        ## DECODER ##

        self.conv5 = nn.Conv2d(64, 32, kernel_size=3,stride=1,padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.up5 = nn.Upsample(scale_factor=2, mode='nearest') # 32 x 28 x 28

        self.conv6 = nn.Conv2d(32, 16, kernel_size=3,stride=1,padding=1)
        self.bn6 = nn.BatchNorm2d(16)        
        self.up6 = nn.Upsample(scale_factor=2, mode='nearest') # 16 x 56 x 56

        self.conv7 = nn.Conv2d(16, 8, kernel_size=3,stride=1,padding=1)
        self.bn7 = nn.BatchNorm2d(8)        
        self.up7 = nn.Upsample(scale_factor=2, mode='nearest') # 8 x 112 x 112

        self.conv8 = nn.Conv2d(8, 4, kernel_size=3,stride=1,padding=1)
        self.bn8 = nn.BatchNorm2d(4)
        self.up8 = nn.Upsample(scale_factor=2, mode='nearest') # 4 x 224 x 224

        self.conv9 = nn.Conv2d(4, 2, kernel_size=3, stride=1,padding=1) # 2 x 224 x 224
        self.sig = nn.Sigmoid()

    def forward(self, x):
        #encode
        x = self.mp1(F.relu(self.bn1(self.conv1(x))))
        x = self.mp2(F.relu(self.bn2(self.conv2(x))))
        x = self.mp3(F.relu(self.bn3(self.conv3(x))))
        x = self.mp4(F.relu(self.bn4(self.conv4(x))))
        #decode
        x = self.up5(F.relu(self.bn5(self.conv5(x))))
        x = self.up6(F.relu(self.bn6(self.conv6(x))))
        x = self.up7(F.relu(self.bn7(self.conv7(x))))
        x = self.up8(F.relu(self.bn8(self.conv8(x))))

        x = self.sig(self.conv9(x))
        # x = self.conv4(x)
        return x

#net = Net()
