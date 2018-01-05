import os
import traceback
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
from PIL import Image
from ImagesFolder import TrainFolder
from cnn import Net
from skimage.color import rgb2lab, rgb2gray

epochs = 1

original_transform = transforms.Compose([
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    #transforms.ToTensor()
])

color_dir = "clean"#/train"
gray_dir = "grayscale"#/train"

color_model = Net()
modelParams="cnn_deep.pkl"
if os.path.exists(modelParams):
    color_model.load_state_dict(torch.load(modelParams))

def show_image(image):
    """Show image"""
    print(image)
    plt.imshow(image)

#4 is good
def show_exampleImages(numImages):
    images =np.random.choice(len(dataset),numImages)
    images=[0,1,2,3]
    for i in range(numImages):
        sample,_= dataset[images[i]]

        ax = plt.subplot(2, numImages, i + 1)
        plt.tight_layout()
        ax.set_title('Color #{}'.format(i))
        show_image(sample[0])
        ax = plt.subplot(2, numImages, numImages+i + 1)
        show_image(sample[1])

        ax.set_title('Gray #{}'.format(i))
        ax.axis('off')
        if i == numImages-1:
            plt.show()
            break
color_dir = "clean"#/train"
def displayImage(numImages):
    # get random images
    # convert them to grayscale and ab
    # pass grayscale through model
    # recombine with ab
    # display
    numfiles = sum(1 for f in os.listdir(color_dir) if os.path.isfile(os.path.join(color_dir, f)) and f[0] != '.')
    for i in range(numImages):
        currImage=random.choice(os.listdir("train/train"))
        img_original = Image.open("train/train/"+currImage)

        img_original = original_transform(img_original)
        img_original = np.asarray(img_original)

        img_lab = rgb2lab(img_original)
        img_lab = (img_lab + 128) / 255
        img_ab = img_lab[:, :, 1:3]
        img_ab = torch.from_numpy(img_ab.transpose((2, 0, 1)))
        img_original = rgb2gray(img_original)
        img_original = torch.from_numpy(img_original)


        img_original = img_original[1].unsqueeze(1).float()
        print(img_original.shape)
        img_original = Variable(img_original)
        #print(img_original)
    # img_ab = Variable(img_ab)
        output = color_model(img_original)
        print(output)
    # ems_loss = torch.pow((img_ab - output), 2).sum() / torch.from_numpy(np.array(list(output.size()))).prod()
displayImage(2)
