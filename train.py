import os
import traceback

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
import numpy as np

from ImagesFolder import TrainFolder
from cnn import Net
have_cuda = torch.cuda.is_available()
epochs = 1

original_transform = transforms.Compose([
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    #transforms.ToTensor()
])

color_dir = "train"#/train"
gray_dir = "grayscale"#/train"
train_set = TrainFolder(color_dir,original_transform )
train_set_size = len(train_set)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)
color_model = Net()
if os.path.exists('cnn_params.pkl'):
    color_model.load_state_dict(torch.load('cnn_params.pkl'))
if have_cuda:
    color_model.cuda()
optimizer = optim.Adadelta(color_model.parameters())


def train(epoch):
    color_model.train()

    try:
        for batch_idx, (data, classes) in enumerate(train_loader):
            messagefile = open('./message.txt', 'a')

            original_img = data[0].unsqueeze(1).float()
            img_ab = data[1].float()
            if have_cuda:
                original_img = original_img.cuda()
                img_ab = img_ab.cuda()
                classes = classes.cuda()
            original_img = Variable(original_img)
            img_ab = Variable(img_ab)
            classes = Variable(classes)
            optimizer.zero_grad()
            output = color_model(original_img)
            ems_loss = torch.pow((img_ab - output), 2).sum() / torch.from_numpy(np.array(list(output.size()))).prod()
            loss = ems_loss
            lossmsg = 'loss: %.9f\n' % (loss.data[0])
            messagefile.write(lossmsg)
            ems_loss.backward(retain_variables=True)
            optimizer.step()
            if batch_idx % 20 == 0:
                message = 'Train Epoch:%d\tPercent:[%d/%d (%.0f%%)]\tLoss:%.9f\n' % (
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data[0])
                messagefile.write(message)
                torch.save(color_model.state_dict(), 'cnn_params.pkl')

                print('Train Epoch: {}[{}/{}({:.0f}%)]\tLoss: {:.9f}\n'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.data[0]))
            messagefile.close()
    except Exception:
        logfile = open('log.txt', 'w')
        logfile.write(traceback.format_exc())
        logfile.close()
    finally:
        torch.save(color_model.state_dict(), 'cnn_params.pkl')


for epoch in range(1, epochs + 1):
    train(epoch)
