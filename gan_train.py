#### SUNEEL ####
# NOTE: unfinished/ untested #

import os
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
import numpy as np

from ImagesFolder import TrainFolder
from gan import Generator, Discriminator
have_cuda = torch.cuda.is_available()
epochs = 3

original_transform = transforms.Compose([
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    #transforms.ToTensor()
])


g_iters = 10
d_iters = 10

color_dir = "train"#/train"
gray_dir = "grayscale"#/train"
train_set = TrainFolder(color_dir,original_transform )
train_set_size = len(train_set)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)

G = Generator()
D = Discriminator()

gParams = 'gan_g_params.pkl'
dParams = 'gan_d_params.pkl'

if os.path.exists(gParams):
    G.load_state_dict(torch.load(gParams))

if os.path.exists(dParams):
    D.load_state_dict(torch.load(dParams))

#load cuda
if have_cuda:
    G.cuda()
    D.cuda()


# Binary cross entropy loss and optimizer
criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0003)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0003)



def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def gan_train(epoch):
    G.train()
    D.train()
    try:
        for i, (images, classes) in enumerate(train_loader):
            # Build mini-batch dataset
            batch_size = images[0].size(0)

            messagefile = open('./message.txt', 'a')

            bw_image = images[0].unsqueeze(1).float()
            ab_image = images[1].float()

            bw_image = to_var(bw_image)
            ab_image = to_var(ab_image)
            classes = to_var(classes)

            # Create the labels which are later used as input for the BCE loss
            real_labels = to_var(torch.ones(batch_size))
            fake_labels = to_var(torch.zeros(batch_size))

            #============= Train the discriminator =============#
            # Compute BCE_Loss using real images where BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
            # Second term of the loss is always zero since real_labels == 1
            outputs = D(ab_image)
            d_loss_real = criterion(outputs, real_labels)
            real_score = outputs
            
            # Compute BCELoss using fake images
            # First term of the loss is always zero since fake_labels == 0
            # z = to_var(torch.randn(batch_size, 64))
            fake_images = G(bw_image)
            outputs = D(fake_images)
            d_loss_fake = criterion(outputs, fake_labels)
            fake_score = outputs

            # n = np.array(outputs.size(), dtype='int64')
            # # print n.dtype
            # ems_loss = torch.pow((ab_image - outputs), 2).sum() / torch.from_numpy(n).prod()
            # loss = ems_loss
            # lossmsg = 'loss: %.9f\n' % (loss.data[0])
            # messagefile.write(lossmsg)
            # ems_loss.backward(retain_variables=True)

            # Backprop + Optimize
            d_loss = d_loss_real + d_loss_fake
            D.zero_grad()
            d_loss.backward()
            d_optimizer.step()
            
            #=============== Train the generator ===============#
            # Compute loss with fake images
            # z = to_var(torch.randn(batch_size, 64))
            fake_images = G(bw_image) #again, consider removing
            outputs = D(fake_images)
            
            # We train G to maximize log(D(G(z)) instead of minimizing log(1-D(G(z)))
            # For the reason, see the last paragraph of section 3. https://arxiv.org/pdf/1406.2661.pdf
            g_loss = criterion(outputs, real_labels)
            
            # Backprop + Optimize
            D.zero_grad()
            G.zero_grad()
            g_loss.backward()
            g_optimizer.step()
            
            if (i+1) % 10 == 0 or i == 0:
                print('Epoch [%d/%d], Step[%d/%d], d_loss: %.4f, '
                      'g_loss: %.4f, D(x): %.2f, D(G(z)): %.2f' 
                      %(epoch, epochs, i+1, len(train_loader), d_loss.data[0], g_loss.data[0],
                        real_score.data.mean(), fake_score.data.mean()))
        
        # Save real images
        # if (epoch+1) == 1:
        #     images = images.view(images.size(0), 1, 28, 28)
        #     save_image(denorm(images.data), './data/real_images.png')
        
        # Save sampled images
        # fake_images = fake_images.view(fake_images.size(0), 1, 28, 28)
        # save_image(denorm(fake_images.data), './data/fake_images-%d.png' %(epoch+1))
    except Exception:
        logfile = open('log.txt', 'w')
        logfile.write(traceback.format_exc())
        logfile.close()
    finally:
        # Save the trained parameters 
        torch.save(G.state_dict(), gParams)
        torch.save(D.state_dict(), dParams)



for epoch in range(1, epochs + 1):
    gan_train(epoch)
