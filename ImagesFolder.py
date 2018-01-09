from torchvision import datasets, transforms
from skimage.color import rgb2lab, rgb2gray
import torch
import numpy as np
import os
scale_transform = transforms.Compose([
    transforms.RandomCrop(224),
    #transforms.ToTensor()
])

#https://github.com/shufanwu/colorNet-pytorch/blob/master/myimgfolder.py
class TrainFolder(datasets.ImageFolder):

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img_original = self.transform(img)
            img_original = np.asarray(img_original)

            img_lab = rgb2lab(img_original)
            img_lab = (img_lab + 128) / 255
            img_ab = img_lab[:, :, 1:3]
            img_ab = torch.from_numpy(img_ab.transpose((2, 0, 1)))
            img_original = rgb2gray(img_original)
            img_original = torch.from_numpy(img_original)
        else:
            img_original=img
        if self.target_transform is not None:
            target = self.target_transform(target)
        return (img_original, img_ab), target


class ValFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)

        img_scale = img.copy()
        img_original = img
        img_scale = scale_transform(img_scale)

        img_scale = np.asarray(img_scale)
        img_original = np.asarray(img_original)

        img_scale = rgb2gray(img_scale)
        img_scale = torch.from_numpy(img_scale)
        img_original = rgb2gray(img_original)
        img_original = torch.from_numpy(img_original)
        return (img_original, img_scale), target
