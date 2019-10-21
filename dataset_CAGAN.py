import os
import numpy as np
import cv2
from skimage import io
from skimage import color
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class LabDataset(Dataset):
    def __init__(self, baseroot, imglist):
        self.baseroot = baseroot
        self.imglist = imglist
        self.transform = transforms.ToTensor()

    def __getitem__(self, index):
        imgpath = os.path.join(self.baseroot, self.imglist[index])      # path of one image
        rgb = cv2.imread(imgpath)                                       # read one image
        rgb = rgb[:, :, ::-1]
        # convert RGB to Lab, finally get Tensor
        lab = color.rgb2lab(rgb).astype(np.float32)                     # skimage Lab: L [0, 100], a [-128, 127], b [-128, 127], order [H, W, C]
        lab = self.transform(lab)                                       # Tensor Lab: L [0, 100], a [-128, 127], b [-128, 127], order [C, H, W]
        # normaization
        l = lab[[0], ...] / 50 - 1.0                                    # L, normalized to [-1, 1]
        ab = lab[[1, 2], ...] / 110.0                                   # a and b, normalized to [-1, 1], approximately
        return l, ab
    
    def __len__(self):
        return len(self.imglist)

class LabAndSalDataset(Dataset):
    def __init__(self, colorbaseroot, salbaseroot, imglist):
        self.colorbaseroot = colorbaseroot
        self.salbaseroot = salbaseroot
        self.imglist = imglist
        self.transform = transforms.ToTensor()

    def __getitem__(self, index):
        imgpath = os.path.join(self.colorbaseroot, self.imglist[index]) # path of one image
        rgb = cv2.imread(imgpath)                                       # read one image
        rgb = rgb[:, :, ::-1]
        # convert RGB to Lab, finally get Tensor
        lab = color.rgb2lab(rgb).astype(np.float32)                     # skimage Lab: L [0, 100], a [-128, 127], b [-128, 127], order [H, W, C]
        lab = self.transform(lab)                                       # Tensor Lab: L [0, 100], a [-128, 127], b [-128, 127], order [C, H, W]
        # normaization
        l = lab[[0], ...] / 50 - 1.0                                    # L, normalized to [-1, 1]
        ab = lab[[1, 2], ...] / 110.0                                   # a and b, normalized to [-1, 1], approximately
        
        salpath = os.path.join(self.salbaseroot, self.imglist[index])   # path of one saliency map
        saliencymap = io.imread(salpath)                                # read one image
        saliencymap = np.expand_dims(saliencymap, axis = 0)
        saliencymap = saliencymap / 255                                 # saliencymap [0, 1]
        saliencymap = torch.from_numpy(saliencymap.astype(np.float32))
        return l, (saliencymap, ab)
    
    def __len__(self):
        return len(self.imglist)

class Lab_Sal_LabelDataset(Dataset):
    def __init__(self, colorbaseroot, salbaseroot, imglist, stringlist, scalarlist):
        self.colorbaseroot = colorbaseroot
        self.salbaseroot = salbaseroot
        self.imglist = imglist
        self.stringlist = stringlist
        self.scalarlist = scalarlist
        self.transform1 = transforms.ToTensor()
        self.transform2 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __getitem__(self, index):
        imgname = self.imglist[index]
        imgpath = os.path.join(self.colorbaseroot, self.imglist[index]) # path of one image
        ## The Lab color space part
        rgb = cv2.imread(imgpath)                                       # read one image
        rgb = rgb[:, :, ::-1]
        # convert RGB to Lab, finally get Tensor
        lab = color.rgb2lab(rgb).astype(np.float32)                     # skimage Lab: L [0, 100], a [-128, 127], b [-128, 127], order [H, W, C]
        lab = self.transform1(lab)                                      # Tensor Lab: L [0, 100], a [-128, 127], b [-128, 127], order [C, H, W]
        # normaization
        l = lab[[0], ...] / 50 - 1.0                                    # L, normalized to [-1, 1]
        ab = lab[[1, 2], ...] / 110.0                                   # a and b, normalized to [-1, 1], approximately
        ## The saliency map part
        salpath = os.path.join(self.salbaseroot, self.imglist[index])   # path of one saliency map
        saliencymap = io.imread(salpath)                                # read one image
        saliencymap = np.expand_dims(saliencymap, axis = 0)
        saliencymap = saliencymap / 255                                 # saliencymap [0, 1]
        saliencymap = torch.from_numpy(saliencymap.astype(np.float32))
        ## The category label part
        stringname = imgname[:9]                                        # category by str: like n01440764
        for index, value in enumerate(self.stringlist):
            if stringname == value:
                target = self.scalarlist[index]                         # target: 1~1000
                target = int(target) - 1                                # target: 0~999
                target = np.array(target, dtype = np.int64)
                target = torch.from_numpy(target)
        return l, (saliencymap, ab, target)
    
    def __len__(self):
        return len(self.imglist)
