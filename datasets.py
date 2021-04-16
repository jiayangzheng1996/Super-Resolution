# This implementation of dataset is adopted
# from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/srgan/
# we have made some modification to this dataset to fit our need

import glob
import random
import os
import numpy as np
import cv2

import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from PIL import Image


class SRDataset(Dataset):
    def __init__(self, hr_shape, hr_dir, lr_dir=None, scaling=4):
        hr_height, hr_width = hr_shape
        self.lr_transform = None
        self.hr_transform = None


        self.lr_transform = transforms.Compose([
            transforms.Resize((hr_height // scaling, hr_width // scaling), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.hr_transform = transforms.Compose([
            transforms.Resize((hr_height, hr_width), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.lr_files = None
        if lr_dir is not None:
            self.lr_files = sorted(glob.glob(lr_dir + "/*.*"))
        self.hr_files = sorted(glob.glob(hr_dir + "/*.*"))

    def __getitem__(self, index):
        if self.lr_files is None:
            self.lr_files = self.hr_files
        lr_img = Image.open(self.lr_files[index % len(self.lr_files)])
        hr_img = Image.open(self.hr_files[index % len(self.hr_files)])
        img_lr = self.lr_transform(lr_img)
        img_hr = self.hr_transform(hr_img)

        # lr = cv2.normalize(img_hr.permute(1, 2, 0).numpy(), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # cv2.imshow("test", lr)
        # cv2.waitKey()
        # cv2.destroyAllWindows()

        return {"lr": img_lr, "hr": img_hr}

    def __len__(self):
        if self.lr_files is not None:
            assert len(self.lr_files) == len(self.hr_files)
        return len(self.hr_files)
