import os,sys
import torch
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import random
from PIL import Image, ImageOps
import cv2
import numpy as np
import torch.nn.functional as F
import pdb

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def image_loader(path):
    image = cv2.imread(path)
    image = np.float32(image) / 255.0
    image = cv2.resize(image, (256, 256))
    return image

def rgb_preprocess(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return transforms.ToTensor()(image)

def lab_preprocess(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    image = transforms.ToTensor()(image)
    # Normalize to range [-1, 1]
    image = transforms.Normalize([50,0,0], [50,127,127])(image)
    return image

class myImageFloder(data.Dataset):
    def __init__(self, filepath, filenames, training):

        self.refs = filenames
        self.filepath = filepath

    def __getitem__(self, index):
        refs = self.refs[index]

        images = [image_loader(os.path.join(self.filepath, ref)) for ref in refs]

        images_lab = [lab_preprocess(ref) for ref in images]
        images_rgb = [rgb_preprocess(ref) for ref in images]

        return images_lab, images_rgb, 1

    def __len__(self):
        return len(self.refs)
