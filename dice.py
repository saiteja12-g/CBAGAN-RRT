import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2
from PIL import Image
import os 

SMOOTH = 1e-6

def dice(im1, im2):
    im1 = np.asarray(im1).astype(bool)
    im2 = np.asarray(im2).astype(bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    intersection = np.logical_and(im1, im2)
    return 2. * intersection.sum() / (im1.sum() + im2.sum())

def combine(first_folder,second_folder):

    list=[]

    for root,dirs,files in os.walk(first_folder):
        for file in files:
            first_image=os.path.join(root,file)
            second_image=os.path.join(second_folder,file)
            first_image=cv2.imread(first_image)
            second_image=cv2.imread(second_image)
            result = dice(first_image,second_image)
            list.append(result)

    return sum(list)/len(list)
                  
first_folder='results/sagan/ground_truth'
second_folder='results/sagan/sagan_cbam_attention'

result = combine(first_folder,second_folder)

print("Mean Dice Score =", result)