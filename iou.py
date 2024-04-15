import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2
from PIL import Image
import os 

SMOOTH = 1e-6

def iou_numpy(outputs: np.array, labels: np.array):
    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    thresholded = np.ceil(np.clip(20 * (iou - 0.5), 0, 10)) / 10
    return np.mean(thresholded)  

def combine(first_folder,second_folder):

    list=[]

    for root,dirs,files in os.walk(first_folder):
        for file in files:
            first_image=os.path.join(root,file)
            second_image=os.path.join(second_folder,file)
            first_image=cv2.imread(first_image)
            second_image=cv2.imread(second_image)
            result = iou_numpy(first_image,second_image)
            list.append(result)

    return sum(list)/len(list)
                  
first_folder='results/sagan/ground_truth'
second_folder='results/sagan/sagan_cbam_attention'

result = combine(first_folder,second_folder)

print(" Mean IOU Score =", result)