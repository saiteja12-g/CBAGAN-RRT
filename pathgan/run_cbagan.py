import cv2
import numpy as np
import pandas as pd
from utils import rgb2binary
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from PIL import Image, ImageDraw
import os
import re
import glob
import sys
from  pathlib import Path
from collections import deque
from models.rrt.cbagan_rrt import RRT

dataset_path = '../dataset/sagan_results/gt/results/'
data_path = '../' 

roi_result_list = glob.glob(dataset_path + './*.png')
roi_result_list.sort(key = lambda s: int(re.search('\d+', s).group()))

data = pd.read_csv(dataset_path + './results.csv')
data = data.sort_values(by="true_roi").reset_index(drop=True)

map_file = os.path.dirname(data.true_roi[8])

Tasks = pd.read_csv(data_path + map_file + '.csv')

task_num = int(re.search('task\_(\d+)', os.path.basename(data.true_roi[8])).group(1))

start = (Tasks.istart[task_num], Tasks.jstart[task_num])
goal = (Tasks.igoal[task_num], Tasks.jgoal[task_num])

colors = {'start': np.array([0, 0, 1]), 'goal': np.array([1, 0, 0]), 'roi': np.array([0, 1, 0])}

Map = re.search('map\_(\d+)', data.true_roi[8]).group(0)

Map_img = Image.open('../dataset/maps/' + Map + '.png').convert('RGB')

true_roi = Image.open(data_path + data.true_roi[8]).convert('RGB')

dataset_path= '../dataset/sagan_results/sagan_cbam_attention/'

pred_roi = Image.open(dataset_path + data.pred_roi[8]).convert('RGB')

fig, ax = plt.subplots(ncols=4,  figsize=(12, 10), dpi=200)
for axx, map_img in zip(ax.flatten(), [Map_img, true_roi, pred_roi]):
    k = map_img.load()
    k[int(start[1]), int(start[0])] = tuple(colors['start']*255)
    k[int(goal[1]), int(goal[0])] = tuple(colors['goal']*255)
    
    axx.axes.xaxis.set_visible(False)
    axx.axes.yaxis.set_visible(False)
    axx.imshow(map_img);
plt.tight_layout()

pred_roi = Image.open(dataset_path + data.pred_roi[8]).convert('RGB')
mask_grid = rgb2binary(np.array(pred_roi))
mask_grid = np.where(np.sum(mask_grid, axis=2) > 2, 0, 1)

fig, ax = plt.subplots(dpi=150)
ax.axes.xaxis.set_visible(False)
ax.axes.yaxis.set_visible(False)
cv2.imwrite('input2.png', mask_grid)

model = RRT(mask_grid)

