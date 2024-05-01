# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 22:27:04 2023

@author: Ching Heng
"""

from mpl_toolkits import mplot3d
import os
import numpy as np
import matplotlib.pyplot as plt

gt_dir = "./images_4asec"
dir_list = os.listdir(gt_dir)
gt_pos_dict = {}

for file in dir_list:
    if file.split('.')[1] == 'txt':
        file_name =  file.split('.')[0]
        if file_name.find('7Params') == -1:
            trans_mat = np.loadtxt(os.path.join(gt_dir, file))
            gt_pos_dict[file_name] = trans_mat[0:3, 3]
            
            
def getSeqDatas(seq, gt_pos_dict, dir_list):
    seq_files = []
    for file in dir_list:
        if file.split('.')[1] == 'txt':
            file_name =  file.split('.')[0]
            if file_name[3:5] == seq and file_name.find('7Params') == -1:
                seq_files.append(file_name)
                
    xdata = []
    ydata = []
    zdata = []
    for file in seq_files:
        xdata.append(gt_pos_dict[file][0])
        ydata.append(gt_pos_dict[file][1])
        zdata.append(gt_pos_dict[file][2])
    return xdata, ydata, zdata


fig = plt.figure() 
ax = plt.axes(projection='3d')

# specify which seq would be drawn
xdata, ydata, zdata = getSeqDatas('00', gt_pos_dict, dir_list)
ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens', linewidths=0.01);