# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 19:31:50 2023

@author: Ching Heng
"""

import numpy as np
from kornia.geometry import conversions
import os

## 以第一張相片為起始點(0, 0, 0) (X:right, Y:down, Z:forward)
startPoint = np.loadtxt('./images_4asec/seq12-0615.txt')
## get the inverse matrix
invMat = np.linalg.inv(startPoint)

gt_dir = "./images_4asec"
dir_list = os.listdir(gt_dir)
gt_pos_dict = {}

for file in dir_list:
    if file.split('.')[1] == 'txt':
        file_name =  file.split('.')[0]
        trans_mat = np.loadtxt(os.path.join(gt_dir, file))
        gt_pos_dict[file_name] = np.matmul(invMat, trans_mat)[0:3, 3]
            