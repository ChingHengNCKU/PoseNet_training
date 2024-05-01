# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 18:55:41 2023

@author: Ching Heng
"""

import torch
import nni
import os
import copy
import numpy as np
import time
import datetime
import matplotlib.pyplot as plt
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from nni.compression import TorchEvaluator
from typing import Any, Callable

from Tu_code_1115.modules.model import PoseNet
# from Tu_code_1105.modules.criterion import PoseLoss
from Tu_code_1115.modules.datasets import MAGICLab
from Tu_code_1115.modules import utils


''' Import PoseNet Model '''
# model = PoseNet()
device = torch.device('cuda')

image_height = 256
image_width = 256
x = torch.randn(1, 3, image_height, image_width, requires_grad=True)

# load entire pruned torch model
# the origin pruned model will not be changed
model_ = torch.load('./Models/magiclab_1115_2_pruned_basic_sp60.pt', map_location=torch.device('cuda'))
model_.to(device)
# model.eval()

model_pruned = copy.deepcopy(model_)
model_pruned.to(device)

''' settings '''
epochs = 50
root = './Outputs'
output_name = 'magiclab_1115_2_pruned_basic_sp60' 
## loss function  
# s_x, s_q use the latest from thr result s_x.txt, s_q.txt
criterion = utils.PoseLoss(s_x=-4.771486818790435791e-01, s_q=-2.678077459335327148e+00)
## optimizer
params= list(model_pruned.parameters()) + list(criterion.parameters())
optimizer = optim.Adam(params, lr=0.001)
## dataset
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5543, 0.5372, 0.5282], std=[0.2555, 0.2539, 0.2427])
])
trainset = MAGICLab('../i15_1111', '4markers_camera1111.xml', train=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=16, shuffle=True,
                          num_workers=10, pin_memory=True, drop_last=True)
testset = MAGICLab('../i15_1111', '4markers_camera1111.xml', train=False, transform=transform)
testloader = DataLoader(testset, batch_size=1, shuffle=False,
                          num_workers=10, pin_memory=True, drop_last=True)
## scheduler
lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(trainloader)*epochs)

def trainer(model, optimizer, lr_scheduler, criterion):
    t_start = time.time()
    print('Start fine-tuning...')
    
    model.train()
    errors = []
    err_history = {'p':[],'o':[]}
    # training loop
    for i in range(epochs):
        print(f'\nEpoch {i+1}:')
        for j, data in enumerate(trainloader):
            inputs = data['image'].to(device)
            poses = data['pose'].to(device)
            outputs = model(inputs)
            
            optimizer.zero_grad()
            
            loss = criterion(outputs, poses)
            # avoid raising .grad is None warning by calling .retain_grad() before .backward()
            loss.retain_grad()
            loss.backward()
            optimizer.step()
            
            # Calculate and print errors batch-wise 
            with torch.no_grad():
                position_errors, orientation_errors = utils.cal_pose_error(outputs, poses)
                pme, ome = position_errors.median(), orientation_errors.median()
                errors = [pme, ome]
                err_history['p'].append(pme.item())
                err_history['o'].append(ome.item())
            utils.print_loss_and_error(j+1, len(trainloader), loss, errors)
            
    # Record training time
    t_end = time.time()
    t = t_end - t_start
    with open(os.path.join(root, f'{output_name}.fine-tuning_time.txt'), 'w') as f:
        f.write(str(datetime.timedelta(seconds=t)))

    # plot
    
    plt.plot([k for k in range(len(err_history['p']))], err_history['p'], label='Postion Error')
    plt.ylabel('Error(m)')
    plt.xlabel('iteration')
    plt.legend()
    plt.savefig(f"Outputs/plots/{output_name}_position_ft.png")
    plt.close()
    plt.plot([k for k in range(len(err_history['o']))], err_history['o'], label='Orientation Error')
    plt.ylabel('Error(m)')
    plt.xlabel('iteration')
    plt.legend()
    plt.savefig(f"Outputs/plots/{output_name}_oreient_ft.png")
    plt.close()
    
    # Save model
    torch.save(model, os.path.join('Models', f'{output_name}_ft.pt'))
            
def evaluator(model):
    errors = []
    total_inference_time = 0
    model.eval()
    print('Start testing...')
    with torch.no_grad():
        for data in testloader:
            inputs = data['image'].to(device)
            poses = data['pose'].to(device)

            # Test
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            outputs = model(inputs)
            end_event.record()
            torch.cuda.synchronize()
            inference_time = start_event.elapsed_time(end_event)
            total_inference_time += inference_time

            # Calculate errors
            position_error, orientation_error = utils.cal_pose_error(outputs, poses)
            errors.append([position_error.item(), orientation_error.item()])
    errors = np.array(errors)
    median_error = np.median(errors, axis=0)
    inference_time = total_inference_time / len(testloader)
    
    # Output testing results
    np.savetxt(os.path.join(root, f'{output_name}_ft-{epochs}ep.test_pme.txt'), median_error[:1])
    np.savetxt(os.path.join(root, f'{output_name}_ft-{epochs}ep.test_ome.txt'), median_error[1:])
    with open(os.path.join(root, f'{output_name}_ft-{epochs}ep.infer_time.txt'), 'w') as f:
        f.write(f'{inference_time:.3f} ms')
    print(f'Testing median error: {median_error[0]:.3f} m, {median_error[1]:.3f} degree')
    print(f'Inference time: {inference_time:.3f} ms')
    
    
# trainer(model_pruned, optimizer, lr_scheduler, criterion)
# evaluator(model_pruned)
# input_names = ["input"]
# output_names = ["output"]
# ONNX_FILE_PATH = f"./Models/ONNX/{output_name}_ft.onnx"

# # convert and export
# torch.onnx.export(model_pruned.to('cpu'), x, ONNX_FILE_PATH, verbose=True, input_names=input_names, output_names=output_names
#                   , opset_version=17, export_params=True, do_constant_folding=True)