# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 20:38:20 2023

@author: Ching Heng
"""

import torch
import torchvision
import os 

from Tu_code_1115.modules.model import PoseNet

''' Import PoseNet Model '''
posenet = PoseNet()
device = torch.device('cuda')
model_name = 'magiclab_1115_2'

image_height = 256
image_width = 256
x = torch.randn(1, 3, image_height, image_width, requires_grad=True)
for param in posenet.parameters():
    param.requires_grad = False
checkpoint = torch.load(f'./Models/{model_name}.pt', map_location=torch.device('cuda'))
posenet.load_state_dict(checkpoint['model'])
posenet.eval()

''' Convert .pt model to .onnx format '''
input_names = ["input"]
output_names = ["output"]
ONNX_FILE_PATH = f"./Models/ONNX/{model_name}.onnx"

# convert and export
torch.onnx.export(posenet, x, ONNX_FILE_PATH, verbose=True, input_names=input_names, output_names=output_names
                  , opset_version=17, export_params=True, do_constant_folding=True)

''' Quantization part use ONNX RUNTIME API '''
import onnxruntime
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType
from PIL import Image
import numpy as np

## Define some utils functions and class ##
def preprocess_image(image_path, height, width, channels=3):
    image = Image.open(image_path)
    image = image.resize((width, height))
    image_data = np.asarray(image).astype(np.float32)
    image_data = image_data.transpose([2, 0, 1]) # transpose to CHW
    # mean, std infos are from transform.txt  
    mean = np.array([0.5543, 0.5372, 0.5282])
    std = np.array([0.2555, 0.2539, 0.2427])
    for channel in range(image_data.shape[0]):
        image_data[channel, :, :] = (image_data[channel, :, :] / 255 - mean[channel]) / std[channel]
    image_data = np.expand_dims(image_data, 0)
    return image_data

def preprocess_func(images_folder, height, width, size_limit=0):
    image_names = os.listdir(images_folder)
    if size_limit > 0 and len(image_names) >= size_limit:
        batch_filenames = [image_names[i] for i in range(size_limit)]
    else:
        batch_filenames = image_names
    unconcatenated_batch_data = []

    for image_name in batch_filenames:
        image_filepath = images_folder + '/' + image_name
        image_data = preprocess_image(image_filepath, height, width)
        unconcatenated_batch_data.append(image_data)
    batch_data = np.concatenate(np.expand_dims(unconcatenated_batch_data, axis=0), axis=0)
    return batch_data


class PoseNetDataReader(CalibrationDataReader):
    def __init__(self, calibration_image_folder):
        self.image_folder = calibration_image_folder
        self.preprocess_flag = True
        self.enum_data_dicts = []
        self.datasize = 0

    def get_next(self):
        if self.preprocess_flag:
            self.preprocess_flag = False
            nhwc_data_list = preprocess_func(self.image_folder, image_height, image_width, size_limit=0)
            self.datasize = len(nhwc_data_list)
            self.enum_data_dicts = iter([{'input': nhwc_data} for nhwc_data in nhwc_data_list])
        return next(self.enum_data_dicts, None)
    
## Quantize the model ##
# official recommands needing pre-process model for doing:
    # 1. Symbolic Shape Inference. It works best with transformer models.
    # 2. ONNX Runtime Model Optimization.
    # 3. ONNX Shape Inference.
# using command line to run
# python -m onnxruntime.quantization.preprocess --input './Models/ONNX/modelName.onnx' --output './Models/ONNX/modelName-infer.onnx'

# change it to your real calibration data set
calibration_data_folder = f"calibration_folders/magiclab_1115_calibration_folder"
dr = PoseNetDataReader(calibration_data_folder)

# Static Quantization model from float32 to int8 
quantize_static(f'./Models/ONNX/{model_name}-infer.onnx',
                f'./Models/ONNX/{model_name}_int8.onnx',
                dr, 
                weight_type=QuantType.QInt8)

''' Pruning part using NNI API '''
## Basic Pruner ##
### NNI Version=2.10.1 ###
from nni.compression.speedup import ModelSpeedup
from nni.compression.utils.counter import count_flops_params
from nni.compression.pruning.basic_pruner import L1NormPruner, L2NormPruner
import copy

## clone an original PoseNet for pruning
posenet_pruned = copy.deepcopy(posenet)
config_list = [{
    'op_types': ['Conv2d'],
    'sparse_ratio': 0.75,
}]
pruner = L1NormPruner(posenet_pruned, config_list)
# show the wrapped model structure, `PrunerModuleWrapper` have wrapped the layers that configured in the config_list.
# print(posenet)
# compress the model and generate the masks
_, masks = pruner.compress()
# show the masks sparsity
for name, mask in masks.items():
    print(name, ' sparsity : ', '{:.2}'.format(mask['weight'].sum() / mask['weight'].numel()))
# need to unwrap the model, if the model is wrapped before speedup
pruner.unwrap_model()
# speedup the model
ModelSpeedup(posenet_pruned, dummy_input=x, masks_or_file=masks).speedup_model()
# print(posenet_pruned)
# showing the results containing FLOPs and Params
# flops, params, results = count_flops_params(posenet_pruned, x)
## save 'ENTIRE' pruned torch model (.pt) 
torch.save(posenet_pruned, f'./Models/{model_name}_pruned_basic_sp75.pt')