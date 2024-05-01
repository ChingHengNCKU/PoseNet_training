from torch import nn
from torchvision import models
from kornia.geometry import conversions
from PIL import Image
import torch.nn.functional as F
import torch.nn.init
import torch
import torchvision
from att import AttentionBlock

class PoseNet(nn.Module):
    """PoseNet (Efficientnet_B4_Weights)."""

    def __init__(self, droprate=0.5, device='cuda'):
        """
        Args:
            weights (Efficientnet_B4_Weights):IMAGENET1K_V1, return Efficientnet_B4_Weights.IMAGENET1K_V1 which is a model 
            pre-trained on imagenet1k_v1.
        """
        super().__init__()
        self.droprate = droprate
        self.eff_layers = models.efficientnet_b4(weights='IMAGENET1K_V1')
        fe_out_planes = self.eff_layers.classifier[1].in_features #get input feature dim of nn.Linear
        self.eff_layers = nn.Sequential(*list(self.eff_layers.children())[:-1]) #remove the Linear layer
        ## Atloc 
        self.att = AttentionBlock(fe_out_planes)
        self.fc_poses = nn.Linear(fe_out_planes, 7)

        nn.init.kaiming_normal_(self.fc_poses.weight.data)
        if self.fc_poses.bias is not None:
            nn.init.constant_(self.fc_poses.bias.data, 0)
        self.dropout = nn.Dropout(p=self.droprate)
        
    ##TODO: resize the input images to 1024*1024 pixels##
    
    def forward(self, images):
        """
        Args:
            images (torch.Tensor): Input images. 

        Returns:
            poses (torch.Tensor): Output poses.

        Shape:
            images: (batch_size, 3, height, width).
            poses: (batch_size, 7).
        """
        x = self.eff_layers(images)
        x = torch.flatten(x, 1)
        #Atloc
        x = self.att(x.view(x.size(0), -1))
        # if self.droprate > 0:
        x = self.dropout(x)
        poses = self.fc_poses(x)
        
        poses[:, 3:] = conversions.normalize_quaternion(poses[:, 3:].clone()) # normalize q
        poses[:, 3:][poses[:, 3] < 0] = -poses[:, 3:][poses[:, 3] < 0] # constrain q0 > 0
        return poses