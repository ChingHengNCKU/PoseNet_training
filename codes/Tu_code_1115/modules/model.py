from torch import nn
from torchvision import models

class PoseNet(nn.Module):
    """PoseNet (EfficientNetV2-S)."""

    def __init__(self, weights=None):
        """
        Args:
            weights (Weights, str): Pretrained weights.
        """
        super().__init__()
        net = models.efficientnet_v2_s(weights=weights)
        net.classifier[1] = nn.Linear(net.classifier[1].in_features, 6)
        self.net = net

    def forward(self, images):
        """
        Args:
            images (torch.Tensor): Input images.

        Returns:
            poses (torch.Tensor): Output poses.

        Shape:
            images: (batch_size, 3, height, width).
            poses: (batch_size, 6).
        """
        poses = self.net(images)
        return poses
