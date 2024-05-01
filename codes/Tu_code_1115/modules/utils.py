import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Subset
from kornia.geometry import conversions
from torch import nn

def get_seven_scenes_mean_std(scene, validation=False):
    """Get RGB mean and standard deviation of the scene in 7-Scenes dataset.

    Args:
        scene (str): Scene in 7-Scenes dataset.
        validation

    Returns:
        [mean (list): RGB mean of training data,
         std (list): RGB standard deviation of training data.]
    """
    mean = []
    std = []
    if scene == 'chess':
        if validation:
            mean = [0.5019, 0.4424, 0.4466]
            std = [0.2037, 0.2241, 0.2127]
        else:
            mean = [0.5007, 0.4412, 0.4456]
            std = [0.2034, 0.2243, 0.2128]
    elif scene == 'fire':
        if validation:
            mean = [0.5212, 0.4584, 0.4168]
            std = [0.2310, 0.2381, 0.2251]
        else:
            mean = [0.5219, 0.4618, 0.4210]
            std = [0.2314, 0.2379, 0.2260]
    elif scene == 'heads':
        if validation:
            mean = [0.4572, 0.4504, 0.4577]
            std = [0.2753, 0.2728, 0.2620]
        else:
            mean = [0.4565, 0.4500, 0.4580]
            std = [0.2743, 0.2720, 0.2618]
    elif scene == 'office':
        if validation:
            mean = [0.4723, 0.4435, 0.4369]
            std = [0.2588, 0.2619, 0.2543]
        else:
            mean = [0.4732, 0.4439, 0.4369]
            std = [0.2586, 0.2613, 0.2535]
    elif scene == 'pumpkin':
        if validation:
            mean = [0.5506, 0.4504, 0.4589]
            std = [0.1968, 0.2168, 0.1809]
        else:
            mean = [0.5501, 0.4492, 0.4577]
            std = [0.1961, 0.2158, 0.1800]
    elif scene == 'redkitchen':
        if validation:
            mean = [0.5225, 0.4377, 0.4304]
            std = [0.2149, 0.2492, 0.2336]
        else:
            mean = [0.5228, 0.4394, 0.4311]
            std = [0.2149, 0.2484, 0.2329]
    elif scene == 'stairs':
        if validation:
            mean = [0.4451, 0.4300, 0.4276]
            std = [0.1756, 0.1578, 0.1048]
        else:
            mean = [0.4470, 0.4310, 0.4289]
            std = [0.1759, 0.1575, 0.1046]
    return [mean, std]

def split_trainset(dataset, scene):
    """
    """
    indices_val = []
    if scene == 'chess':
        indices_val = range(3100, 3500)
    elif scene == 'fire':
        indices_val = range(400, 600)
    elif scene == 'heads':
        indices_val = range(200, 300)
    elif scene == 'office':
        indices_val = range(5000, 5600)
    elif scene == 'pumpkin':
        indices_val = range(3000, 3400)
    elif scene == 'redkitchen':
        indices_val = range(6300, 7000)
    elif scene == 'stairs':
        indices_val = range(1800, 2000)
    elif scene == 'magiclab':
#        indices_val = list(range(1161, 1324)) + list(range(1818, 1970))
        indices_val = range(500, 560)
    indices_train = [i for i in range(len(dataset)) if i not in indices_val]
    trainset = Subset(dataset, indices_train)
    validationset = Subset(dataset, indices_val)
    return trainset, validationset

def freeze_posenet(posenet, frozen_layers=0):
    """(EfficientNet)
    """
    requires_grad = False
    for name, param in posenet.net.features.named_parameters():
        if name[0] == str(frozen_layers):
            requires_grad = True
        param.requires_grad = requires_grad
    return posenet

def quaternion_product(q, r):
    """Calculate quaternion product from MATLAB.

    Args:
        q, r (torch.Tensor): Input quaternions.

    Returns:
        n (torch.Tensor): Output quaternion.

    Shape:
        q: (N, 4).
        r: (N, 4).
        n: (N, 4).
    """
    n0 = r[:, 0] * q[:, 0] - r[:, 1] * q[:, 1] - r[:, 2] * q[:, 2] - r[:, 3] * q[:, 3] # (N)
    n1 = r[:, 0] * q[:, 1] + r[:, 1] * q[:, 0] - r[:, 2] * q[:, 3] + r[:, 3] * q[:, 2] # (N)
    n2 = r[:, 0] * q[:, 2] + r[:, 1] * q[:, 3] + r[:, 2] * q[:, 0] - r[:, 3] * q[:, 1] # (N)
    n3 = r[:, 0] * q[:, 3] - r[:, 1] * q[:, 2] + r[:, 2] * q[:, 1] + r[:, 3] * q[:, 0] # (N)
    n = torch.stack((n0, n1, n2, n3), dim=1) # (N, 4)
    return n

def cal_relative_pose(pose1, pose2):
    """
    """
    x1, x2 = pose1[:, :3].unsqueeze(2), pose2[:, :3].unsqueeze(2) # (N, 3, 1)
    log_q1, log_q2 = pose1[:, 3:], pose2[:, 3:]                   # (N, 3)
    q1 = conversions.quaternion_log_to_exp(log_q1)                # (N, 4)
    q2 = conversions.quaternion_log_to_exp(log_q2)                # (N, 4)
    q2_inv = torch.cat((q2[:, :1], -q2[:, 1:]), dim=1)            # (N, 4)
    q12 = quaternion_product(q2_inv, q1)                          # (N, 4)
    log_q12 = conversions.quaternion_exp_to_log(q12)              # (N, 3)
    R2 = conversions.quaternion_to_rotation_matrix(q2)            # (N, 3, 3)
    x12 = R2.transpose(1, 2) @ (x1-x2)                            # (N, 3, 1)
    x12 = x12.squeeze(2)                                          # (N, 3)
    pose12 = torch.cat((x12, log_q12), dim=1)                     # (N, 6)
    return pose12

def cal_pose_error(pred, targ):
    """
    """
    rel_pose = cal_relative_pose(pred, targ)                        # (N, 6)
    x = rel_pose[:, :3]                                             # (N, 3)
    log_q = rel_pose[:, 3:]                                         # (N, 3)
    q = conversions.quaternion_log_to_exp(log_q)                    # (N, 4)
    position_errors = torch.linalg.norm(x, dim=1)                   # (N)
    orientation_errors = conversions.rad2deg(2*torch.acos(q[:, 0])) # (N)
    return position_errors, orientation_errors

def print_loss_and_error(iteration, total_iterations, loss, errors):
    """Clean the line and print the loss and error.

    Args:
        iteration (int): This iteration.
        total_iterations (int): Total iterations.
        loss (float): Loss value.
        errors (list): [position error, orientation error].
    """
    string = (f'{iteration}/{total_iterations} - loss: {loss:.3f}' +
              f' - error: {errors[0]:.3f} m, {errors[1]:.3f} degree')
    space = '                                                            '
    print(f'\r{space}', end='\r')
    if iteration != total_iterations:
        print(string, end='', flush=True)
    else:
        print(string)

def output_plot(output_name, history, xlabel='Iteration', ylabel='', ylim=None, linewidth=1.5):
    """Output plot and values of the history.

    Args:
        output_name (str): Output name.
        history (numpy.ndarray): History.
        xlabel (str): X-label of the plot.
        ylabel (str): Y-label of the plot.
        ylim (list): Y-limit of the plot. [bottom, top]
    """
    np.savetxt(f'{output_name}.txt', history)
    plt.plot(history, linewidth=linewidth)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if ylim is not None:
        plt.ylim(ylim)
    plt.savefig(f'{output_name}.png')
    plt.close()

def output_plot_2(output_name, train_history, val_iter_history, val_history,
                  xlabel='Iteration', ylabel='', ylim=None):
    """Output plot and values of the history.

    Args:
        output_name (str): Output name.
        train_history (numpy.ndarray): Training history.
        val_iter_history ():
        val_history (numpy.ndarray): Validation history.
        xlabel (str): X-label of the plot.
        ylabel (str): Y-label of the plot.
        ylim (list): Y-limit of the plot. [bottom, top]
    """
    np.savetxt(f'{output_name}.train.txt', train_history)
    np.savetxt(f'{output_name}.val.txt', val_history)
    plt.plot(train_history, linewidth=0.5)
    plt.plot(val_iter_history, val_history)
    plt.legend(['Training', 'Validation'])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if ylim is not None:
        plt.ylim(ylim)
    plt.savefig(f'{output_name}.png')
    plt.close()

class PoseLoss(nn.Module):
    """Camera pose loss"""

    def __init__(self, s_x=0.0, s_q=-3.0, requires_grad=True):
        """
        Args:
            s_x, s_q (float): Initial value of s_x, s_q.
            requires_grad
        """
        super().__init__()
        self.s_x = nn.Parameter(torch.tensor(s_x), requires_grad=requires_grad)
        self.s_q = nn.Parameter(torch.tensor(s_q), requires_grad=requires_grad)

    def forward(self, pred, targ):
        """
        Args:
            pred (torch.Tensor): Predicted camera poses.
            targ (torch.Tensor): Target camera poses.

        Returns:
            loss (torch.Tensor): Loss value.

        Shape:
            pred: (batch_size, 6).
            targ: (batch_size, 6).
            loss: (1).
        """
        batch_size = pred.shape[0]
        pred1, pred2 = pred[:int(batch_size/2), :], pred[int(batch_size/2):, :]
        targ1, targ2 = targ[:int(batch_size/2), :], targ[int(batch_size/2):, :]
        pred_rel = cal_relative_pose(pred1, pred2)
        targ_rel = cal_relative_pose(targ1, targ2)

        l1_loss = nn.L1Loss()
        loss = (l1_loss(pred[:, :3], targ[:, :3]) * torch.exp(-self.s_x) + self.s_x +
                l1_loss(pred[:, 3:], targ[:, 3:]) * torch.exp(-self.s_q) + self.s_q +
                l1_loss(pred_rel[:, :3], targ_rel[:, :3]) * torch.exp(-self.s_x) + self.s_x +
                l1_loss(pred_rel[:, 3:], targ_rel[:, 3:]) * torch.exp(-self.s_q) + self.s_q)
        return loss
