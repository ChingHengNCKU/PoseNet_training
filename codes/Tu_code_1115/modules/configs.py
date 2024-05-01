import os
import torch
from torchvision import transforms
from ray import tune
from easydict import EasyDict

from modules import utils

"""Configuration.
"""

# Population Based Training

# 7-Scenes chess
pbt_chess = EasyDict()
pbt_chess.output_name = 'pbt_chess'
pbt_chess.dataset = 'SevenScenes'
pbt_chess.path = r'/home/user/Datasets/7-Scenes'
pbt_chess.scene = 'chess'
pbt_chess.resize = 256
pbt_chess.val_batch_size = 64
pbt_chess.train_num_workers = 1
pbt_chess.val_num_workers = 1
pbt_chess.pin_memory = True
pbt_chess.pretrained_weights = 'DEFAULT'
pbt_chess.resources = {'cpu': 10, 'gpu': 1}
pbt_chess.num_samples = 20
pbt_chess.checkpoint_interval = 60
pbt_chess.stop_time = 3600
pbt_chess.pbt_args = {
    'time_attr': 'time_total_s',
    'perturbation_interval': pbt_chess.checkpoint_interval,
    'hyperparam_mutations': {
        'batch_size': [4, 8, 12, 16, 24, 32, 48, 64],
        'frozen_layers': [0, 1, 2, 3, 4, 5, 6, 7],
        'lr': tune.loguniform(1e-5, 1e-1),
        'weight_decay': tune.uniform(1e-4, 1e0)
    },
    'quantile_fraction': 0.25,
    'resample_probability': 0.25,
    'perturbation_factors': (1.2, 0.8)
}

mean, std = utils.get_seven_scenes_mean_std(pbt_chess.scene, validation=True)
pbt_chess.transform = transforms.Compose([
    transforms.Resize(pbt_chess.resize),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])
pbt_chess.train_dataset_args = {
    'path': pbt_chess.path,
    'scene': pbt_chess.scene,
    'train': True,
    'transform': pbt_chess.transform
}
pbt_chess.test_dataset_args = {
    'path': pbt_chess.path,
    'scene': pbt_chess.scene,
    'train': False,
    'transform': pbt_chess.transform
}

# 7-Scenes fire
pbt_fire = EasyDict()
pbt_fire.output_name = 'pbt_fire'
pbt_fire.dataset = 'SevenScenes'
pbt_fire.path = r'/home/user/Datasets/7-Scenes'
pbt_fire.scene = 'fire'
pbt_fire.resize = 256
pbt_fire.val_batch_size = 64
pbt_fire.train_num_workers = 1
pbt_fire.val_num_workers = 1
pbt_fire.pin_memory = True
pbt_fire.pretrained_weights = 'DEFAULT'
pbt_fire.resources = {'cpu': 10, 'gpu': 1}
pbt_fire.num_samples = 20
pbt_fire.checkpoint_interval = 60
pbt_fire.stop_time = 3600
pbt_fire.pbt_args = {
    'time_attr': 'time_total_s',
    'perturbation_interval': pbt_fire.checkpoint_interval,
    'hyperparam_mutations': {
        'batch_size': [4, 8, 12, 16, 24, 32, 48, 64],
        'frozen_layers': [0, 1, 2, 3, 4, 5, 6, 7],
        'lr': tune.loguniform(1e-5, 1e-1),
        'weight_decay': tune.uniform(1e-4, 1e0)
    },
    'quantile_fraction': 0.25,
    'resample_probability': 0.25,
    'perturbation_factors': (1.2, 0.8)
}

mean, std = utils.get_seven_scenes_mean_std(pbt_fire.scene, validation=True)
pbt_fire.transform = transforms.Compose([
    transforms.Resize(pbt_fire.resize),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])
pbt_fire.train_dataset_args = {
    'path': pbt_fire.path,
    'scene': pbt_fire.scene,
    'train': True,
    'transform': pbt_fire.transform
}
pbt_fire.test_dataset_args = {
    'path': pbt_fire.path,
    'scene': pbt_fire.scene,
    'train': False,
    'transform': pbt_fire.transform
}

# 7-Scenes heads
pbt_heads = EasyDict()
pbt_heads.output_name = 'pbt_heads'
pbt_heads.dataset = 'SevenScenes'
pbt_heads.path = r'/home/user/Datasets/7-Scenes'
pbt_heads.scene = 'heads'
pbt_heads.resize = 256
pbt_heads.val_batch_size = 64
pbt_heads.train_num_workers = 1
pbt_heads.val_num_workers = 1
pbt_heads.pin_memory = True
pbt_heads.pretrained_weights = 'DEFAULT'
pbt_heads.resources = {'cpu': 10, 'gpu': 1}
pbt_heads.num_samples = 20
pbt_heads.checkpoint_interval = 60
pbt_heads.stop_time = 3600
pbt_heads.pbt_args = {
    'time_attr': 'time_total_s',
    'perturbation_interval': pbt_heads.checkpoint_interval,
    'hyperparam_mutations': {
        'batch_size': [4, 8, 12, 16, 24, 32, 48, 64],
        'frozen_layers': [0, 1, 2, 3, 4, 5, 6, 7],
        'lr': tune.loguniform(1e-5, 1e-1),
        'weight_decay': tune.uniform(1e-4, 1e0)
    },
    'quantile_fraction': 0.25,
    'resample_probability': 0.25,
    'perturbation_factors': (1.2, 0.8)
}

mean, std = utils.get_seven_scenes_mean_std(pbt_heads.scene, validation=True)
pbt_heads.transform = transforms.Compose([
    transforms.Resize(pbt_heads.resize),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])
pbt_heads.train_dataset_args = {
    'path': pbt_heads.path,
    'scene': pbt_heads.scene,
    'train': True,
    'transform': pbt_heads.transform
}
pbt_heads.test_dataset_args = {
    'path': pbt_heads.path,
    'scene': pbt_heads.scene,
    'train': False,
    'transform': pbt_heads.transform
}

# 7-Scenes office
pbt_office = EasyDict()
pbt_office.output_name = 'pbt_office'
pbt_office.dataset = 'SevenScenes'
pbt_office.path = r'/home/user/Datasets/7-Scenes'
pbt_office.scene = 'office'
pbt_office.resize = 256
pbt_office.val_batch_size = 64
pbt_office.train_num_workers = 1
pbt_office.val_num_workers = 1
pbt_office.pin_memory = True
pbt_office.pretrained_weights = 'DEFAULT'
pbt_office.resources = {'cpu': 10, 'gpu': 1}
pbt_office.num_samples = 20
pbt_office.checkpoint_interval = 60
pbt_office.stop_time = 3600
pbt_office.pbt_args = {
    'time_attr': 'time_total_s',
    'perturbation_interval': pbt_office.checkpoint_interval,
    'hyperparam_mutations': {
        'batch_size': [4, 8, 12, 16, 24, 32, 48, 64],
        'frozen_layers': [0, 1, 2, 3, 4, 5, 6, 7],
        'lr': tune.loguniform(1e-5, 1e-1),
        'weight_decay': tune.uniform(1e-4, 1e0)
    },
    'quantile_fraction': 0.25,
    'resample_probability': 0.25,
    'perturbation_factors': (1.2, 0.8)
}

mean, std = utils.get_seven_scenes_mean_std(pbt_office.scene, validation=True)
pbt_office.transform = transforms.Compose([
    transforms.Resize(pbt_office.resize),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])
pbt_office.train_dataset_args = {
    'path': pbt_office.path,
    'scene': pbt_office.scene,
    'train': True,
    'transform': pbt_office.transform
}
pbt_office.test_dataset_args = {
    'path': pbt_office.path,
    'scene': pbt_office.scene,
    'train': False,
    'transform': pbt_office.transform
}

# 7-Scenes pumpkin
pbt_pumpkin = EasyDict()
pbt_pumpkin.output_name = 'pbt_pumpkin'
pbt_pumpkin.dataset = 'SevenScenes'
pbt_pumpkin.path = r'/home/user/Datasets/7-Scenes'
pbt_pumpkin.scene = 'pumpkin'
pbt_pumpkin.resize = 256
pbt_pumpkin.val_batch_size = 64
pbt_pumpkin.train_num_workers = 1
pbt_pumpkin.val_num_workers = 1
pbt_pumpkin.pin_memory = True
pbt_pumpkin.pretrained_weights = 'DEFAULT'
pbt_pumpkin.resources = {'cpu': 10, 'gpu': 1}
pbt_pumpkin.num_samples = 20
pbt_pumpkin.checkpoint_interval = 60
pbt_pumpkin.stop_time = 3600
pbt_pumpkin.pbt_args = {
    'time_attr': 'time_total_s',
    'perturbation_interval': pbt_pumpkin.checkpoint_interval,
    'hyperparam_mutations': {
        'batch_size': [4, 8, 12, 16, 24, 32, 48, 64],
        'frozen_layers': [0, 1, 2, 3, 4, 5, 6, 7],
        'lr': tune.loguniform(1e-5, 1e-1),
        'weight_decay': tune.uniform(1e-4, 1e0)
    },
    'quantile_fraction': 0.25,
    'resample_probability': 0.25,
    'perturbation_factors': (1.2, 0.8)
}

mean, std = utils.get_seven_scenes_mean_std(pbt_pumpkin.scene, validation=True)
pbt_pumpkin.transform = transforms.Compose([
    transforms.Resize(pbt_pumpkin.resize),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])
pbt_pumpkin.train_dataset_args = {
    'path': pbt_pumpkin.path,
    'scene': pbt_pumpkin.scene,
    'train': True,
    'transform': pbt_pumpkin.transform
}
pbt_pumpkin.test_dataset_args = {
    'path': pbt_pumpkin.path,
    'scene': pbt_pumpkin.scene,
    'train': False,
    'transform': pbt_pumpkin.transform
}

# 7-Scenes redkitchen
pbt_redkitchen = EasyDict()
pbt_redkitchen.output_name = 'pbt_redkitchen'
pbt_redkitchen.dataset = 'SevenScenes'
pbt_redkitchen.path = r'/home/user/Datasets/7-Scenes'
pbt_redkitchen.scene = 'redkitchen'
pbt_redkitchen.resize = 256
pbt_redkitchen.val_batch_size = 64
pbt_redkitchen.train_num_workers = 1
pbt_redkitchen.val_num_workers = 1
pbt_redkitchen.pin_memory = True
pbt_redkitchen.pretrained_weights = 'DEFAULT'
pbt_redkitchen.resources = {'cpu': 10, 'gpu': 1}
pbt_redkitchen.num_samples = 20
pbt_redkitchen.checkpoint_interval = 60
pbt_redkitchen.stop_time = 3600
pbt_redkitchen.pbt_args = {
    'time_attr': 'time_total_s',
    'perturbation_interval': pbt_redkitchen.checkpoint_interval,
    'hyperparam_mutations': {
        'batch_size': [4, 8, 12, 16, 24, 32, 48, 64],
        'frozen_layers': [0, 1, 2, 3, 4, 5, 6, 7],
        'lr': tune.loguniform(1e-5, 1e-1),
        'weight_decay': tune.uniform(1e-4, 1e0)
    },
    'quantile_fraction': 0.25,
    'resample_probability': 0.25,
    'perturbation_factors': (1.2, 0.8)
}

mean, std = utils.get_seven_scenes_mean_std(pbt_redkitchen.scene, validation=True)
pbt_redkitchen.transform = transforms.Compose([
    transforms.Resize(pbt_redkitchen.resize),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])
pbt_redkitchen.train_dataset_args = {
    'path': pbt_redkitchen.path,
    'scene': pbt_redkitchen.scene,
    'train': True,
    'transform': pbt_redkitchen.transform
}
pbt_redkitchen.test_dataset_args = {
    'path': pbt_redkitchen.path,
    'scene': pbt_redkitchen.scene,
    'train': False,
    'transform': pbt_redkitchen.transform
}

# 7-Scenes stairs
pbt_stairs = EasyDict()
pbt_stairs.output_name = 'pbt_stairs'
pbt_stairs.dataset = 'SevenScenes'
pbt_stairs.path = r'/home/user/Datasets/7-Scenes'
pbt_stairs.scene = 'stairs'
pbt_stairs.resize = 256
pbt_stairs.val_batch_size = 64
pbt_stairs.train_num_workers = 1
pbt_stairs.val_num_workers = 1
pbt_stairs.pin_memory = True
pbt_stairs.pretrained_weights = 'DEFAULT'
pbt_stairs.resources = {'cpu': 10, 'gpu': 1}
pbt_stairs.num_samples = 20
pbt_stairs.checkpoint_interval = 60
pbt_stairs.stop_time = 3600
pbt_stairs.pbt_args = {
    'time_attr': 'time_total_s',
    'perturbation_interval': pbt_stairs.checkpoint_interval,
    'hyperparam_mutations': {
        'batch_size': [4, 8, 12, 16, 24, 32, 48, 64],
        'frozen_layers': [0, 1, 2, 3, 4, 5, 6, 7],
        'lr': tune.loguniform(1e-5, 1e-1),
        'weight_decay': tune.uniform(1e-4, 1e0)
    },
    'quantile_fraction': 0.25,
    'resample_probability': 0.25,
    'perturbation_factors': (1.2, 0.8)
}

mean, std = utils.get_seven_scenes_mean_std(pbt_stairs.scene, validation=True)
pbt_stairs.transform = transforms.Compose([
    transforms.Resize(pbt_stairs.resize),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])
pbt_stairs.train_dataset_args = {
    'path': pbt_stairs.path,
    'scene': pbt_stairs.scene,
    'train': True,
    'transform': pbt_stairs.transform
}
pbt_stairs.test_dataset_args = {
    'path': pbt_stairs.path,
    'scene': pbt_stairs.scene,
    'train': False,
    'transform': pbt_stairs.transform
}

# MAGIC Lab
pbt_magiclab = EasyDict()
pbt_magiclab.output_name = 'pbt_magiclab_1115'
pbt_magiclab.dataset = 'MAGICLab'
pbt_magiclab.path = r'/home/user/Datasets/MAGIC Lab'
#pbt_magiclab.xml = 'images_4asec_2487_addMarkers.xml'
pbt_magiclab.xml = 'four_markers_camera_1111.xml'
pbt_magiclab.scene = 'magiclab'
pbt_magiclab.resize = 256
#pbt_magiclab.mean = [0.4554, 0.4445, 0.4162]
#pbt_magiclab.std = [0.2628, 0.2602, 0.2585]
pbt_magiclab.mean = [0.5522, 0.5347, 0.5265]
pbt_magiclab.std = [0.2578, 0.2561, 0.2444]
pbt_magiclab.val_batch_size = 48
pbt_magiclab.train_num_workers = 1
pbt_magiclab.val_num_workers = 1
pbt_magiclab.pin_memory = True
pbt_magiclab.pretrained_weights = 'DEFAULT'
pbt_magiclab.resources = {'cpu': 10, 'gpu': 1}
pbt_magiclab.num_samples = 20
#pbt_magiclab.checkpoint_interval = 60
pbt_magiclab.checkpoint_interval = 30
#pbt_magiclab.stop_time = 3600
pbt_magiclab.stop_time = 1800
pbt_magiclab.pbt_args = {
    'time_attr': 'time_total_s',
    'perturbation_interval': pbt_magiclab.checkpoint_interval,
    'hyperparam_mutations': {
        'batch_size': [4, 8, 12, 16, 24, 32, 48],
        'frozen_layers': [0, 1, 2, 3, 4, 5, 6, 7],
        'lr': tune.loguniform(1e-5, 1e-1),
        'weight_decay': tune.uniform(1e-4, 1e0)
    },
    'quantile_fraction': 0.25,
    'resample_probability': 0.25,
    'perturbation_factors': (1.2, 0.8)
}

pbt_magiclab.transform = transforms.Compose([
    transforms.Resize(pbt_magiclab.resize),
    transforms.ToTensor(),
    transforms.Normalize(mean=pbt_magiclab.mean, std=pbt_magiclab.std)
])
pbt_magiclab.train_dataset_args = {
    'path': pbt_magiclab.path,
    'xml': pbt_magiclab.xml,
    'train': True,
    'transform': pbt_magiclab.transform
}
pbt_magiclab.test_dataset_args = {
    'path': pbt_magiclab.path,
    'xml': pbt_magiclab.xml,
    'train': False,
    'transform': pbt_magiclab.transform
}

# Population Based Training Replay

# 7-Scenes chess
pbt_replay_chess = EasyDict()
pbt_replay_chess.policy_txt = r'ray_results/pbt_chess/pbt_policy_27b13_00005.txt'
pbt_replay_chess.resources = {'cpu': 10, 'gpu': 1}

pbt_replay_chess.dataset = pbt_chess.dataset
pbt_replay_chess.path = pbt_chess.path
pbt_replay_chess.scene = pbt_chess.scene
pbt_replay_chess.resize = pbt_chess.resize
pbt_replay_chess.train_num_workers = pbt_chess.train_num_workers
pbt_replay_chess.test_num_workers = pbt_chess.val_num_workers
pbt_replay_chess.pin_memory = pbt_chess.pin_memory
pbt_replay_chess.pretrained_weights = pbt_chess.pretrained_weights
pbt_replay_chess.checkpoint_interval = pbt_chess.checkpoint_interval
pbt_replay_chess.stop_time = pbt_chess.stop_time
mean, std = utils.get_seven_scenes_mean_std(pbt_replay_chess.scene)
pbt_replay_chess.transform = transforms.Compose([
    transforms.Resize(pbt_replay_chess.resize),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])
pbt_replay_chess.train_dataset_args = {
    'path': pbt_replay_chess.path,
    'scene': pbt_replay_chess.scene,
    'train': True,
    'transform': pbt_replay_chess.transform
}
pbt_replay_chess.test_dataset_args = {
    'path': pbt_replay_chess.path,
    'scene': pbt_replay_chess.scene,
    'train': False,
    'transform': pbt_replay_chess.transform
}

# 7-Scenes fire
pbt_replay_fire = EasyDict()
pbt_replay_fire.policy_txt = r'ray_results/pbt_fire/pbt_policy_e21ce_00018.txt'
pbt_replay_fire.resources = {'cpu': 10, 'gpu': 1}

pbt_replay_fire.dataset = pbt_fire.dataset
pbt_replay_fire.path = pbt_fire.path
pbt_replay_fire.scene = pbt_fire.scene
pbt_replay_fire.resize = pbt_fire.resize
pbt_replay_fire.train_num_workers = pbt_fire.train_num_workers
pbt_replay_fire.test_num_workers = pbt_fire.val_num_workers
pbt_replay_fire.pin_memory = pbt_fire.pin_memory
pbt_replay_fire.pretrained_weights = pbt_fire.pretrained_weights
pbt_replay_fire.checkpoint_interval = pbt_fire.checkpoint_interval
pbt_replay_fire.stop_time = pbt_fire.stop_time
mean, std = utils.get_seven_scenes_mean_std(pbt_replay_fire.scene)
pbt_replay_fire.transform = transforms.Compose([
    transforms.Resize(pbt_replay_fire.resize),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])
pbt_replay_fire.train_dataset_args = {
    'path': pbt_replay_fire.path,
    'scene': pbt_replay_fire.scene,
    'train': True,
    'transform': pbt_replay_fire.transform
}
pbt_replay_fire.test_dataset_args = {
    'path': pbt_replay_fire.path,
    'scene': pbt_replay_fire.scene,
    'train': False,
    'transform': pbt_replay_fire.transform
}

# 7-Scenes heads
pbt_replay_heads = EasyDict()
pbt_replay_heads.policy_txt = r'ray_results/pbt_heads/pbt_policy_c5c93_00019.txt'
pbt_replay_heads.resources = {'cpu': 10, 'gpu': 1}

pbt_replay_heads.dataset = pbt_heads.dataset
pbt_replay_heads.path = pbt_heads.path
pbt_replay_heads.scene = pbt_heads.scene
pbt_replay_heads.resize = pbt_heads.resize
pbt_replay_heads.train_num_workers = pbt_heads.train_num_workers
pbt_replay_heads.test_num_workers = pbt_heads.val_num_workers
pbt_replay_heads.pin_memory = pbt_heads.pin_memory
pbt_replay_heads.pretrained_weights = pbt_heads.pretrained_weights
pbt_replay_heads.checkpoint_interval = pbt_heads.checkpoint_interval
pbt_replay_heads.stop_time = pbt_heads.stop_time
mean, std = utils.get_seven_scenes_mean_std(pbt_replay_heads.scene)
pbt_replay_heads.transform = transforms.Compose([
    transforms.Resize(pbt_replay_heads.resize),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])
pbt_replay_heads.train_dataset_args = {
    'path': pbt_replay_heads.path,
    'scene': pbt_replay_heads.scene,
    'train': True,
    'transform': pbt_replay_heads.transform
}
pbt_replay_heads.test_dataset_args = {
    'path': pbt_replay_heads.path,
    'scene': pbt_replay_heads.scene,
    'train': False,
    'transform': pbt_replay_heads.transform
}

# 7-Scenes office
pbt_replay_office = EasyDict()
pbt_replay_office.policy_txt = r'ray_results/pbt_office/pbt_policy_1159f_00006.txt'
pbt_replay_office.resources = {'cpu': 10, 'gpu': 1}

pbt_replay_office.dataset = pbt_office.dataset
pbt_replay_office.path = pbt_office.path
pbt_replay_office.scene = pbt_office.scene
pbt_replay_office.resize = pbt_office.resize
pbt_replay_office.train_num_workers = pbt_office.train_num_workers
pbt_replay_office.test_num_workers = pbt_office.val_num_workers
pbt_replay_office.pin_memory = pbt_office.pin_memory
pbt_replay_office.pretrained_weights = pbt_office.pretrained_weights
pbt_replay_office.checkpoint_interval = pbt_office.checkpoint_interval
pbt_replay_office.stop_time = pbt_office.stop_time
mean, std = utils.get_seven_scenes_mean_std(pbt_replay_office.scene)
pbt_replay_office.transform = transforms.Compose([
    transforms.Resize(pbt_replay_office.resize),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])
pbt_replay_office.train_dataset_args = {
    'path': pbt_replay_office.path,
    'scene': pbt_replay_office.scene,
    'train': True,
    'transform': pbt_replay_office.transform
}
pbt_replay_office.test_dataset_args = {
    'path': pbt_replay_office.path,
    'scene': pbt_replay_office.scene,
    'train': False,
    'transform': pbt_replay_office.transform
}

# 7-Scenes pumpkin
pbt_replay_pumpkin = EasyDict()
pbt_replay_pumpkin.policy_txt = r'ray_results/pbt_pumpkin/pbt_policy_265fa_00018.txt'
pbt_replay_pumpkin.resources = {'cpu': 10, 'gpu': 1}

pbt_replay_pumpkin.dataset = pbt_pumpkin.dataset
pbt_replay_pumpkin.path = pbt_pumpkin.path
pbt_replay_pumpkin.scene = pbt_pumpkin.scene
pbt_replay_pumpkin.resize = pbt_pumpkin.resize
pbt_replay_pumpkin.train_num_workers = pbt_pumpkin.train_num_workers
pbt_replay_pumpkin.test_num_workers = pbt_pumpkin.val_num_workers
pbt_replay_pumpkin.pin_memory = pbt_pumpkin.pin_memory
pbt_replay_pumpkin.pretrained_weights = pbt_pumpkin.pretrained_weights
pbt_replay_pumpkin.checkpoint_interval = pbt_pumpkin.checkpoint_interval
pbt_replay_pumpkin.stop_time = pbt_pumpkin.stop_time
mean, std = utils.get_seven_scenes_mean_std(pbt_replay_pumpkin.scene)
pbt_replay_pumpkin.transform = transforms.Compose([
    transforms.Resize(pbt_replay_pumpkin.resize),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])
pbt_replay_pumpkin.train_dataset_args = {
    'path': pbt_replay_pumpkin.path,
    'scene': pbt_replay_pumpkin.scene,
    'train': True,
    'transform': pbt_replay_pumpkin.transform
}
pbt_replay_pumpkin.test_dataset_args = {
    'path': pbt_replay_pumpkin.path,
    'scene': pbt_replay_pumpkin.scene,
    'train': False,
    'transform': pbt_replay_pumpkin.transform
}

# 7-Scenes redkitchen
pbt_replay_redkitchen = EasyDict()
pbt_replay_redkitchen.policy_txt = r'ray_results/pbt_redkitchen/pbt_policy_16a5c_00019.txt'
pbt_replay_redkitchen.resources = {'cpu': 10, 'gpu': 1}

pbt_replay_redkitchen.dataset = pbt_redkitchen.dataset
pbt_replay_redkitchen.path = pbt_redkitchen.path
pbt_replay_redkitchen.scene = pbt_redkitchen.scene
pbt_replay_redkitchen.resize = pbt_redkitchen.resize
pbt_replay_redkitchen.train_num_workers = pbt_redkitchen.train_num_workers
pbt_replay_redkitchen.test_num_workers = pbt_redkitchen.val_num_workers
pbt_replay_redkitchen.pin_memory = pbt_redkitchen.pin_memory
pbt_replay_redkitchen.pretrained_weights = pbt_redkitchen.pretrained_weights
pbt_replay_redkitchen.checkpoint_interval = pbt_redkitchen.checkpoint_interval
pbt_replay_redkitchen.stop_time = pbt_redkitchen.stop_time
mean, std = utils.get_seven_scenes_mean_std(pbt_replay_redkitchen.scene)
pbt_replay_redkitchen.transform = transforms.Compose([
    transforms.Resize(pbt_replay_redkitchen.resize),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])
pbt_replay_redkitchen.train_dataset_args = {
    'path': pbt_replay_redkitchen.path,
    'scene': pbt_replay_redkitchen.scene,
    'train': True,
    'transform': pbt_replay_redkitchen.transform
}
pbt_replay_redkitchen.test_dataset_args = {
    'path': pbt_replay_redkitchen.path,
    'scene': pbt_replay_redkitchen.scene,
    'train': False,
    'transform': pbt_replay_redkitchen.transform
}

# 7-Scenes stairs
pbt_replay_stairs = EasyDict()
pbt_replay_stairs.policy_txt = r'ray_results/pbt_stairs/pbt_policy_43002_00002.txt'
pbt_replay_stairs.resources = {'cpu': 10, 'gpu': 1}

pbt_replay_stairs.dataset = pbt_stairs.dataset
pbt_replay_stairs.path = pbt_stairs.path
pbt_replay_stairs.scene = pbt_stairs.scene
pbt_replay_stairs.resize = pbt_stairs.resize
pbt_replay_stairs.train_num_workers = pbt_stairs.train_num_workers
pbt_replay_stairs.test_num_workers = pbt_stairs.val_num_workers
pbt_replay_stairs.pin_memory = pbt_stairs.pin_memory
pbt_replay_stairs.pretrained_weights = pbt_stairs.pretrained_weights
pbt_replay_stairs.checkpoint_interval = pbt_stairs.checkpoint_interval
pbt_replay_stairs.stop_time = pbt_stairs.stop_time
mean, std = utils.get_seven_scenes_mean_std(pbt_replay_stairs.scene)
pbt_replay_stairs.transform = transforms.Compose([
    transforms.Resize(pbt_replay_stairs.resize),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])
pbt_replay_stairs.train_dataset_args = {
    'path': pbt_replay_stairs.path,
    'scene': pbt_replay_stairs.scene,
    'train': True,
    'transform': pbt_replay_stairs.transform
}
pbt_replay_stairs.test_dataset_args = {
    'path': pbt_replay_stairs.path,
    'scene': pbt_replay_stairs.scene,
    'train': False,
    'transform': pbt_replay_stairs.transform
}

# MAGIC Lab
pbt_replay_magiclab = EasyDict()
#pbt_replay_magiclab.policy_txt = r'ray_results/pbt_magiclab_1024/pbt_policy_e7b20_00008.txt'
pbt_replay_magiclab.policy_txt = r'ray_results/pbt_magiclab_1115/pbt_policy_984a0_00003.txt'
#pbt_replay_magiclab.mean = [0.4554, 0.4448, 0.4164]
#pbt_replay_magiclab.std = [0.2616, 0.2590, 0.2575]
pbt_replay_magiclab.mean = [0.5543, 0.5372, 0.5282]
pbt_replay_magiclab.std = [0.2555, 0.2539, 0.2427]
pbt_replay_magiclab.resources = {'cpu': 10, 'gpu': 1}

pbt_replay_magiclab.dataset = pbt_magiclab.dataset
pbt_replay_magiclab.path = pbt_magiclab.path
pbt_replay_magiclab.xml = pbt_magiclab.xml
pbt_replay_magiclab.resize = pbt_magiclab.resize
pbt_replay_magiclab.train_num_workers = pbt_magiclab.train_num_workers
pbt_replay_magiclab.test_num_workers = pbt_magiclab.val_num_workers
pbt_replay_magiclab.pin_memory = pbt_magiclab.pin_memory
pbt_replay_magiclab.pretrained_weights = pbt_magiclab.pretrained_weights
pbt_replay_magiclab.checkpoint_interval = pbt_magiclab.checkpoint_interval
pbt_replay_magiclab.stop_time = pbt_magiclab.stop_time
pbt_replay_magiclab.transform = transforms.Compose([
    transforms.Resize(pbt_replay_magiclab.resize),
    transforms.ToTensor(),
    transforms.Normalize(mean=pbt_replay_magiclab.mean, std=pbt_replay_magiclab.std)
])
pbt_replay_magiclab.train_dataset_args = {
    'path': pbt_replay_magiclab.path,
    'xml': pbt_replay_magiclab.xml,
    'train': True,
    'transform': pbt_replay_magiclab.transform
}
pbt_replay_magiclab.test_dataset_args = {
    'path': pbt_replay_magiclab.path,
    'xml': pbt_replay_magiclab.xml,
    'train': False,
    'transform': pbt_replay_magiclab.transform
}

# Baseline

# MAGIC Lab
baseline_magiclab = EasyDict()
baseline_magiclab.dataset = 'MAGICLab'
baseline_magiclab.path = r'/home/user/Datasets/MAGIC Lab'
#baseline_magiclab.xml = 'images_4asec_2487_addMarkers.xml'
baseline_magiclab.xml = 'four_markers_camera_1111.xml'
baseline_magiclab.resize = 256
#baseline_magiclab.mean = [0.4554, 0.4448, 0.4164]
#baseline_magiclab.std = [0.2616, 0.2590, 0.2575]
baseline_magiclab.mean = [0.5543, 0.5372, 0.5282]
baseline_magiclab.std = [0.2555, 0.2539, 0.2427]
baseline_magiclab.batch_size = 48
baseline_magiclab.train_num_workers = 1
baseline_magiclab.test_num_workers = 1
baseline_magiclab.pin_memory = True
baseline_magiclab.pretrained_weights = 'DEFAULT'
baseline_magiclab.frozen_layers = 0
baseline_magiclab.lr = 1e-4
baseline_magiclab.weight_decay = 1e-2
#baseline_magiclab.epochs = 300
baseline_magiclab.epochs = 600

baseline_magiclab.transform = transforms.Compose([
    transforms.Resize(baseline_magiclab.resize),
    transforms.ToTensor(),
    transforms.Normalize(mean=baseline_magiclab.mean, std=baseline_magiclab.std)
])
baseline_magiclab.train_dataset_args = {
    'path': baseline_magiclab.path,
    'xml': baseline_magiclab.xml,
    'train': True,
    'transform': baseline_magiclab.transform
}
baseline_magiclab.test_dataset_args = {
    'path': baseline_magiclab.path,
    'xml': baseline_magiclab.xml,
    'train': False,
    'transform': baseline_magiclab.transform
}
