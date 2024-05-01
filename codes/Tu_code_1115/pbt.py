import argparse
import importlib
import os
import time
import pickle
import json
import numpy as np
import torch
import ray
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from torch import optim
from torch.utils.data import DataLoader
from ray import air, tune
from ray.tune.schedulers import PopulationBasedTraining

from modules.criterion import PoseLoss
from modules.model import PoseNet
from modules import utils

class Trainable(tune.Trainable):
    """
    """

    def setup(self, hyperparams, cfg):
        """
        """
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        datasets = importlib.import_module('modules.datasets')
        Dataset = getattr(datasets, cfg.dataset)
        dataset = Dataset(**cfg.train_dataset_args)
        trainset, validationset = utils.split_trainset(dataset, cfg.scene)
        self.trainset = trainset
        self.validationset = validationset
        self.initialize(hyperparams, trainset, validationset, cfg)

    def initialize(self, hyperparams, trainset, validationset, cfg):
        """
        """

        # Load data

        self.trainloader = DataLoader(trainset, batch_size=hyperparams['batch_size'], shuffle=True,
                                      num_workers=cfg.train_num_workers, pin_memory=cfg.pin_memory,
                                      drop_last=True)
        self.validationloader = DataLoader(validationset, batch_size=cfg.val_batch_size, shuffle=True,
                                           num_workers=cfg.val_num_workers, pin_memory=cfg.pin_memory,
                                           drop_last=True)

        # Load model

        posenet = PoseNet(weights=cfg.pretrained_weights)
        posenet = utils.freeze_posenet(posenet, frozen_layers=hyperparams['frozen_layers'])
        posenet.to(self.device)
        self.posenet = posenet

        # Training settings

        criterion = PoseLoss()
        criterion.to(self.device)
        params = list(posenet.parameters()) + list(criterion.parameters())
        optimizer = optim.AdamW(params, lr=hyperparams['lr'], weight_decay=hyperparams['weight_decay'])
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loss_history = []
        self.train_error_history = []
        self.s_history = []
        self.val_iter_history = []
        self.val_loss_history = []
        self.val_error_history = []

    def load_checkpoint(self, checkpoint_dir):
        """
        """
        hyperparams = self.get_config()
        checkpoint = torch.load(os.path.join(checkpoint_dir, 'checkpoint.pt'))
        self.posenet.load_state_dict(checkpoint['model'])
        self.criterion.load_state_dict(checkpoint['criterion'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = hyperparams['lr']
            param_group['weight_decay'] = hyperparams['weight_decay']
        self.train_loss_history = checkpoint['train_loss_history']
        self.train_error_history = checkpoint['train_error_history']
        self.s_history = checkpoint['s_history']
        self.val_iter_history = checkpoint['val_iter_history']
        self.val_loss_history = checkpoint['val_loss_history']
        self.val_error_history = checkpoint['val_error_history']

    def step(self):
        """
        """
        t_start = time.time()
        while True:
            for data in self.trainloader:
                inputs = data['image'].to(self.device)
                poses = data['pose'].to(self.device)

                # Train

                self.optimizer.zero_grad()
                outputs = self.posenet(inputs)
                loss = self.criterion(outputs, poses)
                loss.backward()
                self.optimizer.step()

                # Calculate training errors

                with torch.no_grad():
                    position_errors, orientation_errors = utils.cal_pose_error(outputs, poses)
                    pme, ome = position_errors.median(), orientation_errors.median()

                # Record history

                self.train_loss_history.append(loss.item())
                self.train_error_history.append([pme.item(), ome.item()])
                self.s_history.append([self.criterion.s_x.item(), self.criterion.s_q.item()])

                # Validate

                if time.time()-t_start > self.cfg.checkpoint_interval:
                    losses = torch.tensor([], device=self.device)
                    errors = torch.tensor([], device=self.device)
                    self.posenet.eval()
                    with torch.no_grad():
                        for data in self.validationloader:
                            inputs = data['image'].to(self.device)
                            poses = data['pose'].to(self.device)
                            outputs = self.posenet(inputs)

                            # Calculate validation loss and errors

                            loss = self.criterion(outputs, poses)
                            losses = torch.cat((losses, loss.unsqueeze(0)))
                            position_errors, orientation_errors = utils.cal_pose_error(outputs, poses)
                            errors = torch.cat((
                                errors,
                                torch.stack((position_errors, orientation_errors), dim=1)
                            ))

                        # Record history

                        loss = losses.mean()
                        pme = errors[:, 0].median()
                        ome = errors[:, 1].median()
                        self.val_iter_history.append(len(self.train_loss_history))
                        self.val_loss_history.append(loss.item())
                        self.val_error_history.append([pme.item(), ome.item()])
                    self.posenet.train()

                    # Return the metric

                    if pme.isnan():
                        return {'position_error': 1e8}
                    return {'position_error': pme.item()}

    def save_checkpoint(self, checkpoint_dir):
        """
        """
        torch.save({
            'model': self.posenet.state_dict(),
            'criterion': self.criterion.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'train_loss_history': self.train_loss_history,
            'train_error_history': self.train_error_history,
            's_history': self.s_history,
            'val_iter_history': self.val_iter_history,
            'val_loss_history': self.val_loss_history,
            'val_error_history': self.val_error_history
        }, os.path.join(checkpoint_dir, 'checkpoint.pt'))
        return checkpoint_dir

    def reset_config(self, new_hyperparams):
        """
        """
        self.initialize(new_hyperparams, self.trainset, self.validationset, self.cfg)
        return True

def plot_results(results, cfg):
    """
    """
    path = os.path.join('ray_results', cfg.output_name)

    # Plot histories

    for root, dirs, files in os.walk(path):
        for file in files:
            if file == 'checkpoint.pt':
                checkpoint = torch.load(os.path.join(root, 'checkpoint.pt'))
                train_loss_history = checkpoint['train_loss_history']
                train_error_history = checkpoint['train_error_history']
                s_history = checkpoint['s_history']
                val_iter_history = checkpoint['val_iter_history']
                val_loss_history = checkpoint['val_loss_history']
                val_error_history = checkpoint['val_error_history']
                utils.output_plot_2(os.path.join(root, 'loss'), train_loss_history,
                                    val_iter_history, val_loss_history,
                                    ylabel='Loss')
                utils.output_plot_2(os.path.join(root, 'pme'), np.array(train_error_history)[:, 0],
                                    val_iter_history, np.array(val_error_history)[:, 0],
                                    ylabel='Position Median Error (m)', ylim=[0, 0.5])
                utils.output_plot_2(os.path.join(root, 'ome'), np.array(train_error_history)[:, 1],
                                    val_iter_history, np.array(val_error_history)[:, 1],
                                    ylabel='Orientation Median Error (degree)', ylim=[0, 20])
                utils.output_plot(os.path.join(root, 's_x'), np.array(s_history)[:, 0],
                                  ylabel='s_x')
                utils.output_plot(os.path.join(root, 's_q'), np.array(s_history)[:, 1],
                                  ylabel='s_q')

    # Plot hyperparameters

    hyperparams = []
    x = []
    keys = ['batch_size', 'frozen_layers', 'lr', 'weight_decay']
    names = ['Batch Size', 'Frozen Layers', 'Learning Rate', 'Weight Decay']
    df = results.get_dataframe().sort_values(by='position_error', ascending=False)
    for trial_id in df['trial_id']:
        with open(os.path.join(path, f'pbt_policy_{trial_id}.txt'), 'r') as f:
            policies = [json.loads(line) for line in f]
        hyperparams.append({})
        for key in keys:
            hyperparams[-1][key] = [policies[0][-2][key]] + [policy[-1][key] for policy in policies]
        x.append([i+1 for i in range(len(policies)+1)])
    for key, name in zip(keys, names):
        for i in range(cfg.num_samples):
            plt.plot(x[i], hyperparams[i][key], color='black', alpha=(1/cfg.num_samples)*(i+1))
        plt.ylabel(name)
        plt.savefig(os.path.join(path, f'{key}.png'))
        plt.close()

def main():
    """
    Args:
        config (str): configuration name in config.py.
    """

    # Import configuration

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='configuration name in config.py',
                        metavar='str', required=True)
    args = parser.parse_args()
    configs = importlib.import_module('modules.configs')
    cfg = getattr(configs, args.config)

    # Create directory

    if not os.path.isdir('ray_results'):
        os.mkdir('ray_results')

    # Tuning

    if ray.is_initialized():
        ray.shutdown()
    ray.init()
    trainable = tune.with_parameters(Trainable, cfg=cfg)
    trainable = tune.with_resources(trainable, cfg.resources)
    scheduler = PopulationBasedTraining(**cfg.pbt_args)
    tuner = tune.Tuner(
        trainable=trainable,
        tune_config=tune.TuneConfig(
            mode='min',
            metric='position_error',
            scheduler=scheduler,
            num_samples=cfg.num_samples,
            reuse_actors=True
        ),
        run_config=air.RunConfig(
            name=cfg.output_name,
            local_dir='ray_results',
            stop={'time_total_s': cfg.stop_time}
        )
    )
    results = tuner.fit()

    # Save tuning results

    with open(os.path.join('ray_results', cfg.output_name, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    with open(os.path.join('ray_results', cfg.output_name, 'best_result.json'), 'w') as f:
        json.dump(results.get_best_result().metrics, f)

    # Plot results

    plot_results(results, cfg)

if __name__ == '__main__':
    main()
