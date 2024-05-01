import argparse
import importlib
import os
import time
import pickle
import numpy as np
import torch
import ray
from torch import optim
from torch.utils.data import DataLoader
from ray import air, tune
from ray.tune.schedulers import PopulationBasedTrainingReplay

from modules.criterion import PoseLoss
from modules.model import PoseNet
from modules import utils

class Trainable(tune.Trainable):
    """
    """

    def setup(self, hyperparams, trainset, cfg):
        """
        """
        self.cfg = cfg

        # Load training data

        self.trainloader = DataLoader(trainset, batch_size=hyperparams['batch_size'], shuffle=True,
                                      num_workers=cfg.train_num_workers, pin_memory=cfg.pin_memory,
                                      drop_last=True)

        # Load model

        posenet = PoseNet(weights=cfg.pretrained_weights)
        posenet = utils.freeze_posenet(posenet, frozen_layers=hyperparams['frozen_layers'])
        posenet.to(cfg.device)
        self.posenet = posenet

        # Training settings

        criterion = PoseLoss()
        criterion.to(cfg.device)
        params = list(posenet.parameters()) + list(criterion.parameters())
        optimizer = optim.AdamW(params, lr=hyperparams['lr'], weight_decay=hyperparams['weight_decay'])
        self.criterion = criterion
        self.optimizer = optimizer
        self.loss_history = []
        self.error_history = []
        self.s_history = []

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
        self.loss_history = checkpoint['loss_history']
        self.error_history = checkpoint['error_history']
        self.s_history = checkpoint['s_history']

    def step(self):
        """
        """
        t_start = time.time()
        while True:
            for data in self.trainloader:
                inputs = data['image'].to(self.cfg.device)
                poses = data['pose'].to(self.cfg.device)

                # Train

                self.optimizer.zero_grad()
                outputs = self.posenet(inputs)
                loss = self.criterion(outputs, poses)
                loss.backward()
                self.optimizer.step()

                # Calculate errors

                with torch.no_grad():
                    position_errors, orientation_errors = utils.cal_pose_error(outputs, poses)
                    pme, ome = position_errors.median(), orientation_errors.median()

                # Record history

                self.loss_history.append(loss.item())
                self.error_history.append([pme.item(), ome.item()])
                self.s_history.append([self.criterion.s_x.item(), self.criterion.s_q.item()])

                if time.time()-t_start > self.cfg.checkpoint_interval:
                    return {'loss': loss.item(), 'pme': pme.item(), 'ome': ome.item()}

    def save_checkpoint(self, checkpoint_dir):
        """
        """
        torch.save({
            'model': self.posenet.state_dict(),
            'criterion': self.criterion.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'loss_history': self.loss_history,
            'error_history': self.error_history,
            's_history': self.s_history
        }, os.path.join(checkpoint_dir, 'checkpoint.pt'))
        return checkpoint_dir

def main():
    """
    Args:
        config (str): configuration name in config.py.
        output_name (str): name for output files.
    """

    # Import configuration

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='configuration name in config.py',
                        metavar='str', required=True)
    parser.add_argument('-o', '--output_name', help='name for output files',
                        metavar='str', required=True)
    args = parser.parse_args()
    configs = importlib.import_module('modules.configs')
    cfg = getattr(configs, args.config)
    cfg.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_name = args.output_name

    # Create directory

    if not os.path.isdir('results'):
        os.mkdir('results')

    # Load training dataset

    datasets = importlib.import_module('modules.datasets')
    Dataset = getattr(datasets, cfg.dataset)
    trainset = Dataset(**cfg.train_dataset_args)

    # PBT Replay

    if ray.is_initialized():
        ray.shutdown()
    ray.init()
    trainable = tune.with_parameters(Trainable, trainset=trainset, cfg=cfg)
    trainable = tune.with_resources(trainable, cfg.resources)
    replay = PopulationBasedTrainingReplay(cfg.policy_txt)
    tuner = tune.Tuner(
        trainable=trainable,
        tune_config=tune.TuneConfig(
            scheduler=replay
        ),
        run_config=air.RunConfig(
            name=output_name,
            local_dir='results',
            stop={'time_total_s': cfg.stop_time}
        )
    )
    result = tuner.fit()

    # Save the PBT result

    dir_path = os.path.join('results', output_name)
    with open(os.path.join(dir_path, 'result.pkl'), 'wb') as f:
        pickle.dump(result, f)

    # Load pt file

    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file == 'checkpoint.pt':
                checkpoint = torch.load(os.path.join(root, 'checkpoint.pt'))
                posenet = PoseNet()
                posenet.to(cfg.device)
                posenet.load_state_dict(checkpoint['model'])
                loss_history = checkpoint['loss_history']
                error_history = checkpoint['error_history']
                s_history = checkpoint['s_history']

    # Output plots

    utils.output_plot(os.path.join(dir_path, f'{output_name}.loss'), loss_history,
                      ylabel='Loss', linewidth=0.5)
    utils.output_plot(os.path.join(dir_path, f'{output_name}.pme'), np.array(error_history)[:, 0],
                      ylabel='Position Median Error (m)', ylim=[0, 0.5], linewidth=0.5)
    utils.output_plot(os.path.join(dir_path, f'{output_name}.ome'), np.array(error_history)[:, 1],
                      ylabel='Orientation Median Error (degree)', ylim=[0, 20], linewidth=0.5)
    utils.output_plot(os.path.join(dir_path, f'{output_name}.s_x'), np.array(s_history)[:, 0],
                      ylabel='s_x')
    utils.output_plot(os.path.join(dir_path, f'{output_name}.s_q'), np.array(s_history)[:, 1],
                      ylabel='s_q')

    # Load testing data

    testset = Dataset(**cfg.test_dataset_args)
    testloader = DataLoader(testset, batch_size=1, shuffle=False,
                            num_workers=cfg.test_num_workers, pin_memory=cfg.pin_memory)

    # Start testing

    errors = []
    total_inference_time = 0
    posenet.eval()
    with torch.no_grad():
        for data in testloader:
            input = data['image'].to(cfg.device)
            pose = data['pose'].to(cfg.device)

            # Test

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            output = posenet(input)
            end_event.record()
            torch.cuda.synchronize()
            inference_time = start_event.elapsed_time(end_event)
            total_inference_time += inference_time

            # Calculate errors

            position_error, orientation_error = utils.cal_pose_error(output, pose)
            errors.append([position_error.item(), orientation_error.item()])
    errors = np.array(errors)
    median_error = np.median(errors, axis=0)
    inference_time = total_inference_time / len(testloader)

    # Output testing results

    np.savetxt(os.path.join(dir_path, f'{output_name}.test_pme.txt'), median_error[:1])
    np.savetxt(os.path.join(dir_path, f'{output_name}.test_ome.txt'), median_error[1:])
    with open(os.path.join(dir_path, f'{output_name}.infer_time.txt'), 'w') as f:
        f.write(f'{inference_time:.3f} ms')

if __name__ == '__main__':
    main()
