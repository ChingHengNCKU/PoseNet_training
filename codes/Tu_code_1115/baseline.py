import argparse
import importlib
import os
import time
import datetime
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

from modules.criterion import PoseLoss
from modules.model import PoseNet
from modules import utils

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
    output_name = args.output_name
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create directory

    root = os.path.join('results', output_name)
    if not os.path.isdir('results'):
        os.mkdir('results')
    if not os.path.isdir(root):
        os.mkdir(root)

    # Load training data

    print('Load training data...')
    datasets = importlib.import_module('modules.datasets')
    Dataset = getattr(datasets, cfg.dataset)
    trainset = Dataset(**cfg.train_dataset_args)
    trainloader = DataLoader(trainset, batch_size=cfg.batch_size, shuffle=True,
                             num_workers=cfg.train_num_workers, pin_memory=cfg.pin_memory,
                             drop_last=True)

    # Load model

    posenet = PoseNet(weights=cfg.pretrained_weights)
    posenet = utils.freeze_posenet(posenet, frozen_layers=cfg.frozen_layers)
    posenet.to(device)

    # Training settings

    criterion = PoseLoss()
    criterion.to(device)
    params = list(posenet.parameters()) + list(criterion.parameters())
    optimizer = optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(trainloader)*cfg.epochs)

    # Start training

    loss_history = []
    error_history = []
    s_history = []
    t_start = time.time()
    print(f'Start training {output_name}...')
    for i in range(cfg.epochs):
        print(f'Epoch {i+1}/{cfg.epochs}')
        for j, data in enumerate(trainloader):
            inputs = data['image'].to(device)
            poses = data['pose'].to(device)

            # Train

            optimizer.zero_grad()
            outputs = posenet(inputs)
            loss = criterion(outputs, poses)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Calculate errors

            with torch.no_grad():
                position_errors, orientation_errors = utils.cal_pose_error(outputs, poses)
                pme, ome = position_errors.median(), orientation_errors.median()

            # Record history

            loss_history.append(loss.item())
            error_history.append([pme.item(), ome.item()])
            s_history.append([criterion.s_x.item(), criterion.s_q.item()])
            utils.print_loss_and_error(j+1, len(trainloader), loss_history[-1], error_history[-1])

    # Record training time

    t_end = time.time()
    t = t_end - t_start
    with open(os.path.join(root, f'{output_name}.train_time.txt'), 'w') as f:
        f.write(str(datetime.timedelta(seconds=t)))

    # Save model

    torch.save({'model': posenet.state_dict()}, os.path.join(root, f'{output_name}.pt'))

    # Output plots

    utils.output_plot(os.path.join(root, f'{output_name}.loss'), loss_history,
                      ylabel='Loss', linewidth=0.5)
    utils.output_plot(os.path.join(root, f'{output_name}.pme'), np.array(error_history)[:, 0],
                      ylabel='Position Median Error (m)', ylim=[0, 0.5], linewidth=0.5)
    utils.output_plot(os.path.join(root, f'{output_name}.ome'), np.array(error_history)[:, 1],
                      ylabel='Orientation Median Error (degree)', ylim=[0, 20], linewidth=0.5)
    utils.output_plot(os.path.join(root, f'{output_name}.s_x'), np.array(s_history)[:, 0],
                      ylabel='s_x')
    utils.output_plot(os.path.join(root, f'{output_name}.s_q'), np.array(s_history)[:, 1],
                      ylabel='s_q')

    # Load testing data

    print('Load testing data...')
    testset = Dataset(**cfg.test_dataset_args)
    testloader = DataLoader(testset, batch_size=1, shuffle=False,
                            num_workers=cfg.test_num_workers, pin_memory=cfg.pin_memory)

    # Start testing

    errors = []
    total_inference_time = 0
    posenet.eval()
    print(f'Start testing {output_name}...')
    with torch.no_grad():
        for data in testloader:
            input = data['image'].to(device)
            pose = data['pose'].to(device)

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

    np.savetxt(os.path.join(root, f'{output_name}.test_pme.txt'), median_error[:1])
    np.savetxt(os.path.join(root, f'{output_name}.test_ome.txt'), median_error[1:])
    with open(os.path.join(root, f'{output_name}.infer_time.txt'), 'w') as f:
        f.write(f'{inference_time:.3f} ms')
    print(f'Testing median error: {median_error[0]:.3f} m, {median_error[1]:.3f} degree')
    print(f'Inference time: {inference_time:.3f} ms')

if __name__ == '__main__':
    main()
