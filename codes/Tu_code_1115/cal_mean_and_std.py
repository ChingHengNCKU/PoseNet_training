import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from modules.datasets import SevenScenes, MAGICLab
from modules import utils

def main():
#    path = r'/home/user/Datasets/7-Scenes'
#    scene = 'chess'

    path = r'/home/user/Datasets/MAGIC Lab'
    xml = 'four_markers_camera_1111.xml'
    scene = 'magiclab'

    resize = 256
    batch_size = 64
    num_workers = 1
    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor()
    ])

#    trainset = SevenScenes(path, scene, train=True, transform=transform)
#    dataset = SevenScenes(path, scene, train=True, transform=transform)
#    trainset, validationset = utils.split_trainset(dataset, scene)

#    trainset = MAGICLab(path, xml, train=True, transform=transform)
    dataset = MAGICLab(path, xml, train=True, transform=transform)
    trainset, validationset = utils.split_trainset(dataset, scene)

    dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    means = torch.tensor([])
    stds = torch.tensor([])
    for data in dataloader:
        std, mean = torch.std_mean(data['image'], dim=(2, 3))
        means = torch.cat((means, mean))
        stds = torch.cat((stds, std))

    print(f'mean: {means.mean(dim=0)}, std: {stds.mean(dim=0)}')

if __name__ == '__main__':
    main()
