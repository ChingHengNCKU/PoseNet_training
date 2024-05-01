import os
import xml.etree.ElementTree as ET
import numpy as np
import torch
from zipfile import ZipFile
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from kornia.geometry import conversions

class SevenScenes(Dataset):
    """7-Scenes Dataset.

    Attributes:
        imgs (torch.Tensor): Images.
        poses (torch.Tensor): Camera poses.
    """

    def __init__(self, path, scene, train=True, transform=transforms.ToTensor()):
        """
        Args:
            path (str): Path of 7-Scenes dataset.
            scene (str): Scene in 7-Scenes dataset.
            train (bool): If True, return training data; else, return testing data.
            transform (torchvision.transforms): Image transform.
        """
        dir_path = os.path.join(path, scene)
        zip_paths = [os.path.join(dir_path, file) for file in os.listdir(dir_path)
                     if os.path.splitext(file)[1] == '.zip']
        for zip_path in zip_paths:
            with ZipFile(zip_path, 'r') as zf:
                unzipped = True
                files = zf.namelist()
                for file in files:
                    if os.path.basename(file) == 'Thumbs.db':
                        files.remove(file)
                    elif os.path.basename(file) == '':
                        if not os.path.isdir(os.path.join(dir_path, file)):
                            unzipped = False
                    else:
                        if not os.path.isfile(os.path.join(dir_path, file)):
                            unzipped = False
                if not unzipped:
                    zf.extractall(path=dir_path, members=files)

        seq_paths = []
        seq_frames = []
        if train:
            split_txt_path = os.path.join(dir_path, 'TrainSplit.txt')
        else:
            split_txt_path = os.path.join(dir_path, 'TestSplit.txt')
        with open(split_txt_path, 'r') as f:
            for line in f:
                seq_paths.append(os.path.join(dir_path, f'seq-{int(line.split("sequence")[-1]):02d}'))
                seq_frames.append(int(len(os.listdir(seq_paths[-1]))/3))

        total_frames = sum(seq_frames)
        img_temp = Image.open(os.path.join(seq_paths[0], 'frame-000000.color.png'))
        img_temp = transform(img_temp)
        height, width = img_temp.shape[-2:]
        self.imgs = torch.empty(total_frames, 3, height, width)
        self.poses = torch.empty(total_frames, 6)

        index = 0
        for i, seq_path in enumerate(seq_paths):
            for frame in range(seq_frames[i]):
                frame_name = f'frame-{frame:06d}'
                img_path = os.path.join(seq_path, f'{frame_name}.color.png')
                img = Image.open(img_path)
                img = transform(img)
                self.imgs[index] = img
                pose_txt_path = os.path.join(seq_path, f'{frame_name}.pose.txt')
                T = np.loadtxt(pose_txt_path, dtype='float32') # homogeneous transformation matrix (camera to world)
                R = torch.tensor(T[:3, :3]) # rotation matrix
                t = torch.tensor(T[:3, 3]) # translation vector
                q = conversions.rotation_matrix_to_quaternion(R) # quaternion
                log_q = conversions.quaternion_exp_to_log(q) # log quaternion
                pose = torch.cat((t, log_q))
                self.poses[index] = pose
                index += 1

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Input index.

        Returns:
            {'image' (torch.Tensor): Image,
             'pose' (torch.Tensor): Pose of the image.}

        Shape:
            {'image': (3, height, width),
             'pose': (6).}
        """
        return {'image': self.imgs[idx],
                'pose': self.poses[idx]}

class MAGICLab(Dataset):
    """MAGIC Lab Dataset.

    Attributes:
        imgs (torch.Tensor): Images.
        poses (torch.Tensor): Camera poses.
    """

    def __init__(self, path, xml, train=True, transform=transforms.ToTensor()):
        """
        Args:
            path (str): Path of MAGIC Lab dataset.
            xml (str): 
            train (bool): If True, return training data; else, return testing data.
            transform (torchvision.transforms): Image transform.
        """
        self.imgs = []
        self.poses = []

        tree = ET.parse(os.path.join(path, xml))
        root = tree.getroot()
        rotation = root.find('./chunk/components/component/transform/rotation')
        rotation = np.array(rotation.text.split(), dtype='float32').reshape(3, 3)
        translation = root.find('./chunk/components/component/transform/translation')
        translation = np.array(translation.text.split(), dtype='float32').reshape(3, 1)
        scale = root.find('./chunk/components/component/transform/scale')
        scale = np.array(scale.text, dtype='float32')
        transformation = np.hstack((scale*rotation, translation))
        transformation = np.vstack((transformation, np.array([0, 0, 0, 1], dtype='float32')))

        cameras = root.findall('.//camera/[transform]')
        if train:
            cameras = cameras[:-60]
        else:
            cameras = cameras[-60:]
        for camera in cameras:
            img_name = f'{camera.get("label")}.jpg'
            img_path = os.path.join(path, 'images', img_name)
            img = Image.open(img_path)
            img = transform(img)
            self.imgs.append(img)
            T = np.array(camera[0].text.split(), dtype='float32').reshape(4, 4)
            T = transformation @ T
            R = torch.tensor(T[:3, :3]/scale)
            t = torch.tensor(T[:3, 3])
            q = conversions.rotation_matrix_to_quaternion(R)
            log_q = conversions.quaternion_exp_to_log(q)
            pose = torch.cat((t, log_q))
            self.poses.append(pose)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Input index.

        Returns:
            {'image' (torch.Tensor): Image,
             'pose' (torch.Tensor): Pose of the image.}

        Shape:
            {'image': (3, height, width),
             'pose': (6).}
        """
        return {'image': self.imgs[idx],
                'pose': self.poses[idx]}
