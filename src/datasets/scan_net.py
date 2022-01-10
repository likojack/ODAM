"""
COCO dataset which returns image_id for evaluation.
Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
import os
import json
import math
import numpy as np
from pathlib import Path
from PIL import Image
import pickle
import sys
import torch
import torch.utils.data
import torchvision

sys.path.append("/home/kejie/repository/imovotenet/src/datasets")
import src.datasets.transforms as T
from src.utils.geometry_utils import get_homogeneous
from src.utils.file_utils import get_file_name
from src.utils.geometry_utils import angle2class


class ScanNet(torch.utils.data.Dataset):
    def __init__(self, split, transforms):
        super(ScanNet, self).__init__()
        
        self.transforms = transforms
        
        self.base_dir = "./data/ScanNet/imovotenet_scan2cad"
        with open(os.path.join(self.base_dir, "{}.json".format(split)), "r") as f:
            self.data = json.load(f)
        self.data = [d for d in self.data if len(d['objects'])>0]

    def __getitem__(self, idx):
        """load and prepare data
        """

        datum = self.data[idx]
        img, target, img_path = self.process_item(datum)
        return img[0], target[0], img_path[0]

    def process_item(self, datum):
        objects = torch.tensor(datum['objects']).float()
        img = Image.open(datum['img_path'])
        objects[:, -1] = angle2class(objects[:, -1])

        # # 0th is class and instance id:
        # # inst_id * 100 + class_id
        # objects[:, 0] = objects[:, 0] % 100
        target = {
            "objects": objects,
        }
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return [img], [target], [datum['img_path']]

    def __len__(self):
        return len(self.data)


def build(image_set, args):
    if "train" not in image_set:
        normalize = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        transforms = normalize
    else:
        normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
        transforms = T.Compose([
            T.RandomResize(scales, max_size=1333),
            normalize])

    dataset = ScanNet(image_set, transforms)
    return dataset


if __name__ == "__main__":
    """ testing for ScanNet initialization and data loading
    """
    
    scan_net_dataset = ScanNet()
    data_loader = torch.utils.data.DataLoader(
        dataset=scan_net_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True)
    for idx, inputs in enumerate(data_loader):
        print(inputs)