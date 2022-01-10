import torch.utils.data
import torchvision

from .scan_net import build as build_scan_net
from .scan_net_track import build as build_scan_net_track


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


def build_dataset(image_set, args):
    if args.dataset_file == 'scan_net':
        return build_scan_net(image_set, args)
    if args.dataset_file == 'scan_net_track':
        return build_scan_net_track(image_set, args)
    if args.dataset_file == "sun_rgbd":
        return build_sun_rgbd(image_set, args)

    raise ValueError(f'dataset {args.dataset_file} not supported')