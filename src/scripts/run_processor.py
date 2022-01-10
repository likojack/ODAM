import argparse
import numpy as np
import os
import pickle
from PIL import Image
import torch
from tqdm import tqdm

from src.datasets.scan_net_track import ScanNetTrack
from src.processor import OdamProcess
import src.datasets.scannet_utils as scannet_utils
from src.models.detr import build as build_detector
from src.models.associator import build as build_associator
from src.config.configs import ConfigLoader

from src.datasets.transforms import get_transforms


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config_path")
    arg_parser.add_argument("--detect_threshold", default=0.6, type=float)
    arg_parser.add_argument("--use_prior", action="store_true")
    arg_parser.add_argument("--no_code", action="store_true")
    arg_parser.add_argument("--representation", default="super_quadric", help="[cube, super_quadric, quadric]")   
    arg_parser.add_argument("--out_dir", default="./result/test") 
    args = arg_parser.parse_args()

    cfg = ConfigLoader().merge_cfg([args.config_path])
    detector, _, _ = build_detector(cfg)
    detector.to("cuda")
    checkpoint = torch.load("./experiments/detector.pth")
    detector.load_state_dict(checkpoint["model"])
    detector.eval()

    associator = build_associator(cfg)
    weight = torch.load("./experiments/associator.pth")
    associator.load_state_dict(weight['model'])
    associator.cuda().eval()

    transforms = get_transforms()

    dataset_root = "./data/ScanNet/scans"
    dataset = ScanNetTrack()
    
    for seq_id in dataset.files:
        print("processing: {seq_id}")
        out_dir = os.path.join(args.out_dir, seq_id)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_path = os.path.join(out_dir, seq_id)

        n_frames = len(dataset.files[seq_id]['img_names'])
        intr_mat = scannet_utils.read_intrinsic(
            dataset.intr_path.format(seq_id))[:3, :3]
        meta_path = os.path.join(
            "./data/ScanNet/scans/", "{0}/{0}.txt".format(seq_id))
        axis_align_mat = scannet_utils.read_meta_file(
            dataset.meta_path.format(seq_id))

        processor = OdamProcess(
            detector,
            associator,
            transforms,
            args.use_prior,
            representation=args.representation
        )
        processor.init_sequence(intr_mat, dataset.img_h, dataset.img_w)
        
        for frame_id in tqdm(range(0, n_frames)):
            img_name = dataset.files[seq_id]['img_names'][frame_id]
            T_cw = scannet_utils.read_extrinsic(
                dataset.pose_path.format(seq_id, img_name))
            if np.isnan(T_cw).any():
                continue
            T_wc = np.linalg.inv(T_cw)
            T_wc = axis_align_mat @ T_wc
            rgb = Image.open(dataset.img_path.format(seq_id, img_name))
            processor.process_frame(rgb, int(img_name), T_wc)

        out = processor.optim_process(processor.tracks)
        merge_tracks = processor.merge_process(out)
        out = processor.optim_process(merge_tracks)

        with open(os.path.join(out_dir, seq_id), "wb") as f:
            out_dict = {
                "tracks": out["tracks"],
                "bboxes_qc": out["bboxes_qc"],
                "bboxes_dl": out["bboxes_dl"],
                "quadrics": out["quadrics"],
            }
            pickle.dump(out_dict, f)