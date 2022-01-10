import argparse
import cv2
import numpy as np
import os
import pickle
from scipy.optimize import linear_sum_assignment
from PIL import Image
import torch
from tqdm import tqdm

from src.models.detr import build as build_model
import src.datasets.scannet_utils as scannet_utils
from src.config.configs import ConfigLoader
import src.datasets.transforms as T
import src.utils.geometry_utils as geo_utils
import src.utils.box_utils as box_utils
from src.utils.file_utils import get_date_time


"""
TRACK format:
0: img_name
1: obj_class
2 - 6: bounding box
6 - 9: scale
9 - 12: translate T_wo
12: rot R_wo
13: score
14: track_id

"""


SCANNET_DIR = "./data/ScanNet/scans/"


def convert_det_to_list(detections, det_id, img_name, img_h, img_w, T_wc):
    """ convert a detection from detr output to a TRACK format. """

    obj = [img_name, detections["classes"][det_id]]
    bbox_2d = detections["bboxes"][det_id]
    bbox_2d[:, 0] = np.clip(bbox_2d[:, 0], a_min=0, a_max=img_w)
    bbox_2d[:, 1] = np.clip(bbox_2d[:, 1], a_min=0, a_max=img_h)
    obj += bbox_2d.flatten().tolist()
    obj += detections["dimensions"][det_id].tolist()
    t_co = detections["translates"][det_id]
    t_wo = (geo_utils.get_homogeneous(t_co[None, :]) @ T_wc.T)[0, :3]
    obj += t_wo.tolist()
    obj += [detections["angles"][det_id] / 180. * np.pi]
    obj += [detections["scores"][det_id]]
    obj += [-1]  # haven't assigned to any track yet
    return obj


def init_tracks(
    img, depth_map, tracks, tracks_points, detections, img_name, used_detections,
    T_wc, img_h, img_w, depth_intr_mat, track_threshold
):
    """initialize a new object trajectory using unmatched detections"""

    orb = cv2.ORB_create()
    kps = orb.detect(img, None)
    kps = np.stack([np.asarray(t.pt) for t in orb.detect(img, None)], axis=0)

    n_detections = len(detections["scores"])
    current_track_id = len(tracks)
    for det_id in range(n_detections):
        # this is already matchde to an object track
        if det_id in used_detections:
            continue
        if detections["scores"][det_id] < track_threshold:
            continue
        obj = convert_det_to_list(
            detections, det_id, img_name, img_h, img_w, T_wc
        )
        bbox = np.asarray(obj)[2: 6].reshape(2, 2)
        pts_in_box_indicator = geo_utils.pts_in_box(kps, bbox)
        new_kps = kps[pts_in_box_indicator]
        new_kps[:, 0] = new_kps[:, 0] / img.shape[1] * depth_map.shape[1]
        new_kps[:, 1] = new_kps[:, 1] / img.shape[0] * depth_map.shape[0]
        kps_indices = new_kps.astype(np.int32)
        depths = depth_map[kps_indices[:, 1], kps_indices[:, 0]]
        valid_depth_ids = depths > 0.1
        new_kps = new_kps[valid_depth_ids]
        depths = depths[valid_depth_ids]
        new_pts = geo_utils.unproject(new_kps, depths, depth_intr_mat)
        if len(new_pts) == 0:
            continue
        # o3d_pts = o3d_helper.np2pc(new_pts)
        # _img = cv2.resize(img, (depth_map.shape[1], depth_map.shape[0]))
        # scene_pts, scene_pts_colors = geo_utils.rgbd_to_colored_pc(
        #     _img, depth_map,
        #     depth_intr_mat[0,0], depth_intr_mat[1,1], depth_intr_mat[0,2],
        #     depth_intr_mat[1,2])
        # scene_pts = o3d_helper.np2pc(scene_pts, scene_pts_colors)
        # o3d.visualization.draw_geometries([o3d_pts, scene_pts])
        new_pts = (geo_utils.get_homogeneous(new_pts) @ T_wc.T)[:, :3]
        tracks_points[current_track_id] = new_pts

        obj[-1] = current_track_id
        tracks.append([obj])

        current_track_id += 1


def match_tracks(
    tracks, detections, img_name, used_detections, 
    deactivate_track_ids, img_h, img_w, T_wc, threshold
):
    """ match detections to tracks if the bbox overlap is larger than threshold
    """

    orders = np.argsort(detections["scores"])[::-1]
    used_tracks = []
    for det_id in orders:
        if det_id in used_detections:
            continue
        obj = convert_det_to_list(detections, det_id, img_name, img_h, img_w, T_wc)
        target_bbox = np.asarray(obj)[2: 6].reshape(2, 2)
        dimensions = obj[6: 9]
        target_bbox_3d = np.array([
            [-dimensions[0], -dimensions[1], -dimensions[2]],
            [dimensions[0], dimensions[1], dimensions[2]],
        ])
        target_bbox_3d = target_bbox_3d / 2.
        t_wo = np.asarray(obj[9: 12])
        target_bbox_3d += t_wo[None, :]
        target_class = obj[1]

        best_id = -1
        max_iou_2d = -1
        max_iou_3d = -1
        for track_id, track in enumerate(tracks):
            last_frame = track[-1]
            source_track_id = last_frame[-1]
            if source_track_id in used_tracks:
                continue
            source_class = last_frame[1]
            disable_2d = False
            # ignore if this track is not observed in the last 5 frames
            if (img_name - last_frame[0]) > 5:
                disable_2d = True
            source_bbox = np.asarray(last_frame)[2: 6].reshape(2, 2)
            dimensions = np.mean(np.asarray(track)[:, 6: 9], axis=0)
            source_bbox_3d = np.array([
                [-dimensions[0], -dimensions[1], -dimensions[2]],
                [dimensions[0], dimensions[1], dimensions[2]],
            ])
            t_wo = np.mean(np.asarray(track)[:, 9: 12], axis=0)
            source_bbox_3d = source_bbox_3d / 2.
            source_bbox_3d += t_wo[None, :]
            iou_3d = box_utils.iou_3d(target_bbox_3d, source_bbox_3d)
            if not disable_2d:
                iou_2d = box_utils.iou_2d(source_bbox, target_bbox)
                if iou_2d > max_iou_2d and iou_3d > max_iou_3d and source_class == target_class:
                    max_iou_2d = iou_2d
                    max_iou_3d = iou_3d
                    best_id = source_track_id
            else:
                if iou_3d > max_iou_3d and target_class == source_class:
                    best_id = source_track_id
                    max_iou_3d = iou_3d

        if max_iou_2d > threshold or max_iou_3d > 0.2:
            assert best_id != -1
            obj[-1] = best_id
            tracks[best_id].append(obj)
            assert len(np.unique(np.asarray(tracks[best_id])[:, 1])) == 1
            used_detections.append(det_id)
            used_tracks.append(best_id)


def match_tracks_feature(
    img, depth_map, tracks, tracks_points, detections, img_name, used_detections, 
    deactivate_track_ids, img_h, img_w, T_wc, intr_mat, depth_intr_mat, threshold
):
    """ Hungarian with feature point matching"""


    orders = np.argsort(detections["scores"])[::-1]
    used_tracks = []
    n_detections = len(detections['bboxes'])
    cost_mat = np.zeros((n_detections, len(tracks))) + 100.
    for det_id in orders:
        if det_id in used_detections:
            continue

        obj = convert_det_to_list(detections, det_id, img_name, img_h, img_w, T_wc)
        target_bbox = np.asarray(obj)[2: 6].reshape(2, 2)
        target_class = obj[1]

        best_id = -1
        for track_id, track in enumerate(tracks):
            last_frame = track[-1]
            source_track_id = last_frame[-1]
            source_class = last_frame[1]
            if target_class != source_class:
                continue
            track_points = tracks_points[track_id]
            track_points = (geo_utils.get_homogeneous(track_points) @ np.linalg.inv(T_wc).T)[:, :3]
            track_points = geo_utils.projection(track_points, intr_mat)[:, :2]
            pt_ids = geo_utils.pts_in_box(track_points, np.array([0, 0, img.shape[1], img.shape[0]]))
            track_points = track_points[pt_ids]
            if len(track_points) == 0:
                continue
            pts_in_box = geo_utils.pts_in_box(track_points, target_bbox)
            cost = 1 - np.sum(pts_in_box) / len(pts_in_box)
            if cost > 0.2:
                continue
            cost_mat[det_id, track_id] = cost

    row_indices, col_indices = linear_sum_assignment(cost_mat)

    # img = cv2.resize(img, (depth_map.shape[1], depth_map.shape[0]))
    orb = cv2.ORB_create()
    # kp: [x, y]
    kps = np.stack([np.asarray(t.pt) for t in orb.detect(img, None)], axis=0)

    for row_ind, col_ind in zip(row_indices, col_indices):
        if cost_mat[row_ind, col_ind] > 1:
            continue
        obj = convert_det_to_list(detections, row_ind, img_name, img_h, img_w, T_wc)
        obj[-1] = col_ind
        tracks[col_ind].append(obj)
        assert len(np.unique(np.asarray(tracks[col_ind])[:, 1])) == 1
        assert not any(np.asarray(tracks[col_ind])[:, -1] == -1)
        bbox = np.asarray(obj)[2: 6].reshape(2, 2)
        pts_in_box_indicator = geo_utils.pts_in_box(kps, bbox)
        new_kps = kps[pts_in_box_indicator]
        new_kps[:, 0] = new_kps[:, 0] / img.shape[1] * depth_map.shape[1]
        new_kps[:, 1] = new_kps[:, 1] / img.shape[0] * depth_map.shape[0]
        kps_indices = new_kps.astype(np.int32)
        depths = depth_map[kps_indices[:, 1], kps_indices[:, 0]]
        valid_depth_ids = depths > 0.1
        new_kps = new_kps[valid_depth_ids]
        depths = depths[valid_depth_ids]
        new_pts = geo_utils.unproject(new_kps, depths, depth_intr_mat)
        new_pts = (geo_utils.get_homogeneous(new_pts) @ T_wc.T)[:, :3]
        all_track_points = np.concatenate(
            (tracks_points[track_id], new_pts), axis=0)
        tracks_points[col_ind] = np.random.permutation(all_track_points)[:1000]
        used_detections.append(row_ind)
        

def deactive_tracks(tracks, img_name, deactive_track_ids):
    for track_id, track in enumerate(tracks):
        if img_name - track[-1][0] > 5:
            deactive_track_ids.append(track_id)
 

def process_seq(seq, model, transform, out_dir):
    intr_dir = os.path.join(
        "./data/ScanNet/",
        "scans/{}/frames/intrinsic/intrinsic_color.txt".format(seq)
    )
    depth_intr_dir = os.path.join(
        "./data/ScanNet/",
        "scans/{}/frames/intrinsic/intrinsic_depth.txt".format(seq)
    )
    
    extr_dir = "./data/ScanNet/scans/{}/frames/pose/{}.txt"
    intr_mat = scannet_utils.read_intrinsic(intr_dir)[:3, :3]
    depth_intr_mat = scannet_utils.read_intrinsic(depth_intr_dir)[:3, :3]

    meta_file = os.path.join(SCANNET_DIR, seq, seq + '.txt') # includes axisAlignment info for the train set scans. 
    axis_align_matrix = scannet_utils.read_meta_file(meta_file)

    imgs = [f.split(".")[0] for f in os.listdir("./data/ScanNet/scans/{}/frames/color/".format(seq)) if f.endswith(".jpg")]
    imgs = sorted(imgs, key=lambda a: int(a))
    depth_dir = "./data/ScanNet/scans/{}/frames/depth/".format(seq)
    tracks = []

    out_dir = os.path.join(out_dir, seq)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    tracks_points = {}

    for img_idx, img_name in enumerate(tqdm(imgs)):
        img_name = int(img_name)
        depth_map = cv2.imread(os.path.join(depth_dir, "{}.png".format(img_name)), -1) / 1000.
        T_cw = scannet_utils.read_extrinsic(extr_dir.format(seq, img_name))
        T_wc = np.linalg.inv(T_cw)
        T_wc = axis_align_matrix @ T_wc
        if np.isnan(T_cw).any():
            continue
        img_path = f"./data/ScanNet/scans/{seq}/frames/color/{img_name}.jpg"
        img = Image.open(img_path)
        img_w, img_h = img.size
        img_tensor, _ = transform(img, None)
        img_tensor = img_tensor.unsqueeze(0)
        img_tensor = img_tensor.to("cuda")
        
        # run detector
        detr_outputs = model(img_tensor)
        out_objects = model.postprocess(
            detr_outputs, [img_w, img_h], args.det_threshold, intr_mat
        )
        if not out_objects:
            continue
        out_objects = {k: v[0] for k, v in out_objects.items()}
        used_detections = []
        deactivate_track_ids = []
        # for i in range(len(out_objects["scores"])):
        #     drawing.draw_2d_box(img, out_objects["bboxes"][i], color="red")
        # # init unmatched detections if score is above threshold
        # if img_name == 24:
        #     import pdb
        #     pdb.set_trace()
        # # match deteciont to existing tracks
        match_tracks_feature(
            np.asarray(img), depth_map, tracks, tracks_points, out_objects, img_name, used_detections, 
            deactivate_track_ids, img_h, img_w, T_wc, intr_mat, depth_intr_mat, args.match_threshold
        )
        init_tracks(
            np.asarray(img), depth_map, tracks, tracks_points, out_objects, img_name, used_detections, 
            T_wc, img_h, img_w, depth_intr_mat, args.track_threshold
        )

        # terminate unmatched tracks
        deactive_tracks(tracks, img_name, deactivate_track_ids)

    out_tracks = []
    for track in tracks:
        out_tracks.append(np.asarray(track))
    with open(os.path.join(out_dir, seq), "wb") as f:
        out_dict = {"tracks": out_tracks}
        pickle.dump(out_dict, f)


def main(args):
    with open("./data/ScanNet/scannetv2_{}.txt".format(args.split), "r") as f:
        seqs = f.read().splitlines()
    
    detr_cfg = ConfigLoader().merge_cfg([args.config_path])
    model, _, _ = build_model(detr_cfg)
    model.to("cuda")
    checkpoint = torch.load(args.pretrained_path)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    normalize = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transforms = normalize

    date_time = get_date_time()
    out_dir = "./result/tracking/scan2cad/{}".format(date_time)
    for seq in tqdm(seqs):
        process_seq(seq, model, transforms, out_dir)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--split")
    arg_parser.add_argument("--det_threshold", type=float, default=0.7)
    arg_parser.add_argument("--match_threshold", type=float, default=0.5)
    arg_parser.add_argument("--track_threshold", type=float, default=0.8)
    arg_parser.add_argument("--config_path")
    arg_parser.add_argument("--pretrained_path")
    args = arg_parser.parse_args()
    main(args)