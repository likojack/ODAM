from copy import deepcopy
import numpy as np
import os
import torch
from PIL import Image
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment


from src.datasets.scan_net_track import ScanNetTrack
from src.models.associator import build as build_model
from src.main_track import get_args_parser 
import src.utils.geometry_utils as geo_utils
from src.config.configs import ConfigLoader
import src.datasets.scannet_utils as scannet_utils
import src.utils.box_utils as box_utils
from src.super_quadric.sq_libs import SuperQuadric
from src.utils.file_utils import get_date_time


def prepare_tracks(dataset, tracks, img_size, T_wc, intr_mat, img, n_times=100):
    """ return track in a format that can be used for matcher
    
    use shape_decoder and the object pose to get projected bounding box
    """

    cam_azi = scannet_utils.get_cam_azi(T_wc)
    for idx, track in enumerate(tracks):
        # use projection of 3D bounding box for projected bbox.
        azi_wo = np.mean(track[:, 12], axis=0)
        R_wo = box_utils.rotz(azi_wo)
        t_wo = np.mean(track[:, 9: 12], axis=0)
        dimensions = np.mean(track[:, 6: 9], axis=0)
        dimensions = np.clip(dimensions, a_min=0.05, a_max=np.inf)
        Q = SuperQuadric(t_wo, azi_wo, np.sqrt(dimensions/2), shapes=np.array([-0., -0.]))
        pts, _ = Q.compute_ellipsoid_points(use_numpy=True)
        # box_3d_w = box_utils.get_3d_box(dimensions, R_wo, t_wo)
        box_3d_c = (geo_utils.get_homogeneous(pts) @ np.linalg.inv(T_wc).T)[:, :3]
        pixels = geo_utils.projection(box_3d_c, intr_mat)
        x_min, y_min, _ = np.min(pixels, axis=0)
        x_max, y_max, _ = np.max(pixels, axis=0)
        track[:, -4:] = np.array([[x_min, y_min, x_max, y_max]])
        tracks[idx] = track
    tracks = dataset._preprocess_tracks(deepcopy(tracks), img_size, T_wc, cam_azi, n_times)
    return tracks


def init_tracks(dataset, no_code, no_depth):
    """ initial state of a scene.

    Use detections at the first frame to initialize tracking
    """

    out_tracks = []
    tracks = dataset[:, 0, :]
    valid_id = tracks[:, 0] != -1
    for track in tracks[valid_id]:
        track = np.delete(track, 14, axis=0)  # delete ground-truth instance ID
        out_track = track[None, :]
        if no_depth:
            out_track[:, 9: 12] = -1
        if no_code:
            out_track[:, 14:78] = -1
        out_tracks.append(out_track)
    return out_tracks


def collater(data_list):
    """ batching
    """
    # assert len(data_list) == 1, "use batch_size 1 now"
    # return data_list[0]
    max_dets = 30
    matches = []
    tracks = []
    detections = []
    img_names = []
    poses = []
    track_batch_split = []
    detection_batch_split = []
    valid_list = []

    n_features = data_list[0]['detections'].shape[0]
    detections = torch.ones((len(data_list), n_features, max_dets), dtype=torch.float) * -1

    for b_id, data in enumerate(data_list):
        tracks.append(data['tracks'])
        matches.append(data['match'])
        img_names.append(data['img_path'])
        poses.append(data['pose'])
        _n_tracks = data['tracks'].shape[0]
        _n_detections = data['detections'].shape[1]
        track_batch_split.append(_n_tracks)
        detection_batch_split.append(_n_detections)
        valid_list.append((_n_tracks, _n_detections))
        detections[b_id, :, :_n_detections] = data['detections']

    total_detections = torch.sum(torch.tensor(detection_batch_split))
    total_tracks = torch.sum(torch.tensor(track_batch_split))

    gt_scores = torch.zeros((total_tracks, total_detections), dtype=torch.float)
    gt_masks = torch.zeros((total_tracks, total_detections), dtype=torch.float)
    track_0, detect_0 = 0, 0

    # create cross attention mask
    for b_idx, track_range, detect_range in zip(np.arange(len(data_list)), track_batch_split, detection_batch_split):
        track_1 = track_0 + track_range
        detect_1 = detect_0 + detect_range
        gt_masks[track_0: track_1, detect_0: detect_1] = 1
        track_0 = track_1
        detect_0 = detect_1

    tracks = torch.cat(tracks, dim=0)
    return {
        "tracks": tracks,
        "detections": detections,
        "gt_matches": matches,
        "gt_masks": gt_masks,
        "img_names": img_names,
        "track_batch_split": track_batch_split,
        "detection_batch_split": detection_batch_split,
        "poses": poses,
        "valid_list": valid_list}


def attach_to_tracks(pred_match, detections, tracks, T_wc, img_size, score_mat, match_threshold, no_code, no_depth):
    """ use the matching prediction to attach detections to tracks

    transform rotation and translate in detections to world coord. using T_wc 
    """

    img_w, img_h = img_size
    cam_azi = scannet_utils.get_cam_azi(T_wc)
    t_co = detections[:, 9: 12]
    t_wo = (geo_utils.get_homogeneous(t_co) @ T_wc.T)[:, :3]
    detections[:, 9: 12] = t_wo
    sin = detections[:, 12]
    cos = detections[:, 13]
    azi_co = np.arctan2(sin, cos)
    azi_wo = azi_co + cam_azi
    det_ids = np.arange(len(detections))

    for match_track_id, det_id, detection in zip(pred_match, det_ids, detections):
        match_score = score_mat[match_track_id, det_id]
        if match_score < match_threshold:
            continue
        attach_ = np.zeros((1, 82))
        attach_[0, :9] = detection[:9]
        attach_[0, 2: 6] *= np.array([img_w, img_h, img_w, img_h])
        attach_[0, 9: 12] = t_wo[det_id]
        attach_[0, 12] = azi_wo[det_id]
        attach_[0, 13] = detection[14]
        if no_depth:
            attach_[0, 9: 12] = -1
        if no_code:
            attach_[0, 14: 78] = -1
        else:
            attach_[0, 14: 78] = detection[15: 79]
        attach_[0, 78: 82] = detection[2: 6] * np.array([img_w, img_h, img_w, img_h])
        if match_track_id == -1:
            tracks.append(attach_)
        else:
            tracks[match_track_id] = np.concatenate(
                [tracks[match_track_id], attach_], axis=0
            )
