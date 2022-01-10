"""some functions to load tracking GT"""

import os
import numpy as np
import pickle
import sys
from scipy.spatial.transform import Rotation

import src.datasets.scannet_utils as scannet_utils
import src.utils.box_utils as box_utils
import src.utils.geometry_utils as geo_utils
import src.utils.visual_utils as visual_utils
import src.super_quadric.quadric_helper as helper


def get_depth_planes(pts_w, T_wc):
    """ get planes for min/max depth, then transform to world coordinate
    """

    pts_c = (geo_utils.get_homogeneous(pts_w) @ np.linalg.inv(T_wc).T)[:, :3]
    min_depth_plane_c = np.array([0, 0, -1., np.min(pts_c[:, 2])])
    min_depth_plane_w = np.linalg.inv(T_wc).T @ min_depth_plane_c
    min_depth_plane_w = helper.normalize_plane(min_depth_plane_w[None, :])
    min_depth_plane_w = helper.plane_2vect(min_depth_plane_w[0, :])

    max_depth_plane_c = np.array([0, 0, -1., np.max(pts_c[:, 2])])
    max_depth_plane_w = np.linalg.inv(T_wc).T @ max_depth_plane_c
    max_depth_plane_w = helper.normalize_plane(max_depth_plane_w[None, :])
    max_depth_plane_w = helper.plane_2vect(max_depth_plane_w[0, :])

    return [min_depth_plane_w, max_depth_plane_w]


def compute_iou(gt_annotations, bbox_pred, obj_class, used_gt_ids, threshold):
    max_iou = -1
    max_iou_2d = -1
    matched_bbox = None
    matched_gt_id = -1
    if obj_class in [2, 4, 10]:
        bbox_pred[4: 7, 2] = 0
    for gt_id, gt_anno in enumerate(gt_annotations):
        gt_class, gt_bbox = gt_anno
        if gt_class in [2, 4, 10]:
            gt_bbox[4: 7, 2] = 0
        gt_is_table = True if gt_class == 4 or gt_class == 10 else False
        pred_is_table = True if obj_class == 4 or obj_class == 10 else False
        if gt_class == obj_class: #or all([gt_is_table, pred_is_table]):
            iou, iou_2d = box_utils.box3d_iou(gt_bbox, bbox_pred)
            if max_iou < iou and gt_id not in used_gt_ids:
                max_iou = iou
                max_iou_2d = iou_2d
                matched_bbox = gt_bbox
                matched_gt_id = gt_id
    if matched_gt_id != -1 and max_iou > threshold:
        used_gt_ids.append(matched_gt_id)
    return max_iou, max_iou_2d, matched_bbox


def averaging_T_wos(T_wos):
    out_T_wo = np.eye(4)
    T_wos = [T_wo for T_wo in T_wos if len(T_wo) > 0]
    T_wos = np.asarray(T_wos)
    rot_mat = Rotation.from_matrix(T_wos[:, :3, :3]).mean().as_matrix()
    out_T_wo[:3, :3] = rot_mat
    out_T_wo[:3, 3] = np.mean(T_wos[:, :3, 3], axis=0)
    return out_T_wo


def preprocess_gt(tracks):
    valid_objs = (tracks[:, :, 0] != -1).any(axis=1)
    tracks = tracks[valid_objs, :, :]
    return tracks


def load_poses(pose_dir, img_names, axis_align_mat, K):
    """ load T_wc and P_cws from dataset loader

    P_cw is the projection and transformation matrix from world to cam
    """

    T_wcs = []
    P_cws = []
    n_imgs = len(img_names)
    for img_id in range(n_imgs):
        # T_wc = dataset_loader.get_frame_pose(img_names[img_id])
        T_cw = scannet_utils.read_extrinsic(os.path.join(
            pose_dir, "{}.txt".format(img_names[img_id])))
        T_wc = np.linalg.inv(T_cw)
        # T_wc = geo_utils.pad_transform_matrix(T_wc)
        T_wc = axis_align_mat @ T_wc
        T_cw = np.linalg.inv(T_wc)
        T_wcs.append(T_wc)
        P_cw = K @ T_cw[:3, :]
        P_cws.append(P_cw)
    return T_wcs, P_cws


def load_gt_object(obj_track, n_imgs, T_wcs, img_h, img_w, K):
    lines = []
    bboxes_lines = []
    plane_vecs = []
    T_wos = []
    scales = []
    depth_planes = []

    valid_frames = obj_track[:, 0] != -1
    obj_class = obj_track[valid_frames, 1][0]
    t_wo = np.mean(obj_track[valid_frames, 9: 12], axis=0)
    for img_id in range(n_imgs):
        if not valid_frames[img_id]:
            lines.append([])
            bboxes_lines.append([])
            plane_vecs.append([])
            T_wos.append([])
            scales.append([])
            depth_planes.append([])
            continue
        T_wc = T_wcs[img_id]
        R_wo = box_utils.rotz(obj_track[img_id, 12])
        T_wo = np.eye(4)
        T_wo[:3, :3] = R_wo
        T_wo[:3, 3] = t_wo
        T_wos.append(T_wo)
        
        line = np.stack([T_wc[:3, 3], obj_track[img_id, 9: 12]], axis=0)
        lines.append(line)
        
        bbox = obj_track[img_id, 2: 6].reshape(2, 2)
        bbox_line = helper.bbox_to_lines(bbox, img_size=(img_h, img_w), edge_threshold=20)
        bboxes_lines.append(bbox_line)

        P = K @ np.linalg.inv(T_wc)[:3, :]
        bbox_planes = [helper.normalize_plane(line[None, :] @ P) for line in bbox_line.values()]
        plane_vec = [helper.plane_2vect(plane[0, :]) for plane in bbox_planes]
        plane_vecs.append(plane_vec)
        scales.append(obj_track[img_id][6: 9])
        
        bbox_w = box_utils.get_3d_box(scales[-1], T_wo[:3, :3], T_wo[:3, 3])
        minmax_depth_planes = get_depth_planes(bbox_w, T_wc)
        depth_planes.append(minmax_depth_planes)
  
    return lines, bboxes_lines, plane_vecs, obj_class, T_wos, scales, depth_planes


def load_pred_object(obj_track, frame_ids, T_wcs, img_h, img_w, K):
    lines = []
    bboxes_lines = []
    plane_vecs = []
    T_wos = []
    scales = []
    depth_planes = []
    n_imgs = len(frame_ids)
    obj_class = int(np.median(obj_track[:, 1]))
    obj_frames = obj_track[:, 0].astype(np.int32)
    t_wo = np.mean(obj_track[:, 9: 12], axis=0)
    last_key_frame = None
    total_frames = 0
    used_frames = 0
    for img_id in range(n_imgs):
        if frame_ids[img_id] not in obj_frames:
            lines.append([])
            bboxes_lines.append([])
            plane_vecs.append([])
            T_wos.append([])
            scales.append([])
            depth_planes.append([])
            continue
        total_frames += 1
        # if last_key_frame is None:
        #     last_key_frame = {}
        #     last_key_frame['pose'] = T_wcs[img_id]
        #     last_key_frame['time'] = img_id
        # else:
        #     if (np.linalg.norm(last_key_frame['pose'][:3, 3] - T_wcs[img_id][:3, 3]) < 0.5) and \
        #         (last_key_frame['time'] - img_id < 20) and \
        #         ():
        #         lines.append([])
        #         bboxes_lines.append([])
        #         plane_vecs.append([])
        #         T_wos.append([])
        #         scales.append([])
        #         depth_planes.append([])
        #         continue
        used_frames += 1
        current_frame_in_track = np.where(frame_ids[img_id]==obj_frames)[0][0]
        assert obj_track[current_frame_in_track, 0] == frame_ids[img_id]

        T_wc = T_wcs[img_id]
        R_wo = box_utils.rotz(obj_track[current_frame_in_track, 12])
        T_wo = np.eye(4)
        T_wo[:3, :3] = R_wo
        T_wo[:3, 3] = t_wo
        T_wos.append(T_wo)
        
        line = np.stack([T_wc[:3, 3], obj_track[current_frame_in_track, 9: 12]], axis=0)
        lines.append(line)
        
        bbox = obj_track[current_frame_in_track, 2: 6].reshape(2, 2)
        bbox_line = helper.bbox_to_lines(bbox, img_size=(img_h, img_w), edge_threshold=20)
        bboxes_lines.append(bbox_line)

        P = K @ np.linalg.inv(T_wc)[:3, :]
        bbox_planes = [helper.normalize_plane(line[None, :] @ P) for line in bbox_line.values()]
        plane_vec = [helper.plane_2vect(plane[0, :]) for plane in bbox_planes]
        plane_vecs.append(plane_vec)
        scales.append(obj_track[current_frame_in_track][6: 9])
        
        bbox_w = box_utils.get_3d_box(scales[-1], T_wo[:3, :3], T_wo[:3, 3])
        minmax_depth_planes = get_depth_planes(bbox_w, T_wc)
        depth_planes.append(minmax_depth_planes)
    return lines, bboxes_lines, plane_vecs, obj_class, T_wos, scales, depth_planes

