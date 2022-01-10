import numpy as np
import os
from scipy.spatial.transform import Rotation

import src.utils.tracking_gt_utils as tracking_gt_utils
import src.utils.box_utils as box_utils
import src.super_quadric.sq_libs as quadric_libs
from src.utils.file_utils import get_date_time


def logging(out_dir, info):
    with open(os.path.join(out_dir, "log.txt"), "w") as f:
        for line in info:
            f.write(line + "\n")


def create_dir(out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)


def optim_process(tracks, img_names, T_wcs, P_cws, img_h, img_w, K, representation, prior, n_iters, n_views):
    lines_all_objects = []
    bboxes_lines_all_objects = []
    plane_vecs_all_objects = []
    objs_class = []
    T_wos_all_objects = []
    scales_all_objects = []
    depth_all_objects = []
    n_objs = len(tracks)
    for obj_id in range(n_objs):
        lines, bboxes_lines, plane_vecs, obj_class, T_wos, scales, depth_planes = \
            tracking_gt_utils.load_pred_object(tracks[obj_id], img_names, T_wcs, img_h, img_w, K)
        lines_all_objects.append(lines)
        bboxes_lines_all_objects.append(bboxes_lines)
        plane_vecs_all_objects.append(plane_vecs)
        objs_class.append(obj_class)
        T_wos_all_objects.append(T_wos)
        scales_all_objects.append(scales)
        depth_all_objects.append(depth_planes)
    quadrics = []
    bboxes_qc = []
    bboxes_dl = []
    for obj_id in range(n_objs):
        obj_class = objs_class[obj_id]
        T_wo = tracking_gt_utils.averaging_T_wos(T_wos_all_objects[obj_id])
        scales = [s for s in scales_all_objects[obj_id] if len(s) > 0]
        scales = np.mean(np.asarray(scales), axis=0)
        bbox_pred = box_utils.get_3d_box(scales, T_wo[:3, :3], T_wo[:3, 3])
        bboxes_dl.append(bbox_pred)
        bbox_lines = bboxes_lines_all_objects[obj_id]
        valid_frames = [img_id 
                        for img_id, _ in enumerate(img_names) 
                        if len(bbox_lines[img_id]) > 0]
        bbox_lines = [b for b in bbox_lines if len(b) > 0]
        optimizer = quadric_libs.SuperQuadricOptimizer(
            T_wo[:3, 3], Rotation.from_matrix(T_wo[:3, :3]).as_euler("zxy")[0],
            scales, obj_class, representation, prior)
        if len(valid_frames) < n_views:
            bbox_qc = bbox_pred
            quadrics.append(optimizer.Q_init)
            bboxes_qc.append(bbox_qc)
        else:
            Q_est = optimizer.run(
                bbox_lines, None, np.asarray(P_cws)[valid_frames], n_iters)
            pred_points, _ = Q_est.compute_ellipsoid_points(use_numpy=True)
            bbox_qc = box_utils.compute_oriented_bbox(pred_points)
            quadrics.append(Q_est)
            bboxes_qc.append(bbox_qc)
    out_dict = {
        "tracks": tracks,
        "bboxes_qc": bboxes_qc,
        "bboxes_dl": bboxes_dl,
        "quadrics": quadrics
    }
    return out_dict
