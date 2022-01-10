import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib._color_data as mcd
import numpy as np
import os
import open3d as o3d
import PIL
from PIL import Image, ImageDraw
import sys
from scipy.spatial.transform import Rotation as R
import torch

import src.utils.geometry_utils as geo_utils
import src.utils.box_utils as box_utils
from src.models.associator import hungarian_matching
# from src.utils.misc import hungarian_matching


SEMANTIC2NAME = [
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "bookshelf",
    "picture",
    "counter",
    "desk",
    "curtain",
    "fridge",
    "shower",
    "toilet",
    "sink",
    "bath",
    "others"
]

SEMANTIC2NAME = {k: v for k, v in enumerate(SEMANTIC2NAME)}


def denormalize_img(img):
    mean = np.zeros(img.shape)
    mean[0, :, :] = 0.485
    mean[1, :, :] = 0.456
    mean[2, :, :] = 0.406
    scale = np.zeros(img.shape)
    scale[0, :, :] = 0.229
    scale[1, :, :] = 0.224
    scale[2, :, :] = 0.225
    return img * scale + mean


def extract_2d_bbox(obj, img):
    """for tracking"""

    img_h, img_w, _ = img.shape
    bbox = obj[2: 6].detach().cpu().numpy()
    bbox[0] *= img_w
    bbox[1] *= img_h
    bbox[2] *= img_w
    bbox[3] *= img_h
    return bbox


def extract_bv_bbox(obj, T_cw):
    """ draw bird eye view of a 3D bbox
    """

    azimuth = obj[12].detach().cpu().numpy()
    t_wo = obj[9: 12].detach().cpu().numpy()
    dimensions = obj[6: 9].detach().cpu().numpy()
    corners = geo_utils.get_corner_by_dims(dimensions)
    # rotate to camera before translate
    R_azi = R.from_euler("y", azimuth).as_dcm()
    corners = corners @ R_azi.T
    
    # translate from world to camera    
    t_co = geo_utils.get_homogeneous(t_wo[None, :]) @ T_cw.T
    corners += t_co[:, :3]

    # transform to camera coordinate
    corners_bv = corners

    return corners_bv


def draw_rectangle(axis, bbox, id_, is_frame):
    if id_ == -1:
        color = mcd.XKCD_COLORS['xkcd:black'].upper()
    else:
        color = list(mcd.XKCD_COLORS.values())[id_].upper()

    x_values = [bbox[0, 0], bbox[1, 0], bbox[2, 0], bbox[3, 0], bbox[0, 0]]
    # y is on the depth axis
    y_values = [bbox[0, 2], bbox[1, 2], bbox[2, 2], bbox[3, 2], bbox[0, 2]]
    axis.plot(x_values, y_values, color=color, linewidth=3 if is_frame else 6)


def draw_bv_bboxes(axis, bbox_list):
    axis.set_aspect('equal')
    for bbox in bbox_list:
        draw_rectangle(axis, bbox[0], bbox[1], bbox[2]=='frame')


def draw_2d_bboxes(axis, img, bbox_list):
    """ an elemtn of bbox_list: [bbox_params (4), obj_id (1)]
    """
    axis.imshow(img)
    for bbox in bbox_list:
        bbox_params = bbox[0]
        bbox_w = bbox_params[2] - bbox_params[0]
        bbox_h = bbox_params[3] - bbox_params[1]

        text = bbox[1]
        axis.text(bbox_params[0], bbox_params[1], text, fontsize=10,
                bbox=dict(facecolor='yellow', alpha=0.5))

        obj_id = bbox[2]
        if obj_id == -1:
            color = mcd.XKCD_COLORS['xkcd:black'].upper()
        else:
            color = list(mcd.XKCD_COLORS.values())[obj_id].upper()
        # Create a Rectangle patch
        rect = patches.Rectangle(
            (bbox_params[0], bbox_params[1]), bbox_w, bbox_h, linewidth=3,
            edgecolor=color, facecolor='none')
        # Add the patch to the Axes
        axis.add_patch(rect)


def get_bboxes_from_matching(tracks, detections, matches, T_cw, img):
    bboxes_draw_gt = []
    bboxes_2d_draw = []
    used_objs_frame = []
    cur_obj_id = 0
    for track_idx, track in enumerate(tracks):
        if track_idx in matches[:, 0]:
            pair = matches[track_idx == matches[:, 0]][0]
            if pair[1] != -1:
                used_objs_frame.append(pair[1])
                obj1 = detections[:, pair[1]] * 1.
                bbox_bv1 = extract_bv_bbox(obj1, T_cw=T_cw)
                bbox_2d1 = extract_2d_bbox(obj1, img)
                bboxes_2d_draw.append([bbox_2d1, 'frame', track_idx])
                bboxes_draw_gt.append([bbox_bv1, track_idx, 'frame'])
            track = track * 1.
            obj0 = torch.mean(track[:, track[0, :] != -1], dim=1)
            bbox_bv0 = extract_bv_bbox(obj0, T_cw=T_cw)
            bboxes_draw_gt.append([bbox_bv0, track_idx, 'track'])
        cur_obj_id += 1

    for pair in matches:
        if pair[1] != -1 and pair[1] not in used_objs_frame:
            used_objs_frame.append(pair[1])
            obj1 = detections[:, pair[1]] * 1.
            bbox_bv1 = extract_bv_bbox(obj1, T_cw)
            bbox_2d1 = extract_2d_bbox(obj1, img)
            bboxes_2d_draw.append([bbox_2d1, 'frame', cur_obj_id])
            bboxes_draw_gt.append([bbox_bv1, cur_obj_id, 'frame'])
            cur_obj_id += 1
    
    for detect_id in range(detections.shape[1]):
        if detect_id not in used_objs_frame:
            obj1 = detections[:, detect_id] * 1.
            bbox_bv1 = extract_bv_bbox(obj1, T_cw)
            bbox_2d1 = extract_2d_bbox(obj1, img)
            bboxes_2d_draw.append([bbox_2d1, 'frame', -1])
            bboxes_draw_gt.append([bbox_bv1, -1, 'frame'])

    return bboxes_draw_gt, bboxes_2d_draw


def xyz_from_bbox_offset_depth(
    img_size, intr_mat, bbox_cxcywh, offset, depth, dimensions, theta):
    
    """
    In dim2corners, we use world coordinate, which is z axis up, and the roation
    started from [1, 0, 0] following right-hand rule.
    Input dimensions follows [long axis, short axis and height], and input theta 
    is the rotation of the long axis away from the camera heading direction.
    Therefore, we need to switch  


    Args:
        theta: under the bird eye view, rotation of the long axis from the 
            camera heading direction (z axis)
        dimensions: an array of 3 elements, long axis, short axis, height
    
    Returns:
        2D bounding box and 3D corners.
    """
    
    ori_width, ori_height = img_size
    cx, cy, w, h = bbox_cxcywh
    cx *= ori_width
    w *= ori_width
    cy *= ori_height
    h *= ori_height
    bbox = np.array([
        [cx-w/2, cy-h/2],
        [cx+w/2, cy+h/2]])
    resize_offset = offset * np.array([ori_width, ori_height])
    xy = resize_offset + np.mean(bbox, axis=0)
    depth = np.array([depth])
    t_co = geo_utils.unproject(xy[None, :], depth, intr_mat)
    theta = -theta * (180 / 30) / 180 * np.pi
    corners_3d = box_utils.dim2corners(dimensions[[1, 0, 2]], theta)
    
    # convert from world coordinate (z axis up) to cam coordiante
    # (z axis forward and y axis down)
    corners_3d = corners_3d[:, [0, 2, 1]]
    corners_3d[:, 2] *= -1
    corners_3d += t_co
    return bbox, corners_3d


def save_tracking_result(
    in_data, assignments, n_iters, output_dir, eval=False):
    n_tracks = in_data['track_batch_split'][0]
    n_detections = in_data['detection_batch_split'][0]
    tracks = in_data['tracks'][:n_tracks]
    detections = in_data['detections'][0, :, :n_detections]
    gt_matches = in_data['gt_matches'][0]

    img_path = in_data['img_names'][0]
    T_cw = in_data['poses'][0]
    T_cw = np.eye(4)

    img = cv2.imread(img_path, -1)

    # pred_matches = get_matching(assignments)[0].detach().cpu().numpy()
    padding = np.arange(n_tracks)
    # pred_matches = np.concatenate([padding[None, :], pred_matches], axis=0).T
    pred_matches = hungarian_matching(assignments[0][0, :n_tracks, :n_detections], 0.2)
    out_matches = np.stack([np.arange(n_tracks), np.zeros(n_tracks)-1], axis=1)
    out_matches[pred_matches.astype(np.int32), 1] = np.arange(len(pred_matches))
    out_matches = out_matches.astype(np.int32)
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    
    bboxes_by_draw, bboxes_2d_draw = get_bboxes_from_matching(
        tracks, detections, gt_matches, T_cw, img)
    draw_bv_bboxes(axs[0, 0], bboxes_by_draw)
    draw_2d_bboxes(axs[1, 0], img, bboxes_2d_draw)
    
    bboxes_by_draw, bboxes_2d_draw = get_bboxes_from_matching(
        tracks, detections, out_matches, T_cw, img)
    draw_bv_bboxes(axs[0, 1], bboxes_by_draw)
    draw_2d_bboxes(axs[1, 1], img, bboxes_2d_draw)
    if not eval:
        out_dir = os.path.join(output_dir, "vis")
    else:
        out_dir = os.path.join(output_dir, "vis_eval")  
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)  
    plt.savefig(os.path.join(out_dir, "{}.png".format(n_iters)))
    plt.close()


def save_detection_result(
    samples, targets, img_paths, outputs, n_iters, out_dir, eval):
    """ save prediction results
    """

    rgb = samples.tensors[0].detach().cpu().numpy()
    mask = samples.mask[0]
    nonzeros = torch.nonzero(~mask).cpu().numpy()
    y_min, x_min = np.min(nonzeros, axis=0)
    y_max, x_max = np.max(nonzeros, axis=0)
    ori_width = x_max - x_min
    ori_height = y_max - y_min
    rgb = (denormalize_img(rgb).transpose(1, 2, 0) * 255).astype(np.uint8)
    img = Image.fromarray(rgb)

    bbox_2d_gt = []
    bbox_bv_gt = []
    bbox_2d_pred = []
    bbox_bv_pred = []
    
    intr_mat = np.array([
        [1170.0, 0, 647.0],
        [0, 1170, 483.0],
        [0, 0, 1]
    ])
    for obj in targets[0]['objects']:
        obj = obj.detach().cpu().numpy()
        obj_class = obj[0]
        bbox_2d, corners_3d = xyz_from_bbox_offset_depth(
            [ori_width, ori_height], intr_mat,
            obj[1: 5], obj[8: 10], obj[10], obj[5: 8], obj[11]
        )
        bbox_2d_gt.append([bbox_2d.flatten(), SEMANTIC2NAME[int(obj_class)], 0])
        bbox_bv_gt.append([corners_3d, 0, "frame"])

    for bbox, offset, angle, dimensions, depth, src_logits in zip(
        outputs['pred_boxes'][0], outputs['pred_offset'][0],
        outputs['pred_angle'][0], outputs['pred_size'][0],
        outputs['pred_depth'][0], outputs['pred_logits'][0]):
        score, class_id = src_logits.softmax(-1)[:-1].max(-1)
        class_id = int(class_id.cpu().numpy())
        if score > 0.2:
            color = "green"
        else:
            color = "yellow"
            continue
        angle = torch.argmax(angle).detach().cpu().numpy()
        bbox_2d, corners_3d = xyz_from_bbox_offset_depth(
            [ori_width, ori_height], intr_mat,
            bbox.detach().cpu().numpy(),
            offset.detach().cpu().numpy(),
            depth.detach().cpu().numpy()[0],
            dimensions.detach().cpu().numpy(),
            angle,
        )
        bbox_2d_pred.append([bbox_2d.flatten(), SEMANTIC2NAME[class_id], -1])
        bbox_bv_pred.append([corners_3d, -1, "frame"])
    
    if not eval:
        out_dir = os.path.join(out_dir, "vis")
    else:
        out_dir = os.path.join(out_dir, "vis_eval")  
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))

    draw_bv_bboxes(axs[0], bbox_bv_gt)
    draw_bv_bboxes(axs[0], bbox_bv_pred)
    
    draw_2d_bboxes(axs[1], img, bbox_2d_gt)
    draw_2d_bboxes(axs[1], img, bbox_2d_pred)

    plt.savefig(os.path.join(out_dir, "{}.png".format(n_iters)))
    plt.close()


def plot_loss(out_dir, loss_list, epoch):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(len(loss_list)), loss_list)
    plt.savefig(os.path.join(out_dir, "loss_epoch{}.png".format(epoch)))
    plt.close()


def bbox_lineset(corners):
    lines = []
    orders = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]
    for order in orders:
        lines.append(corners[order, :])
    return lines
