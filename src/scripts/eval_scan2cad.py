import argparse
import numpy as np
import torch
import json
import sys
import os
import pickle
import quaternion
import collections
from typing import Union, Tuple, Sequence

import src.utils.box_utils as box_utils


"""
To evaluate on Vid2CAD, use min_views: 80
"""


DEBUG = True
SCANNET_DIR = "./data/ScanNet/scans"
VID2CAD_PATH = "/home/kejie/Downloads/results_vid2cad.csv"
SCAN2CAD_PATH = "/home/kejie/Datasets/Scan2CAD/full_annotations.json"

CARE_CLASSES = {
    "03211117": "display",
    "04379243": "table",
    "02808440": "bathtub",
    "02747177": "trashbin",
    "04256520": "sofa",
    "03001627": "chair",
    "02933112": "cabinet",
    "02871439": "bookshelf",
}

DETECTOR_CLASS_MAPPER = {
    0: "03211117",
    1: "04379243",
    2: "02808440",
    3: "02747177",
    4: "04256520",
    5: "03001627",
    6: "02933112",
    7: "02871439",
}   


def get_homogeneous(
    pts: Union['np.ndarray', 'torch.tensor']
    ) -> Union['np.ndarray', 'torch.tensor']:
    """ convert [(b), N, 3] pts to homogeneous coordinate

    Args:
        pts ([(b), N, 3] Union['np.ndarray', 'torch.tensor']): input point cloud

    Returns:
        homo_pts ([(b), N, 4] Union['np.ndarray', 'torch.tensor']): output point
            cloud

    Raises:
        ValueError: if the input tensor/array is not with the shape of [b, N, 3]
            or [N, 3]
        TypeError: if input is not either tensor or array
    """

    batch = False
    if len(pts.shape) == 3:
        pts = pts[0]
        batch = True
    elif len(pts.shape) == 2:
        pts = pts
    else:
        raise ValueError("only accept [b, n_pts, 3] or [n_pts, 3]")

    if isinstance(pts, torch.Tensor):
        ones = torch.ones_like(pts[:, 2:])
        homo_pts = torch.cat([pts, ones], axis=1)
        if batch:
            return homo_pts[None, :, :]
        else:
            return homo_pts
    elif isinstance(pts, np.ndarray):
        ones = np.ones_like(pts[:, 2:])
        homo_pts = np.concatenate([pts, ones], axis=1)
        if batch:
            return homo_pts[None, :, :]
        else:
            return homo_pts
    else:
        raise TypeError("wrong data type")


def get_corner_by_dims(dimensions) -> np.ndarray:
    """get 8 corner points of 3D bbox defined by self.dimensions

    Returns:
        a np.ndarray with shape [8,3] to represent 8 corner points'
        position of the 3D bounding box.
    """

    w, h, l = dimensions[0], dimensions[1], dimensions[2]
    x_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    y_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]
    z_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    corner_pts = np.array([x_corners, y_corners, z_corners], dtype=np.float32).T
    return corner_pts


def make_M_from_tqs(t, q, s):
    q = np.quaternion(q[0], q[1], q[2], q[3])
    T = np.eye(4)
    T[0:3, 3] = t
    R = np.eye(4)
    R[0:3, 0:3] = quaternion.as_rotation_matrix(q)
    S = np.eye(4)
    S[0:3, 0:3] = np.diag(s)

    M = T.dot(R).dot(S)
    return M


def read_meta_file(meta_file):
    lines = open(meta_file).readlines()
    for line in lines:
        if 'axisAlignment' in line:
            axis_align_matrix = [float(x) \
                for x in line.rstrip().strip('axisAlignment = ').split(' ')]
            break
    axis_align_matrix = np.array(axis_align_matrix).reshape((4,4))
    return axis_align_matrix


def read_csv(filename, skip_header=False):
    import csv
    rows = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        if skip_header:
            next(reader, None)
        for row in reader:
            if len(row) == 1:
                rows.append(row[0])
            else:
                rows.append(row)
    return rows


def load_prediction_from_vid2cad(view_threshold, axis_align_matrices, box2cad=None):
    file_path = VID2CAD_PATH
    alignments = read_csv(file_path)
    predictions = {}
    for alignment in alignments[1:]:  # skip the first line as titles
        scan_id = f"scene{alignment[0]}"
        axis_align_matrix = axis_align_matrices[scan_id]
        if scan_id not in predictions:
            predictions[scan_id] = []

        catid_cad = alignment[1]
        if catid_cad not in CARE_CLASSES:
            continue
        id_cad = alignment[2]
        cadkey = catid_cad + "_" + id_cad
        b2c = np.asarray(box2cad[cadkey], dtype=np.float64)

        t = np.asarray(alignment[3:6], dtype=np.float64)
        q0 = np.asarray(alignment[6:10], dtype=np.float64)
        q = np.quaternion(q0[0], q0[1], q0[2], q0[3])
        s = np.asarray(alignment[10:13], dtype=np.float64)/2
        s = s * b2c.diagonal()[:-1]

        T_wo = np.eye(4)
        T_wo[0:3, 3] = t
        T_wo[0:3, 0:3] = quaternion.as_rotation_matrix(q)
        pred_bbx = get_corner_by_dims(s)
        pred_bbx = (get_homogeneous(pred_bbx) @ T_wo.T)[:, :3]
        pred_bbx = (get_homogeneous(pred_bbx) @ axis_align_matrix.T)[:, :3]
        
        if int(alignment[14]) < view_threshold:
            continue
        out = {
            "class": catid_cad,
            "bbox": pred_bbx,
            "num_frames": alignment[14],
            "scores": alignment[15]
        }
        predictions[scan_id].append(out)

    return predictions


def load_prediction_ours(result_dir, min_views):
    scene_list = [f for f in os.listdir(result_dir) if f.startswith("scene")]
    predictions = {}
    for scene in scene_list:
        n_frames = len(os.listdir(os.path.join(SCANNET_DIR, scene, "frames/color")))
        predictions[scene] = []
        result_path = os.path.join(result_dir, scene, scene)
        if not os.path.exists(result_path):
            print(f"{result_path} does not exist")
            continue
        with open(result_path, "rb") as f:
            tracks = pickle.load(f)
        n_objs = len(tracks["tracks"])
        for obj_id in range(n_objs):
            if len(tracks["tracks"][obj_id]) < min_views:
                continue
            obj_class = int(np.median(tracks['tracks'][obj_id][:, 1]))
            if DETECTOR_CLASS_MAPPER[obj_class] not in CARE_CLASSES:
                continue
            out = {
                "bbox": tracks['bboxes_qc'][obj_id],
                "class": DETECTOR_CLASS_MAPPER[obj_class]
            }
            predictions[scene].append(out)
    return predictions


def parse_scan2cad_annotations(annotations, T_align=None):
    annotations_ready = []
    gt_classes = []
    T_ws = make_M_from_tqs(
        annotations["trs"]["translation"],
        annotations["trs"]["rotation"],
        annotations["trs"]["scale"]
    )
    T_sw = np.linalg.inv(T_ws)
    for i, annotation in enumerate(annotations['aligned_models']):
        cat_gt = annotation['catid_cad']
        gt_classes.append(cat_gt)
        t = annotation["trs"]["translation"]
        q = annotation["trs"]["rotation"]
        s = annotation["trs"]["scale"]
        if min(s) < 1e-3:
            continue
        scales_gt = annotation['bbox'] * np.asarray(s) * 2
        # evaluate t, r, s separately
        T_wo = make_M_from_tqs(t, q, np.ones_like(s))
        T_wo = T_sw @ T_wo
        bbox = get_corner_by_dims(scales_gt)
        bbox = (get_homogeneous(bbox) @ T_wo.T)[:, :3]
        if T_align is not None:
            bbox = (get_homogeneous(bbox) @ T_align.T)[:, :3]
        if cat_gt not in CARE_CLASSES:
            continue
        annotations_ready.append(tuple([cat_gt, bbox]))
    return annotations_ready


def match_sequence(
    total_gts, total_preds, total_tps, predictions, 
    gts, axis_align_matrix, threshold, scan_id
):
    used_gts = []
    for gt in gts:
        total_gts[gt[0]] += 1
    for prediction in predictions:
        pred_class = prediction["class"]
        pred_bbx = prediction["bbox"]
        total_preds[pred_class] += 1
        for i, gt in enumerate(gts):
            gt_class = gt[0]
            gt_bbx = gt[1]
            if gt_class == pred_class:
                iou, _ = box_utils.box3d_iou(gt_bbx, pred_bbx)
                if iou > threshold and i not in used_gts:
                    used_gts.append(i)
                    total_tps[pred_class] += 1


def get_f1(gts, predictions, tps):
    total_gts = 0
    total_preds = 0
    total_tps = 0
    for c in CARE_CLASSES:
        if gts[c] == 0:
            accu = 0
        else:
            accu = tps[c] / predictions[c]
        if gts[c] == 0:
            recall = 0
        else:
            recall = tps[c] / gts[c]
        print("class {}:".format(CARE_CLASSES[c]))
        print("accuracy: {}".format(accu))
        print("recall: {}".format(recall))
        f1 = 2 * accu * recall / (accu + recall) if accu + recall != 0 else 0
        print("F1: {}".format(f1))
        total_gts += gts[c]
        total_preds += predictions[c]
        total_tps += tps[c]
    accuracy = total_tps / total_preds
    recall = total_tps / total_gts
    f1 = 2 * accuracy * recall / (accuracy + recall) 
    print("average accuracy: {}, recall: {}, F1: {}".format(accuracy, recall, f1))
    print("------------")


def get_meta_matrix(sequences):
    out = {}
    for scan_id in sequences:
        meta_file = os.path.join(SCANNET_DIR, scan_id, scan_id + '.txt') # includes axisAlignment info for the train set scans. 
        axis_align_matrix = read_meta_file(meta_file)
        out[scan_id] = axis_align_matrix
    return out


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--source")
    arg_parser.add_argument("--result_dir")
    arg_parser.add_argument("--threshold", type=float, default=0.25)
    arg_parser.add_argument("--min_views", type=int, default=1)
    args = arg_parser.parse_args()

    assert args.source in ["vid2cad", "ours"]

    with open("./data/ScanNet/scannetv2_val.txt", "r") as f:   
        sequences = f.read().splitlines()

    axis_align_matrices = get_meta_matrix(sequences)

    scan2cad_path = SCAN2CAD_PATH
    with open(scan2cad_path, "r") as f:
        scan2cad = json.load(f)

    total_gts = {k: 0 for k in CARE_CLASSES}
    total_preds = {k: 0 for k in CARE_CLASSES}
    total_tps = {k: 0 for k in CARE_CLASSES}

    if args.source == "vid2cad":
        b2c_path = "./box2cad.json"
        with open(b2c_path, "r") as f:
            b2c = json.load(f)
        predictions = load_prediction_from_vid2cad(
            args.min_views, axis_align_matrices, b2c)
    else:
        predictions = load_prediction_ours(args.result_dir, args.min_views)
    for scan in scan2cad:
        scan_id = scan['id_scan']
        if scan_id not in predictions:
            continue

        axis_align_matrix = axis_align_matrices[scan_id]
        gts = parse_scan2cad_annotations(scan, axis_align_matrix)

        match_sequence(
            total_gts, total_preds, total_tps,
            predictions[scan_id], gts, axis_align_matrix, 
            args.threshold, scan_id)
        # print(total_gts)
        # print(total_preds)
        # print(total_tps)
        # print("----------")
    get_f1(total_gts, total_preds, total_tps)

if __name__ == "__main__":
    main()