import collections
import numpy as np
import os
import matplotlib.pyplot as plt
import quaternion
from scipy.spatial.transform import Rotation as R

import src.utils.box_utils as box_utils
import src.datasets.scannet_utils as scannet_utils
import src.utils.geometry_utils as geo_utils

import sys
sys.path.append("/home/kejie/repository/DOM")
from dom.libs.o3d_helper import load_scene_mesh, lineset_from_pc


CARE_CLASSES = {
    0: "cabinet",
    1: "bed",
    2: "chair",
    3: "sofa",
    4: "table",
    7: "bookshelf",
    10: "desk",
    12: "fridge",
    14: "toilet",
    16: "bath",
}
CARE_CLASSES = {
    0: "display",
    1: "table",
    2: "bathtub",
    # 3: "trashbin",
    4: "sofa",
    5: "chair",
    6: "cabinet",
    7: "bookshelf",
}
DEBUG = False



def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def get_iou_obb(bb1,bb2):
    iou3d, iou2d = box3d_iou(bb1,bb2)
    return iou3d


def get_iou_main(get_iou_func, args):
    return get_iou_func(*args)


def eval_det_cls(pred, gt, ovthresh=0.25, use_07_metric=False, get_iou_func=box_utils.box3d_iou):
    """ Generic functions to compute precision/recall for object detection
        for a single class.
        Input:
            pred: map of {img_id: [(bbox, score)]} where bbox is numpy array
            gt: map of {img_id: [bbox]}
            ovthresh: scalar, iou threshold
            use_07_metric: bool, if True use VOC07 11 point method
        Output:
            rec: numpy array of length nd
            prec: numpy array of length nd
            ap: scalar, average precision
    """

    # construct gt objects
    class_recs = {} # {img_id: {'bbox': bbox list, 'det': matched list}}
    npos = 0
    for img_id in gt.keys():
        bbox = np.array(gt[img_id])
        det = [False] * len(bbox)
        npos += len(bbox)
        class_recs[img_id] = {'bbox': bbox, 'det': det}
    # pad empty list to all other imgids
    for img_id in pred.keys():
        if img_id not in gt:
            class_recs[img_id] = {'bbox': np.array([]), 'det': []}

    # construct dets
    image_ids = []
    confidence = []
    BB = []
    for img_id in pred.keys():
        for box, score in pred[img_id]:
            image_ids.append(img_id)
            confidence.append(score)
            BB.append(box)
    confidence = np.array(confidence)
    BB = np.array(BB) # (nd,4 or 8,3 or 6)

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, ...]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        #if d%100==0: print(d)
        R = class_recs[image_ids[d]]
        bb = BB[d, ...].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            for j in range(BBGT.shape[0]):
                # iou = get_iou_main(get_iou_func, (bb, BBGT[j, ...]))
                # iou, _ = box_utils.box3d_iou(bb, BBGT[j, ...])
                bbox_a, bbox_b = np.zeros((2, 3)), np.zeros((2, 3))
                bbox_a[0, :] = np.min(bb, axis=0)
                bbox_a[1, :] = np.max(bb, axis=0)
                bbox_b[0, :] = np.min(BBGT[j, ...], axis=0)
                bbox_b[1, :] = np.max(BBGT[j, ...], axis=0)
                iou = box_utils.iou_3d(bbox_a, bbox_b)
                if iou > ovmax:
                    ovmax = iou
                    jmax = j
        #print d, ovmax
        if ovmax > ovthresh:
            if not R['det'][jmax]:
                tp[d] = 1.
                R['det'][jmax] = 1
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    #print('NPOS: ', npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap


def eval_det_cls_wrapper(arguments):
    pred, gt, ovthresh, use_07_metric, get_iou_func = arguments
    rec, prec, ap = eval_det_cls(pred, gt, ovthresh, use_07_metric, get_iou_func)
    return (rec, prec, ap)


def eval_det(pred_all, gt_all, ovthresh=0.25, use_07_metric=False, get_iou_func=box_utils.box3d_iou):
    """ Generic functions to compute precision/recall for object detection
        for multiple classes.
        Input:
            pred_all: map of {img_id: [(classname, bbox, score)]}
            gt_all: map of {img_id: [(classname, bbox)]}
            ovthresh: scalar, iou threshold
            use_07_metric: bool, if true use VOC07 11 point method
        Output:
            rec: {classname: rec}
            prec: {classname: prec_all}
            ap: {classname: scalar}
    """
    pred = {} # map {classname: pred}
    gt = {} # map {classname: gt}
    for img_id in pred_all.keys():
        for classname, bbox, score in pred_all[img_id]:
            if classname not in pred: 
                pred[classname] = {}
            if img_id not in pred[classname]:
                pred[classname][img_id] = []
            if classname not in gt: 
                gt[classname] = {}
            if img_id not in gt[classname]:
                gt[classname][img_id] = []
            pred[classname][img_id].append((bbox, score))
    for img_id in gt_all.keys():
        for classname, bbox in gt_all[img_id]:
            if classname not in gt:
                gt[classname] = {}
            if img_id not in gt[classname]:
                gt[classname][img_id] = []
            gt[classname][img_id].append(bbox)

    rec = {}
    prec = {}
    ap = {}
    for i, classname in enumerate(gt.keys()):
        print('Computing AP for class: ', classname)
        rec[classname], prec[classname], ap[classname] = eval_det_cls(pred[classname], gt[classname], ovthresh, use_07_metric, get_iou_func)
        print(CARE_CLASSES[classname], rec[classname][-1], prec[classname][-1], ap[classname])
    
        plt.subplot(2, 4, i+1)
        plt.plot(rec[classname], prec[classname])
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f"AP for {CARE_CLASSES[classname]} = {ap[classname]}")
    plt.show()
    return rec, prec, ap 


def eval_det_multiprocessing(pred_all, gt_all, ovthresh=0.25, use_07_metric=False, get_iou_func=box_utils.box3d_iou):
    """ Generic functions to compute precision/recall for object detection
        for multiple classes.
        Input:
            pred_all: map of {img_id: [(classname, bbox, score)]}
            gt_all: map of {img_id: [(classname, bbox)]}
            ovthresh: scalar, iou threshold
            use_07_metric: bool, if true use VOC07 11 point method
        Output:
            rec: {classname: rec}
            prec: {classname: prec_all}
            ap: {classname: scalar}
    """
    pred = {} # map {classname: pred}
    gt = {} # map {classname: gt}
    for img_id in pred_all.keys():
        for classname, bbox, score in pred_all[img_id]:
            if classname not in pred: pred[classname] = {}
            if img_id not in pred[classname]:
                pred[classname][img_id] = []
            if classname not in gt: gt[classname] = {}
            if img_id not in gt[classname]:
                gt[classname][img_id] = []
            pred[classname][img_id].append((bbox,score))
    for img_id in gt_all.keys():
        for classname, bbox in gt_all[img_id]:
            if classname not in gt: gt[classname] = {}
            if img_id not in gt[classname]:
                gt[classname][img_id] = []
            gt[classname][img_id].append(bbox)

    rec = {}
    prec = {}
    ap = {}
    p = Pool(processes=10)
    ret_values = p.map(eval_det_cls_wrapper, [(pred[classname], gt[classname], ovthresh, use_07_metric, get_iou_func) for classname in gt.keys() if classname in pred])
    p.close()
    for i, classname in enumerate(gt.keys()):
        if classname in pred:
            rec[classname], prec[classname], ap[classname] = ret_values[i]
        else:
            rec[classname] = 0
            prec[classname] = 0
            ap[classname] = 0
        print(classname, ap[classname])
    
    return rec, prec, ap 


# get top8 (most frequent) classes from annotations. 
def get_top8_classes_scannet():                                                                                                                                                                                                                                                                                           
    top = collections.defaultdict(lambda : "other")
    top["03211117"] = "display"
    top["04379243"] = "table"
    top["02808440"] = "bathtub"
    top["02747177"] = "trashbin"
    top["04256520"] = "sofa"
    top["03001627"] = "chair"
    top["02933112"] = "cabinet"
    top["02871439"] = "bookshelf"
    return top


def matching_scan2cad(predictions, annotations, scan_id):
    """match predictions to groundtruth objects in a sequence
    """

    classes = get_top8_classes_scannet()
    benchmark_per_class = {k: {'n_good': 0, "n_gt": 0, "n_pred": 0} for k in classes.keys()}

    for cat_id in classes:
        benchmark_per_class[cat_id]['n_gt'] = len([f for f in annotations['aligned_models'] if f['catid_cad'] == cat_id])


    T_w_s = scannet_utils.make_M_from_tqs(
        annotations["trs"]["translation"],
        annotations["trs"]["rotation"],
        annotations["trs"]["scale"]
    )

    threshold_translation = 0.2 # <-- in meter
    threshold_rotation = 20 # <-- in deg
    threshold_scale = 20 # <-- in %

    # read prediction
    predict_bboxes = []
    predict_tp = []
    gt_bbxs = []
    used_gt = []

    for track_id, prediction in enumerate(predictions):
        T_wo_pred = prediction['T_wo']
        T_wo_pred = T_w_s @ T_wo_pred
        scales_pred = prediction['scale']
        corners = geo_utils.get_corner_by_dims(scales_pred)
        pred_bbox = (geo_utils.get_homogeneous(corners) @ T_wo_pred.T)[:, :3]
        predict_bboxes.append(pred_bbox)
        predict_tp.append(False)
        cat_pred = prediction['class']

        for gt_id, model in enumerate(annotations['aligned_models']):
            cat_gt = model['catid_cad']
            if cat_pred != cat_gt:
                continue
            if gt_id in used_gt:
                continue
            t = model["trs"]["translation"]
            q = model["trs"]["rotation"]
            s = model["trs"]["scale"]
            Mcad = scannet_utils.make_M_from_tqs(t, q, s)
            scales_gt = model['bbox'] * np.asarray(s) * 2
            sym = model["sym"]
            # evaluate t, r, s separately
            T_w_o = scannet_utils.make_M_from_tqs(t, q, np.ones_like(s))
            gt_bbox = geo_utils.get_corner_by_dims(scales_gt)
            gt_bbox = (geo_utils.get_homogeneous(gt_bbox) @ T_w_o.T)[:, :3]
            gt_bbxs.append(gt_bbox)
            # eval t
            error_translation = np.linalg.norm(T_wo_pred[:3, 3] - T_w_o[:3, 3], ord=2)
            # eval s
            error_scale = 100 * np.abs(np.mean(scales_pred / scales_gt) - 1)
            # eval r
            R_gt = T_w_o[:3, :3]
            m = 1
            if sym == "__SYM_ROTATE_UP_2":
                m = 2
            elif sym == "__SYM_ROTATE_UP_4":
                m = 4
            elif sym == "__SYM_ROTATE_UP_INF":
                m = 36

            tmp = [
                geo_utils.geodesic_distance(np.eye(3), R.from_euler("y", i * 2 / m * np.pi).as_dcm() @ R_gt) 
                for i in range(m)]
            error_rotation = np.min(tmp)
            is_valid_transformation = (
                error_translation <= threshold_translation and
                error_rotation <= threshold_rotation and
                error_scale <= threshold_scale)
            gt_aabb = np.zeros((2, 3))
            pred_aabb = np.zeros((2, 3))
            gt_aabb[0, :] = np.min(gt_bbox, axis=0)
            gt_aabb[1, :] = np.max(gt_bbox, axis=0)
            pred_aabb[0, :] = np.min(pred_bbox, axis=0)
            pred_aabb[1, :] = np.max(pred_bbox, axis=0)
            iou = box_utils.iou_3d(pred_aabb, gt_aabb)
            is_valid_transformation = True if iou > 0.5 else False
            if is_valid_transformation:
                used_gt.append(gt_id)
                benchmark_per_class[cat_gt]['n_good'] += 1
                predict_tp[track_id]= True
                break

    if DEBUG:
        import open3d as o3d
        visual_list = []
        scene_mesh_path = os.path.join(
            "./data/ScanNet/scans", "{0}/{0}_vh_clean_2.ply".format(scan_id))
        visual_list.append(load_scene_mesh(scene_mesh_path, T_w_s))
        for pred_obj, tp in zip(predict_bboxes, predict_tp):
            if tp:
                color = np.array([1, 0, 0])
            else:
                color = np.array([0, 0, 1])
            visual_list.append(lineset_from_pc(pred_obj, colors=color))
        for gt_bbx in gt_bbxs:
            visual_list.append(lineset_from_pc(gt_bbx, colors=np.array([0, 1, 0])))
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.6, origin=[0, 0, 0])
        visual_list.append(mesh_frame)
        o3d.visualization.draw_geometries(visual_list)

    return benchmark_per_class

