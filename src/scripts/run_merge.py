import argparse
import numpy as np
import os
import pickle
import scipy
from sklearn.cluster import AgglomerativeClustering

import src.utils.box_utils as box_utils
import src.utils.geometry_utils as geo_utils

from src.utils.file_utils import get_date_time


def logging(out_dir, info):
    with open(os.path.join(out_dir, "log.txt"), "w") as f:
        for line in info:
            f.write(line + "\n")


def create_dir(out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)


def merge(tracks, mask, img_names):
    out_tracks = []
    dominant_class_id = [t[:, 1] for i, t in enumerate(tracks) if mask[i]]
    dominant_class_id = np.concatenate(np.asarray(dominant_class_id), axis=0)
    dominant_class_id = int(scipy.stats.mode(dominant_class_id).mode)

    for img_name in img_names:
        candidate_frames = []
        track_id_for_frames = []
        for track_id, track in enumerate(tracks):
            if not mask[track_id]:
                continue
            frame = track[track[:, 0] == img_name]
            if len(frame) == 0:
                continue
            assert len(frame) == 1
            candidate_frames.append(frame[0])
            track_id_for_frames.append(track_id)
        if len(candidate_frames)==0:
            continue
        elif len(candidate_frames) == 1:
            candidate_frames[0][1] = dominant_class_id
            out_tracks.append(candidate_frames[0])
        else:
            assert len(candidate_frames) > 1
            # select to use the detection from the longest track
            # i.e. merge the potentially fragemented track to a more
            # complete track
            track_length = [len(tracks[i]) for i in track_id_for_frames]
            selected_id = np.argmax(track_length)
            candidate_frames[selected_id][1] = dominant_class_id
            out_tracks.append(candidate_frames[selected_id])
    return np.asarray(out_tracks)


def get_bbox_from_track(track):
    state = np.mean(track, axis=0)
    center = state[9: 12]
    scale = state[6: 9]
    bbox = geo_utils.get_corner_by_dims(scale)
    bbox[:, 0] += center[0]
    bbox[:, 1] += center[1]
    bbox[:, 2] += center[2]
    bbox = box_utils.compute_oriented_bbox(bbox)
    return bbox


def co_visbility(track_0, track_1):
    vis_0_in_1 = [t for t in track_0 if t in track_1]
    vis_1_in_0 = [t for t in track_1 if t in track_0]
    co_visible = True if max(len(vis_0_in_1)/len(track_0), len(vis_1_in_0)/len(track_1)) > 0.5 else False
    return co_visible


def merge_process(data, img_names):
    n_objs = len(data['tracks'])
    merger = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=0.95,
        affinity="precomputed",
        linkage="average")

    if n_objs == 1:
        merged_tracks = data["tracks"]
    else:
        cost_mat = np.zeros((n_objs, n_objs))
        
        for i in range(n_objs):
            obj_class_0 = int(np.median(data["tracks"][i][:, 1]))
            bbox0 = data["bboxes_qc"][i]
            for j in range(i+1, n_objs):
                is_mergable = False
                obj_class_1 = int(np.median(data["tracks"][j][:, 1]))
                
                # SCANNET SETTING
                # if (obj_class_1 in [2, 3]) and (obj_class_0 in [2, 3]):
                #     is_mergable = True
                # if obj_class_0 == obj_class_1:
                #     is_mergable = True
                # if (obj_class_1 in [4, 10]) and (obj_class_0 in [4, 10]):
                #     is_mergable = True
                
                if (obj_class_1 in [4, 5]) and (obj_class_0 in [4, 5]):
                    is_mergable = True
                if obj_class_0 == obj_class_1:
                    is_mergable = True
                # if co_visbility(tracks[i][:, 0], tracks[j][:, 0]):
                #     is_mergable = False
                if not is_mergable:
                    cost_mat[i, j] = 1
                else:
                    # bbox1 = get_bbox_from_track(tracks[j])
                    bbox1 = data["bboxes_qc"][j]
                    iou = box_utils.box3d_iou(bbox0, bbox1)[0]
                    cost_mat[i, j] = 1 - iou

        cost_mat += cost_mat.T
        merger.fit(cost_mat)
        merge_result = merger.labels_
        cluster_ids = np.unique(merge_result)
        merged_tracks = []
        for id_ in cluster_ids:
            merged_track = merge(data["tracks"], id_ == merge_result, img_names)
            merged_tracks.append(merged_track)
    merged_tracks = [t for t in merged_tracks if len(t) > 0]
    return merged_tracks
