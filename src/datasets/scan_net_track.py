"""
ScanNet dataset loader for associator training
input data format: 
   0   |    1   |   2 - 6   |  6 - 9   | 9 - 12 | 12 | 13  | 14  | 15 - 79  |    79-83   |
 img_id|class_id|detect_bbox|dimensions|  t_wo  | azi|score|gt_id|shape_code|project_bbox|

the processed data format is
   0   |    1   |   2 - 6    |  6 - 9   |9 - 12|  12   | 13    | 14  |   15-79  |
 img_id|class_id|project_bbox|dimensions| t_wo |sin_azi|cos_azi|score|shape_code|


"""

import cv2
import os
import json
import math
import numpy as np
from pathlib import Path
from PIL import Image
import pickle
import scipy.io as sio
import torch
import torch.utils.data

import src.datasets.transforms as T
import src.datasets.scannet_utils as scannet_utils
from src.utils.geometry_utils import get_homogeneous



class ScanNetTrack(torch.utils.data.Dataset):
    @classmethod
    def collater(cls, data_list):
        """ batching
        """
        # assert len(data_list) == 1, "use batch_size 1 now"
        # return data_list[0]
        max_dets = 30
        matches = []
        gt_scores_list = []
        tracks = []
        detections = []
        img_names = []
        poses = []
        track_batch_split = []
        detection_batch_split = []
        global_track_ids = []
        valid_list = []
        gt_list = []

        n_features = data_list[0]['detections'].shape[0]
        detections = torch.ones((len(data_list), n_features, max_dets), dtype=torch.float) * -1

        for b_id, data in enumerate(data_list):
            tracks.append(data['tracks'])
            gt_scores_list.append(data['scores'])
            matches.append(data['match'])
            img_names.append(data['img_path'])
            poses.append(data['pose'])
            _n_tracks = data['tracks'].shape[0]
            _n_detections = data['detections'].shape[1]
            track_batch_split.append(_n_tracks)
            detection_batch_split.append(_n_detections)
            global_track_ids.append(data['global_track_ids'])
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
            gt_scores[track_0: track_1, detect_0: detect_1] = gt_scores_list[b_idx]
            track_0 = track_1
            detect_0 = detect_1

        tracks = torch.cat(tracks, dim=0)
        return {
            "tracks": tracks,
            "detections": detections,
            "gt_matches": matches,
            "gt_scores": gt_scores,
            "gt_masks": gt_masks,
            "img_names": img_names,
            "track_batch_split": track_batch_split,
            "detection_batch_split": detection_batch_split,
            "poses": poses,
            "global_track_ids": global_track_ids,
            "valid_list": valid_list}

    def __init__(self):
        super(ScanNetTrack, self).__init__()
        
        self.base_dir = "./data/ScanNet/"
        self.img_path = os.path.join(
            self.base_dir, "scans", "{}", "frames/color/{}.jpg")
        self.pose_path = os.path.join(
            self.base_dir, "scans", "{}", "frames/pose/{}.txt")
        self.intr_path = os.path.join(
            self.base_dir, "scans", "{}", "frames/intrinsic/intrinsic_color.txt"
        )
        self.meta_path = os.path.join(self.base_dir, "scans", "{0}/{0}.txt")
        self.img_h = 968
        self.img_w = 1296
        self.max_objs = 30
        self.subsample_rate = 2

        self.file_indices = []
        self.files = {}
        # with open("./data/ScanNet/imovotenet_scan2cad/{}_tracking_ls".format(split), "rb") as f:
        #     data = pickle.load(f)
        # self.files = data['files']
        with open("./data/ScanNet/scannet_imgs", "rb") as f:
            data = pickle.load(f)
        self.files = data
        # if split == "val":
        #     self.file_indices = data['file_indices'][:10000]
        # else:
        #     self.file_indices = data['file_indices']

    def _subsample_obj_ids(self, n_objs, max_objs):
        obj_ids = np.arange(n_objs)
        return np.random.permutation(obj_ids)[:max_objs]

    def _clip_objs(self, objs):
        """ clip the number of objs under self.max_objs
        """
        if len(objs) > self.max_objs:
            indices = np.arange(len(objs))
            indices = np.random.permutation(indices)
            objs = np.asarray(objs)[indices[:self.max_objs]].tolist()
        return objs

    def _get_match_list(self, track_ids, obj_ids):
        """object matching between two frames
        """

        matched_list = []
        used_targets = []
        for source_id, track_id in enumerate(track_ids):
            matched = False
            for target_id, obj_id in enumerate(obj_ids):
                if track_id == obj_id:
                    assert matched == False
                    matched = True
                    used_targets.append(obj_id)
                    matched_list.append([source_id, target_id])
            if not matched:
                matched_list.append([source_id, -1])
        
        for target_id, obj_id in enumerate(obj_ids):
            if obj_id in used_targets:
                continue
            matched = False
            for source_id, track_id in enumerate(track_ids):
                if obj_id == track_id:
                    assert matched == False
                    matched = True
                    matched_list.append([source_id, target_id])
            if not matched:
                matched_list.append([-1, target_id])
        matched_list = np.asarray(matched_list)
        return matched_list

    def _preprocess(self, objs, img_size, T_wc, cam_azi):
        """ normalize bounding box, class id, and get valid mask, to tensor
        
        [class_id, min_x, min_y, max_x, max_y, dimensions0, dimensions1,
         dimensions2, location0, location1, location2]
        """

        T_cw = np.linalg.inv(T_wc)
        in_ = torch.ones((len(objs), 79)) * -1  # 11 input dimension, see above
        img_w, img_h = img_size
        for idx, obj in enumerate(objs):
            tmp = np.zeros(79) - 1
            tmp[:2] = obj[:2]
            normalized_bbox = obj[2: 6] / np.array([img_w, img_h, img_w, img_h])
            tmp[2: 6] = normalized_bbox
            tmp[6: 9] = obj[6: 9]
            t_wo = obj[9: 12] 
            t_co = (get_homogeneous(t_wo[None, :]) @ T_cw.T)[0, :3]
            tmp[9: 12] = t_co
            angle = obj[12] - cam_azi
            sin_azi = np.sin(angle)
            cos_azi = np.cos(angle)
            tmp[12: 14] = np.array([sin_azi, cos_azi])
            tmp[14] = obj[13]
            tmp[15: 79] = obj[14:78]
            in_[idx] = torch.tensor(tmp, dtype=torch.float32)
        return in_

    def _preprocess_tracks(self, tracks, img_size, T_wc, cam_azi, n_times=100):
        """
        Args:
            n_times: how many time stamps used for each track
        """

        T_cw = np.linalg.inv(T_wc)
        img_w, img_h = img_size
        in_ = torch.ones((len(tracks), n_times, 79)) * -1
        for idx, track in enumerate(tracks):
            n_steps = len(track)
            tmp = np.zeros((n_steps, 79)) - 1
            projected_bbox = track[-1, -4:]
            assert not (projected_bbox == -1).all(), "wrong projected bbox"
            projected_bbox /= np.array([img_w, img_h, img_w, img_h])
            projected_bbox = np.clip(projected_bbox, a_min=-1, a_max=2)
            track[:, 2: 6] = projected_bbox
            time_steps = track[:, 0]
            obj_class = track[:, 1]
            normalized_bbox = track[:, 2: 6]
            dimensions = track[:, 6: 9]
            t_wo = get_homogeneous(track[:, 9: 12])
            t_co = (t_wo @ T_cw.T)[:, :3]
            angle = track[:, 12] - cam_azi
            sin_azi = np.sin(angle)
            cos_azi = np.cos(angle)
            scores = track[:, 13]

            tmp[:, 0] = time_steps
            tmp[:, 1] = obj_class
            tmp[:, 2: 6] = normalized_bbox
            tmp[:, 6: 9] = dimensions
            tmp[:, 9: 12] = t_co
            tmp[:, 12] = sin_azi
            tmp[:, 13] = cos_azi
            tmp[:, 14] = scores
            tmp[:, 15: 79] = track[:, 14: 78]
            tmp = torch.tensor(tmp, dtype=torch.float32)
            if n_steps > n_times:
                in_[idx, :, :] = tmp[-n_times:, :]
            else:
                in_[idx, :n_steps, :] = tmp
        return in_

    def _get_objs_from_frame(self, frame):
        """ get a list of 
        """

        objs = []
        obj_ids = []
        used_indices = np.arange(79, dtype=np.int32).tolist()
        for obj_id, obj in enumerate(frame):
            if obj[0] != -1:
                obj_ids.append(obj_id)
                objs.append(obj[used_indices])
        return objs, obj_ids

    def _get_objs_from_tracks(self, tracks, projected_bboxes):    
        out_tracks = []
        track_ids = []
        for track_id, track in enumerate(tracks):
            if_valid = track[:, 0] != -1
            if (if_valid).any():
                valid_frames = track[if_valid]
                projected_bbox = projected_bboxes[track_id]
                valid_frames[:, 79: 83] = projected_bbox
                out_tracks.append(valid_frames)
                track_ids.append(track_id)
        return out_tracks, track_ids

    def _concat_unmatched(self, objs_frame, unmatched, match_list):
        if len(unmatched):
            n_matched_objs = len(objs_frame)
            target_ids = np.arange(
                n_matched_objs, n_matched_objs + len(unmatched))
            source_ids = np.ones_like(target_ids) * -1
            unmatched_pairs = np.stack([source_ids, target_ids], axis=1)
            match_list = np.concatenate([match_list, unmatched_pairs], axis=0)
            objs_frame += [np.asarray(t) for t in unmatched]
        if len(objs_frame) > self.max_objs:
            match_list = match_list[:self.max_objs, :]
            del objs_frame[self.max_objs:]
        return objs_frame, match_list

    def get_by_sequence(self, sequence, frame_id, img_name, use_gt_det=False, n_times=100):
        """ get data by sequence and img name 
        """

        tracks = self.files[sequence]['tracks'][:, :frame_id, :]
        frame = self.files[sequence]['tracks'][:, frame_id, :]
        projected_bboxes = frame[:, -4:]
        if str(img_name) in self.files[sequence]['unmatched']:
            unmatched = self.files[sequence]['unmatched'][str(img_name)]
            unmatched = [np.delete(o, 14, axis=0) for o in unmatched]
        else:
            unmatched = []

        objs_frame, obj_ids = self._get_objs_from_frame(frame)
        # remove unobserved objects up to the current frame
        tracks, track_ids = self._get_objs_from_tracks(tracks, projected_bboxes)

        # get matching
        gt_matches = self._get_match_list(track_ids, obj_ids)

        # remove gt object ID
        global_track_ids = [int(t[0, 14]) for t in tracks]
        tracks = [np.delete(t, 14, axis=1) for t in tracks]
        objs_frame = [np.delete(o, 14, axis=0) for o in objs_frame]
        
        img = Image.open(self.img_path.format(sequence, img_name))
        img_size = img.size
        T_cw = scannet_utils.read_extrinsic(self.pose_path.format(sequence, img_name))
        T_wc = np.linalg.inv(T_cw)
        axis_align_mat = scannet_utils.read_meta_file(self.meta_path.format(sequence))
        T_wc = axis_align_mat @ T_wc
        cam_azi = scannet_utils.get_cam_azi(T_wc)
        # preprocess input
        tracks = self._preprocess_tracks(tracks, img_size, T_wc, cam_azi, n_times)

        if not use_gt_det:
            objs_frame, gt_matches = self._concat_unmatched(
                objs_frame, unmatched, gt_matches)
        
        detections = self._preprocess(objs_frame, img_size, T_wc, cam_azi)

        score_mat = np.zeros((len(tracks), len(detections)))  # [n_tracks, n_detections]
        for gt_match in gt_matches:
            if not -1 in gt_match:
                score_mat[gt_match[0], gt_match[1]] = 1
        # detections = detections.T
        # tracks = tracks.permute(0, 2, 1)
        target = {
            "detections": detections.T,  # [n_features, n_detections]
            "tracks": tracks.permute(0, 2, 1),  # [n_tracks, n_features, n_times]
            "match": gt_matches,
            "scores": torch.tensor(score_mat.astype(np.float32)),
            "img_path": self.img_path.format(sequence, img_name),
            "pose": T_cw,
            "global_track_ids": global_track_ids
        }
        return target

    def __getitem__(self, idx):
        """load and prepare data
        """

        sequence, frame_id, img_name = self.file_indices[idx]
        tracks = self.files[sequence]['tracks'][:, :frame_id, :]
        frame = self.files[sequence]['tracks'][:, frame_id, :]
        projected_bboxes = frame[:, -4:]
        if str(img_name) in self.files[sequence]['unmatched']:
            unmatched = self.files[sequence]['unmatched'][str(img_name)]
            unmatched = [np.delete(o, 14, axis=0) for o in unmatched]
        else:
            unmatched = []

        objs_frame, obj_ids = self._get_objs_from_frame(frame)
        # remove unobserved objects up to the current frame
        tracks, track_ids = self._get_objs_from_tracks(tracks, projected_bboxes)

        # get matching
        gt_matches = self._get_match_list(track_ids, obj_ids)

        # remove gt object ID
        global_track_ids = [int(t[0, 14]) for t in tracks]
        tracks = [np.delete(t, 14, axis=1) for t in tracks]
        objs_frame = [np.delete(o, 14, axis=0) for o in objs_frame]
        
        img = Image.open(self.img_path.format(sequence, img_name))
        img_size = img.size
        T_cw = scannet_utils.read_extrinsic(self.pose_path.format(sequence, img_name))
        T_wc = np.linalg.inv(T_cw)
        axis_align_mat = scannet_utils.read_meta_file(self.meta_path.format(sequence))
        T_wc = axis_align_mat @ T_wc
        cam_azi = scannet_utils.get_cam_azi(T_wc)
        # preprocess input
        tracks = self._preprocess_tracks(tracks, img_size, T_wc, cam_azi)
        objs_frame, gt_matches = self._concat_unmatched(
            objs_frame, unmatched, gt_matches)
        detections = self._preprocess(objs_frame, img_size, T_wc, cam_azi)

        score_mat = np.zeros((len(tracks), len(detections)))  # [n_tracks, n_detections]
        for gt_match in gt_matches:
            if not -1 in gt_match:
                score_mat[gt_match[0], gt_match[1]] = 1
        # detections = detections.T
        # tracks = tracks.permute(0, 2, 1)
        target = {
            "detections": detections.T,  # [n_features, n_detections]
            "tracks": tracks.permute(0, 2, 1),  # [n_tracks, n_features, n_times]
            "match": gt_matches,
            "scores": torch.tensor(score_mat.astype(np.float32)),
            "img_path": self.img_path.format(sequence, img_name),
            "pose": T_cw,
            "global_track_ids": global_track_ids
        }
        return target

    def __len__(self):
        return len(self.file_indices)


def build(image_set, args):
    dataset = ScanNetTrack(image_set)
    return dataset


if __name__ == "__main__":
    """ testing for ScanNet initialization and data loading
    """
    import pickle

    dataset = ScanNetTrack("train")
    with open("./data/ScanNet/train_data_files", "wb") as f:
        out = {
            "files": dataset.files,
            "file_indices": dataset.file_indices
        }
        pickle.dump(out, f)