from copy import deepcopy
import logging
import numpy as np
import torch
from easydict import EasyDict

import src.datasets.scannet_utils as scannet_utils
import src.utils.geometry_utils as geo_utils
import src.utils.box_utils as box_utils
from src.super_quadric.sq_libs import SuperQuadric
from src.scripts.run_merge import merge_process
from src.scripts.run_multi_view import optim_process


class OdamProcess:
    def __init__(
        self,
        detector,
        associator,
        transforms,
        scale_prior,
        detect_threshold=0.6,
        match_threshold=0.1,
        score_threshold=0.8,
        representation="super_quadric",
        no_code=True
    ):

        self.detector = detector
        self.associator = associator
        self.transforms = transforms
        self.scale_prior = scale_prior
        self.detect_threshold = detect_threshold
        self.match_threshold = match_threshold
        self.score_threshold = score_threshold
        self.representation = representation
        self.no_code = no_code
        self.run_associator = False
        self.tracks = None
        self.sequence_meta = None
        self.logger = logging.getLogger('OdamProcess')


    def _collater(self, data_list):
        """ batching
        """
        # assert len(data_list) == 1, "use batch_size 1 now"
        # return data_list[0]
        max_dets = 30
        tracks = []
        detections = []
        poses = []
        track_batch_split = []
        detection_batch_split = []
        valid_list = []

        n_features = data_list[0]['detections'].shape[0]
        detections = torch.ones((len(data_list), n_features, max_dets), dtype=torch.float) * -1

        for b_id, data in enumerate(data_list):
            tracks.append(data['tracks'])
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
            "gt_masks": gt_masks,
            "track_batch_split": track_batch_split,
            "detection_batch_split": detection_batch_split,
            "poses": poses,
            "valid_list": valid_list}

    def _init_tracks(self, detections, T_wc):
        """ initialize tracking using detections

        Format of each output track:
        0: img_name
        1: obj_class
        2-6: detected bounding box
        6-9: dimensions
        9-12: t_wo
        12: azi_wo
        13: detect score
        14-78: -1 (placdeholder for object feature)
        78-82: -1 (placeholder for projected bounding box from super-quadric)
        """
        img_size = np.array([[
            self.sequence_meta.img_w,
            self.sequence_meta.img_h,
            self.sequence_meta.img_w,
            self.sequence_meta.img_h,
        ]])
        detect_bboxes = detections[:, 2:6] * img_size
        cam_azi = scannet_utils.get_cam_azi(T_wc)
        tracks = np.zeros((len(detections), 1, 82)) - 1
        tracks[:, 0, :9] = detections[:, :9]
        tracks[:, 0, 2: 6] *= img_size
        tracks[:, 0, -4:] = detect_bboxes
        sin = detections[:, 12]
        cos = detections[:, 13]
        azi_co = np.arctan2(sin, cos)
        azi_wo = azi_co + cam_azi
        t_co = detections[:, 9: 12]
        t_wo = (geo_utils.get_homogeneous(t_co) @ T_wc.T)[:, :3]
        tracks[:, 0, 9: 12] = t_wo
        tracks[:, 0, 12] = azi_wo
        tracks[:, 0, 13]  =  detections[:, 14]
        tracks = [t for t in tracks]
        return tracks

    def _preprocess_tracks(self, tracks, T_wc, cam_azi, n_times=100):
        """
        Args:
            n_times: how many time stamps used for each track
        """

        T_cw = np.linalg.inv(T_wc)
        in_ = torch.ones((len(tracks), n_times, 79)) * -1
        for idx, track in enumerate(tracks):
            n_steps = len(track)
            tmp = np.zeros((n_steps, 79)) - 1
            projected_bbox = track[-1, -4:]
            assert not (projected_bbox == -1).all(), "wrong projected bbox"
            projected_bbox /= np.array([
                self.sequence_meta.img_w,
                self.sequence_meta.img_h,
                self.sequence_meta.img_w,
                self.sequence_meta.img_h
            ])
            projected_bbox = np.clip(projected_bbox, a_min=-1, a_max=2)
            track[:, 2: 6] = projected_bbox
            time_steps = track[:, 0]
            obj_class = track[:, 1]
            normalized_bbox = track[:, 2: 6]
            dimensions = track[:, 6: 9]
            t_wo = geo_utils.get_homogeneous(track[:, 9: 12])
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
                in_[idx, :, :] = tmp[-100:, :]
            else:
                in_[idx, :n_steps, :] = tmp
        return in_

    def _prepare_tracks(self, T_wc, n_times=100):
        """ return track in a format that can be used for matcher

        use shape_decoder and the object pose to get projected bounding box
        """

        cam_azi = scannet_utils.get_cam_azi(T_wc)
        tracks = deepcopy(self.tracks)
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
            pixels = geo_utils.projection(box_3d_c, self.sequence_meta.K)
            x_min, y_min, _ = np.min(pixels, axis=0)
            x_max, y_max, _ = np.max(pixels, axis=0)
            track[:, -4:] = np.array([[x_min, y_min, x_max, y_max]])
            tracks[idx] = track
        tracks = self._preprocess_tracks(tracks, T_wc, cam_azi, n_times)
        tracks = tracks.permute(0, 2, 1)
        return tracks

    def _attach_to_tracks(
        self, pred_match, detections, T_wc, score_mat
    ):
        """ use the matching prediction to attach detections to tracks

        transform rotation and translate in detections to world coord. using T_wc
        """

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
            if match_score < self.score_threshold:
                continue
            attach_ = np.zeros((1, 82))
            attach_[0, :9] = detection[:9]
            attach_[0, 2: 6] *= np.array([
                self.sequence_meta.img_w,
                self.sequence_meta.img_h,
                self.sequence_meta.img_w,
                self.sequence_meta.img_h
            ])
            attach_[0, 9: 12] = t_wo[det_id]
            attach_[0, 12] = azi_wo[det_id]
            attach_[0, 13] = detection[14]
            if self.no_code:
                attach_[0, 14: 78] = -1
            else:
                attach_[0, 14: 78] = detection[15: 79]
            attach_[0, 78: 82] = detection[2: 6] * \
                np.array([
                    self.sequence_meta.img_w,
                    self.sequence_meta.img_h,
                    self.sequence_meta.img_w,
                    self.sequence_meta.img_h
                ])
            if match_track_id == -1:
                self.tracks.append(attach_)
            else:
                self.tracks[match_track_id] = np.concatenate(
                    [self.tracks[match_track_id], attach_], axis=0
                )

    def run_detector(self, rgb, frame_id, T_wc):
        img_tensor, _ = self.transforms(rgb, None)
        img_tensor = img_tensor.unsqueeze(0).to("cuda")
        predictions = self.detector(img_tensor)
        out_objects = self.detector.postprocess(
            predictions, rgb.size,
            float(self.detect_threshold), self.sequence_meta.K
        )
        n_predictions = len(out_objects['bboxes'][0])

        detections = []
        out_objects['angles'] = np.asarray(out_objects['angles'])  / 180. * np.pi
        for i in range(n_predictions):
            sin_azi = np.sin(out_objects["angles"][0][i])
            cos_azi = np.cos(out_objects["angles"][0][i])
            out_objects['bboxes'][0][i][:, 0] /= self.sequence_meta.img_w
            out_objects['bboxes'][0][i][:, 1] /= self.sequence_meta.img_h
            obj = [
                frame_id, out_objects["classes"][0][i],
                out_objects["bboxes"][0][i][0, 0], out_objects["bboxes"][0][i][0, 1],
                out_objects["bboxes"][0][i][1, 0], out_objects["bboxes"][0][i][1, 1],
                out_objects["dimensions"][0][i][0], out_objects["dimensions"][0][i][1],
                out_objects["dimensions"][0][i][2], out_objects["translates"][0][i][0],
                out_objects["translates"][0][i][1], out_objects["translates"][0][i][2],
                sin_azi, cos_azi, out_objects["scores"][0][i]
            ]  # the last one is for global instance id, -1 means not matched
            obj = [float(f) for f in obj]
            lc = np.zeros(64) - 1
            obj += lc.tolist()
            detections.append(obj)
        return detections

    def init_sequence(self, intrinsics, img_h ,img_w):
        """ prepare sequence data
        """

        self.run_associator = False
        self.sequence_meta = EasyDict({
            "K": intrinsics,
            "img_h": img_h,
            "img_w": img_w
        })
        self.tracks = []
        self.T_wcs = []
        self.P_cws = []
        self.usable_frames = []

    def process_frame(self, rgb, frame_id, T_wc):
        """ run detector and associator given a new frame
        """
        self.usable_frames.append(frame_id)
        self.T_wcs.append(T_wc)
        self.P_cws.append(self.sequence_meta.K @ np.linalg.inv(T_wc)[:3, :])

        detections = self.run_detector(rgb, frame_id, T_wc)
        # no initialized tracks and new detections
        if len(detections) == 0:
            return None
        detections = np.asarray(detections)
        if len(detections) > 30:
            detections = detections[:30, :] 
        if not self.run_associator:
            self.run_associator = True
            self.tracks = self._init_tracks(detections, T_wc)
            return None

        # convert to tensor and transform to camera coordinate
        track_tensors = self._prepare_tracks(T_wc)
        input_associator = {
            "detections": torch.from_numpy(detections).float().to("cuda").T,
            "tracks": track_tensors,
            "pose": T_wc,
        }
        # attach matching result to tracks
        data = self._collater([input_associator])
        n_tracks, n_detections = data['valid_list'][0]

        with torch.no_grad():
            predictions = self.associator(data, self.match_threshold, eval_only=True)
        pred_matches = predictions['matches'][0].astype(np.int32)
        score_mat = predictions['pred'][0][0].cpu().exp().numpy()
        detections = data['detections'][0, :, :n_detections].numpy()
        # [n_features, n_dets] -> [n_dets, n_featurese]
        detections = detections.T
        self._attach_to_tracks(
            pred_matches, detections, T_wc, score_mat
        )

    def merge_process(self, data):
        self.logger.info("Merging tracks")
        tracks = merge_process(data, self.usable_frames)
        return tracks

    def optim_process(self, tracks):
        num_opt = 200
        min_views = 10
        optim_out = optim_process(
            tracks,
            self.usable_frames,
            self.T_wcs,
            self.P_cws,
            self.sequence_meta.img_h,
            self.sequence_meta.img_w,
            self.sequence_meta.K,
            self.representation,
            prior=True,
            n_iters=num_opt,
            n_views=min_views
        )
        return optim_out