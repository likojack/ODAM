""" use o3d to visualize results on a scene
"""

import argparse
import json
import numpy as np
import open3d as o3d
import os
import OpenGL.GL as gl
import pickle
import sys
from scipy import interpolate
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import trimesh
from PIL import ImageColor
import pangolin

import src.datasets.scannet_utils as scannet_utils
import src.utils.visual_utils as visual_utils
import src.utils.geometry_utils as geo_utils
import src.utils.tracking_gt_utils as tracking_gt_utils

sys.path.append("/home/kejie/repository/DOM")
import dom.libs.o3d_helper as o3d_helper
import dom.player.utils as player_utils
from dom.libs.drawing import STANDARD_COLORS


SCAN2CAD_MAPPER = {
    "03211117": 0,  #"display",
    "04379243": 1,  #"table",
    "02808440": 2,  #"bathtub",
    "02747177": 3,  #"trashbin",
    "04256520": 4,  #"sofa",
    "03001627": 5,  #"chair",
    "02933112": 6,  #"cabinet",
    "02871439": 7,  #"bookshelf",
}


def parse_scan2cad_annotations(annotations, T_align=None):
    annotations_ready = []
    gt_classes = []
    T_ws = scannet_utils.make_M_from_tqs(
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
        T_wo = scannet_utils.make_M_from_tqs(t, q, np.ones_like(s))
        T_wo = T_sw @ T_wo
        bbox = geo_utils.get_corner_by_dims(scales_gt)
        bbox = (geo_utils.get_homogeneous(bbox) @ T_wo.T)[:, :3]
        if T_align is not None:
            bbox = (geo_utils.get_homogeneous(bbox) @ T_align.T)[:, :3]
        if cat_gt not in SCAN2CAD_MAPPER:
            continue
        annotations_ready.append(tuple([SCAN2CAD_MAPPER[cat_gt], bbox]))
    return annotations_ready


def get_snap_poses(poses, scam):
    poses.append(scam.GetModelViewMatrix().m)
    return poses


def interpolate_poses(poses, n_steps):
    out_poses = np.tile(np.eye(4)[None, :, :], (n_steps, 1, 1))
    
    translates = [p[:3, 3] for p in poses]
    translates = np.asarray(translates).T
    interpolated_positions = []
    for axis in translates:
        x = np.arange(0, len(axis)) / len(axis)
        y = axis
        f = interpolate.interp1d(x, y)
        x_new = np.arange(0, x[-1], x[-1]/n_steps)
        y_new = f(x_new)   # use interpolation function returned by `interp1d`
        interpolated_positions.append(y_new)
    interpolated_positions = np.asarray(interpolated_positions).T
    out_poses[:, :3, 3] = interpolated_positions
    
    rotations = [p[:3, :3] for p in poses]
    key_times = np.arange(len(rotations))
    slerp = Slerp(key_times, R.from_dcm(rotations))
    times = np.arange(0, key_times[-1], key_times[-1] / n_steps)
    interp_rots = slerp(times)
    interp_rots = interp_rots.as_dcm()
    out_poses[:, :3, :3] = interp_rots
    return out_poses


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--path")
    args_parser.add_argument("--min_views", type=int, default=12)
    args_parser.add_argument("--input_pose")
    args_parser.add_argument("--scene_only", action="store_true")
    args = args_parser.parse_args()

    with open(args.path, "rb") as f:
        tracks = pickle.load(f)

    dataset_root = "./data/ScanNet/scans/"
    seq = args.path.split("/")[-1]    
    meta_path = os.path.join(dataset_root, seq, "{}.txt".format(seq))
    mesh_path = os.path.join(
        dataset_root, "{0}/{0}_vh_clean_2.ply".format(seq))
    axis_align_mat = scannet_utils.read_meta_file(meta_path)
    scene_mesh = o3d_helper.load_scene_mesh(mesh_path, trans_mat=axis_align_mat, open_3d=False)
    intrinsic_path = os.path.join(dataset_root, seq, "frames/intrinsic/intrinsic_color.txt")
    extrinsic_dir = os.path.join(dataset_root, seq, "frames/pose")
    K = scannet_utils.read_intrinsic(intrinsic_path)[:3, :3]

    scan2cad_path = "/home/kejie/ext_disk/datasets/Scan2CAD/full_annotations.json"
    with open(scan2cad_path, "r") as f:
        scan2cad = json.load(f)
    for scan in scan2cad:
        if scan['id_scan'] == seq:
            gt_annotations = parse_scan2cad_annotations(scan, axis_align_mat)
            break

    with open(
        ("/home/kejie/repository/imovotenet/data/ScanNet/imovotenet_scan2cad/"
         "val_tracking_ls"), "rb"
    ) as f:
        dataset = pickle.load(f)
    img_names = np.asarray(dataset['files'][seq]['img_names'])
    img_names = img_names.astype(np.int32)
    T_wcs, P_cws = tracking_gt_utils.load_poses(extrinsic_dir, img_names, axis_align_mat, K)
    
    width, height = 400, 300
    scam, dcam, dimg, dpred, dtrack, texture = player_utils.init_panal(
        K[0,0], width, height)
    # define the interface
    panel = pangolin.CreatePanel('ui')
    # panel.SetBounds(height / 768, 1.0, 0.0, 175 / 1024.)
    panel.SetBounds(0 / 768, 0.0, 0.0, 0 / 1024.)
    
    snap_pose = pangolin.VarBool('ui.Snap_Pose', value=False, toggle=False)
    run_pose = pangolin.VarBool('ui.Run_Pose', value=True, toggle=True)
    save_pose = pangolin.VarBool('ui.save_pose', value=False, toggle=True)
    save_window = pangolin.VarBool('ui.Save_Window', value=True, toggle=True)
    pango_alpha = pangolin.VarFloat('ui.alpha', value=0.66, min=0, max=0.99)

    latest_idx = -1
    snap_poses = []
    interpolation = {}
    interpolation['snap_poses'] = snap_poses
    interpolation['interpolated_poses'] = None
    interpolation['steps'] = 200
    interpolation['cur_ptr'] = 0

    while not pangolin.ShouldQuit():
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(0.95, 0.95, 0.95, 1.0)
        dcam.Activate(scam)
        scene_colors = scene_mesh.visual.vertex_colors[:, :3] / 255.
        if args.scene_only:
            player_utils.draw_mesh(
                dcam, scam,
                scene_mesh.vertices, scene_mesh.faces,
                scene_colors,
                alpha_value=1 if args.scene_only else 0.6
            )
        else:
            player_utils.draw_3d_points(
                dcam, scam, scene_mesh.vertices, scene_colors,
                pt_size=1
            )

        if pangolin.Pushed(snap_pose):
            snap_poses = get_snap_poses(snap_poses, scam)



        if args.scene_only:
            pass
            # for gt_id, gt_anno in enumerate(gt_annotations):
            #     gt_class, gt_bbox = gt_anno
            #     # if gt_class != 0 :
            #     #     continue
            #     _rgb = ImageColor.getrgb(STANDARD_COLORS[gt_id%len(STANDARD_COLORS)])
            #     _rgb = np.asarray(_rgb) / 255.
            #     bbox_lines = visual_utils.bbox_lineset(gt_bbox)
            #     player_utils.draw_lines(
            #         dcam, scam, bbox_lines, _rgb, line_width=3)
        else:
            for i, quadric in enumerate(tracks["quadrics"]):
                track = tracks["tracks"][i]
                if len(tracks["tracks"][i]) < args.min_views:
                    continue
                # if track[0, 1] != 0:
                #     continue
                obj_class = tracks["tracks"][i][0, 1]
                surface_points, _ = quadric.compute_ellipsoid_points(use_numpy=True)
                m = trimesh.Trimesh(vertices=surface_points).convex_hull
                faces = m.faces
                pts = m.vertices
                normals = m.vertex_normals
                rgb = ImageColor.getrgb(STANDARD_COLORS[i%len(STANDARD_COLORS)])
                rgb = np.asarray(rgb) / 255.

                color = np.zeros((len(normals), 4))
                color[:, :3] = normals#rgb[None, :]
                color[:, 3] = float(pango_alpha)
                player_utils.draw_mesh(dcam, scam, pts, faces, color)
                bbox_lines = visual_utils.bbox_lineset(tracks['bboxes_qc'][i])
                player_utils.draw_lines(
                    dcam, scam, bbox_lines, rgb, line_width=3)
                
                # for img_id, img_name in enumerate(img_names):
                #     if int(img_name) in track[:, 0]:
                #         t_wo = track[track[:, 0] == int(img_name), 9: 12]
                #         assert len(t_wo) == 1
                #         t_wo = t_wo[0]
                #         t_wc = T_wcs[img_id][:3, 3]
                #         line = np.stack([t_wo, t_wc], axis=0)
                #         line_color = np.concatenate([rgb, np.array([1])], axis=0)
                #         player_utils.draw_line(dcam, scam, line, line_color, line_width=2)
                

        if run_pose.Get():
            if interpolation['interpolated_poses'] is None:
                if args.input_pose is not None:
                    poses = np.load(args.input_pose)
                else:
                    poses = interpolate_poses(snap_poses, interpolation['steps'])
                    if save_pose.Get():
                        np.save("./cam_poses_1.npy", poses)
                interpolation['interpolated_poses'] = poses
            if interpolation['cur_ptr'] < interpolation['steps']:
                T_view_model = poses[interpolation['cur_ptr']]
                pose = pangolin.OpenGlMatrix()
                pose.m = T_view_model
                scam.SetModelViewMatrix(pose)
                interpolation['cur_ptr'] += 1
            else:
                break
            if save_window.Get():
                pangolin.SaveWindowOnRender("{:06d}".format(int(latest_idx)))
                latest_idx += 1

        pangolin.FinishFrame()
    
    if args.scene_only:
        out_dir = f"./scene_videos/{seq}"
    else:
        out_dir = f"./result_videos/{seq}"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    os.system(f"mv 00* {out_dir}")