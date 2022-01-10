""" use o3d to visualize results on a scene
"""

import argparse
import numpy as np
import open3d as o3d
import os
import pickle
import sys
import trimesh
from PIL import ImageColor

import src.datasets.scannet_utils as scannet_utils

import src.utils.o3d_helper as o3d_helper
from src.utils.o3d_helper import STANDARD_COLORS


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--path")
    args_parser.add_argument("--min_views", type=int, default=10)
    args = args_parser.parse_args()

    with open(args.path, "rb") as f:
        tracks = pickle.load(f)

    dataset_root = "./data/ScanNet/scans/"
    seq = args.path.split("/")[-1]    
    meta_path = os.path.join(dataset_root, seq, "{}.txt".format(seq))
    mesh_path = os.path.join(
        dataset_root, "{0}/{0}_vh_clean_2.ply".format(seq))
    axis_align_mat = scannet_utils.read_meta_file(meta_path)
    scene_mesh = o3d_helper.load_scene_mesh(mesh_path, trans_mat=axis_align_mat)

    # visual_list = [scene_mesh]
    visual_list = []
    for i, quadric in enumerate(tracks["quadrics"]):
        if len(tracks["tracks"][i]) < args.min_views:
            continue
        # print(tracks['tracks'][i][0][1])
        print(len(tracks['tracks'][i]))
        surface_points, _ = quadric.compute_ellipsoid_points(use_numpy=True)
        m = trimesh.Trimesh(vertices=surface_points).convex_hull
        faces = m.faces
        pts = m.vertices
        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(pts)
        mesh_o3d.triangles = o3d.utility.Vector3iVector(faces)
        mesh_o3d.compute_vertex_normals()
        rgb = ImageColor.getrgb(STANDARD_COLORS[i%len(STANDARD_COLORS)])
        rgb = np.asarray(rgb) / 255.
        mesh_o3d.paint_uniform_color(rgb)
        visual_list.append(mesh_o3d)
        # bbx = o3d_helper.linemesh_from_pc(tracks['bboxes_qc'][i], colors=rgb[None, :])
        # visual_list.extend(bbx)
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.6, origin=[-0, -0, -0])
    visual_list.append(mesh_frame)
    o3d.visualization.draw_geometries(visual_list)
