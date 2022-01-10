# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

''' Ref: https://github.com/ScanNet/ScanNet/blob/master/BenchmarkScripts '''
import os
import sys
import json
import csv
import quaternion

import src.utils.geometry_utils as geo_utils

try:
    import numpy as np
except:
    print("Failed to import numpy package.")
    sys.exit(-1)

try:
    from plyfile import PlyData, PlyElement
except:
    print("Please install the module 'plyfile' for PLY i/o, e.g.")
    print("pip install plyfile")
    sys.exit(-1)

OBJ_CLASS_IDS = np.array([3,4,5,6,7,8,9,10,11,12,14,16,24,28,33,34,36,39])
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


def flip_axis(pc):#, trans_mat):
    """ convert votenet GT and prediction to original scannet
    """

    pc2 = np.copy(pc)
    pc2[...,[0,1,2]] = pc2[...,[0,2,1]] # cam X,-Z,Y = depth X,Y,Z
    pc2[...,2] *= -1
    # pc2 = (geo_utils.get_homogeneous(pc2) @ trans_mat.T)[: ,:3]

    return pc2


def represents_int(s):
    ''' if string s represents an int. '''
    try: 
        int(s)
        return True
    except ValueError:
        return False


def read_meta_file(meta_file):
    lines = open(meta_file).readlines()
    for line in lines:
        if 'axisAlignment' in line:
            axis_align_matrix = [float(x) \
                for x in line.rstrip().strip('axisAlignment = ').split(' ')]
            break
    axis_align_matrix = np.array(axis_align_matrix).reshape((4,4))
    return axis_align_matrix


def read_label_mapping(filename, label_from='raw_category', label_to='nyu40id'):
    assert os.path.isfile(filename)
    mapping = dict()
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            if len(row[label_to]) == 0:
                continue
            if represents_int(row[label_to]):
                mapping[row[label_from]] = int(row[label_to])
            else:
                mapping[row[label_from]] = row[label_to]
    if represents_int(list(mapping.keys())[0]):
        mapping = {int(k):v for k,v in mapping.items()}
    return mapping


def read_mesh_vertices(filename):
    """ read XYZ for each vertex.
    """
    assert os.path.isfile(filename)
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 3], dtype=np.float32)
        vertices[:,0] = plydata['vertex'].data['x']
        vertices[:,1] = plydata['vertex'].data['y']
        vertices[:,2] = plydata['vertex'].data['z']
    return vertices


def read_mesh_vertices_rgb(filename):
    """ read XYZ RGB for each vertex.
    Note: RGB values are in 0-255
    """
    assert os.path.isfile(filename)
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 6], dtype=np.float32)
        vertices[:,0] = plydata['vertex'].data['x']
        vertices[:,1] = plydata['vertex'].data['y']
        vertices[:,2] = plydata['vertex'].data['z']
        vertices[:,3] = plydata['vertex'].data['red']
        vertices[:,4] = plydata['vertex'].data['green']
        vertices[:,5] = plydata['vertex'].data['blue']
    return vertices


def read_intrinsic(file_path):
    with open(file_path, "r") as f:
        intrinsic = np.asarray(
            [list(map(lambda x: float(x), f.split())) for f in f.read().splitlines()]
        )
    return intrinsic


def read_extrinsic(file_path):
    with open(file_path, "r") as f:
        T_cam_scan = np.linalg.inv(
            np.asarray(
                [list(map(lambda x: float(x), f.split())) for f in f.read().splitlines()]
            )
        )
    return T_cam_scan


def read_aggregation(filename):
    assert os.path.isfile(filename)
    object_id_to_segs = {}
    label_to_segs = {}
    with open(filename) as f:
        data = json.load(f)
        num_objects = len(data['segGroups'])
        for i in range(num_objects):
            object_id = data['segGroups'][i]['objectId'] + 1 # instance ids should be 1-indexed
            label = data['segGroups'][i]['label']
            segs = data['segGroups'][i]['segments']
            object_id_to_segs[object_id] = segs
            if label in label_to_segs:
                label_to_segs[label].extend(segs)
            else:
                label_to_segs[label] = segs
    return object_id_to_segs, label_to_segs


def read_segmentation(filename):
    assert os.path.isfile(filename)
    seg_to_verts = {}
    with open(filename) as f:
        data = json.load(f)
        num_verts = len(data['segIndices'])
        for i in range(num_verts):
            seg_id = data['segIndices'][i]
            if seg_id in seg_to_verts:
                seg_to_verts[seg_id].append(i)
            else:
                seg_to_verts[seg_id] = [i]
    return seg_to_verts, num_verts


def read_instance_vertices(seg_file, agg_file, label_map):
    object_id_to_segs, label_to_segs = read_aggregation(agg_file)
    seg_to_verts, num_verts = read_segmentation(seg_file)
    label_ids = np.zeros(shape=(num_verts), dtype=np.uint32) # 0: unannotated
    for label, segs in label_to_segs.items():
        label_id = label_map[label]
        for seg in segs:
            verts = seg_to_verts[seg]
            label_ids[verts] = label_id
    instance_ids = np.zeros(shape=(num_verts), dtype=np.uint32) # 0: unannotated
    for object_id, segs in object_id_to_segs.items():
        for seg in segs:
            verts = seg_to_verts[seg]
            instance_ids[verts] = object_id
    return instance_ids


def read_gt_annotations(gt_path):
    with open(gt_path, "r") as f:
        gt_annos = json.load(f)
    for gt in gt_annos:
        gt[1] = flip_axis(np.asarray(gt[1]))
        gt[1] = gt[1][[4, 5, 6, 7, 0, 1, 2, 3], :]

        if gt[0] in [1, 2, 3, 4, 10]:
            gt[1][4: 7, 2] = 0
    return gt_annos


def get_cam_azi(T_wc):
    """ get camera azimuth rotation. Assume z axis is upright
    """
    cam_orientation = np.array([[0, 0, 1], [0, 0, 0]])
    cam_orientation = (geo_utils.get_homogeneous(cam_orientation) @ T_wc.T)[:, :3]
    cam_orientation = cam_orientation[0] - cam_orientation[1]
    cam_orientation[2] = 0
    cam_orientation = cam_orientation / np.linalg.norm(cam_orientation)
    theta = np.arctan2(cam_orientation[1], cam_orientation[0])
    return theta


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