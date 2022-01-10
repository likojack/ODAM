import numpy as np


def quadric_2vect(Q):
    """ Vectorize the quadric matrix
    """

    Q_vec = np.array([
        Q[0, 0], Q[0, 1], Q[0, 2], Q[0, 3],
        Q[1, 1], Q[1, 2], Q[1, 3],
        Q[2, 2], Q[2, 3],
        Q[3, 3]])
    return Q_vec


def quadric_2mat(q_vec):
    """ Vectorize the quadric matrix
    """
    Q = np.zeros((4, 4))
    Q[0, 0] = q_vec[0]
    Q[0, 1] = q_vec[1]
    Q[1, 0] = q_vec[1]
    Q[0, 2] = q_vec[2]
    Q[2, 0] = q_vec[2]
    Q[0, 3] = q_vec[3]
    Q[3, 0] = q_vec[3]
    Q[1, 1] = q_vec[4]
    Q[1, 2] = q_vec[5]
    Q[2, 1] = q_vec[5]
    Q[1, 3] = q_vec[6]
    Q[3, 1] = q_vec[6]
    Q[2, 2] = q_vec[7]
    Q[2, 3] = q_vec[8]
    Q[3, 2] = q_vec[8]
    Q[3, 3] = q_vec[9]
    return Q


def plane_2vect(plane):
    """ transform line rep. for quadric vectorization
    """

    plane_vec = np.array([
        plane[0]**2, 2*plane[0]*plane[1], 2*plane[0]*plane[2], 2*plane[0]*plane[3],
        plane[1]**2, 2*plane[1]*plane[2], 2*plane[1]*plane[3],
        plane[2]**2, 2*plane[2]*plane[3],
        plane[3]**2])
    return plane_vec


def plane_2form(plane):
    """ from vectorization for quadric to 4-paramter form
    """

    plane_vec = np.array([
        np.sqrt(plane[0]), np.sqrt(plane[4]),
        np.sqrt(plane[7]), np.sqrt(plane[9])])
    return plane_vec


def normalize_plane(plane):
    """normalize plane"""

    norm = np.linalg.norm(plane[0, :3])
    plane = plane / norm
    return plane


def bbox_to_lines(bbox, img_size=None, edge_threshold=None):
    """ convert bounding box to lines
    
    Args:
        bbox (np.array) [2, 2]: 2D bounding box, [[x_min, y_min], [x_max ,y_max]]
        img_size (tuple): (img_h, img_w)
        edge_threshold (None or int): whether throw away lines that are close
            to the image boundary. If not None, edge_threshold indicates the
            threshold to edge.
    """

    if edge_threshold is not None:
        assert img_size, "img_size needs to be given if edge_threshold is not None"


    x_min, y_min = bbox[0]
    x_max, y_max = bbox[1]

    threshold_h_min = edge_threshold
    threshold_h_max = img_size[0] - edge_threshold
    threshold_w_min = edge_threshold
    threshold_w_max = img_size[1] - edge_threshold

    x_thresholds = (threshold_w_min, threshold_w_max)
    y_thresholds = (threshold_h_min, threshold_h_max)

    lines = {}
    for direction, value, thresholds in zip(
        ["x_min", "y_min", "x_max", "y_max"],
        [x_min, y_min, x_max, y_max],
        [x_thresholds, y_thresholds, x_thresholds, y_thresholds]
    ):
        t_min, t_max = thresholds
        if value > t_min and value < t_max:
            if direction in ['x_min', 'x_max']:
                line = np.array([1, 0, -value])
            else:  # "y" in direction
                line = np.array([0, 1, -value])
            lines[direction] = line

    return lines


def bbox2aabb(bbox):
    aabb = np.zeros((2, 3))
    aabb[0, 0] = np.min(bbox[:, 0])
    aabb[0, 1] = np.min(bbox[:, 1])
    aabb[0, 2] = np.min(bbox[:, 2])
    aabb[1, 0] = np.max(bbox[:, 0])
    aabb[1, 1] = np.max(bbox[:, 1])
    aabb[1, 2] = np.max(bbox[:, 2])
    return aabb


def compute_plane_vector(plane_pts):
    """ compute plane normal and offset from 3 points on the plane
    """

    v1 = plane_pts[0] - plane_pts[1]
    v2 = plane_pts[0] - plane_pts[2]
    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)
    d = np.dot(plane_pts[0], normal)
    plane = np.array([normal[0], normal[1], normal[2], -d])
    return plane


def get_planes_from_3d_aabb(bbox_pred):
    """
    bbox_pred is 8 * 3 matrix
    [x_max, y_max ,z_max]
    [x_max, y_min ,z_max]
    [x_min, y_min ,z_max]
    [x_min, y_max ,z_max]
    [x_max, y_max ,z_min]
    [x_max, y_min ,z_min]
    [x_min, y_min ,z_min]
    [x_min, y_max ,z_min]
    """

    plane0 = np.array([
        bbox_pred[0],
        bbox_pred[1],
        bbox_pred[4],
    ])
    plane1 = np.array([
        bbox_pred[1],
        bbox_pred[2],
        bbox_pred[6],
    ])
    plane2 = np.array([
        bbox_pred[2],
        bbox_pred[3],
        bbox_pred[6],
    ])
    plane3 = np.array([
        bbox_pred[0],
        bbox_pred[3],
        bbox_pred[7],
    ])
    plane4 = np.array([
        bbox_pred[4],
        bbox_pred[5],
        bbox_pred[6],
    ])
    plane5 = np.array([
        bbox_pred[0],
        bbox_pred[1],
        bbox_pred[2],
    ])
    plane_vecs = []
    planes = []
    for plane_pts in [plane0, plane1, plane2, plane3, plane4, plane5]:
        plane = compute_plane_vector(plane_pts)
        planes.append(plane)
        plane_vec = plane_2vect(plane)
        plane_vecs.append(plane_vec)
    return plane_vecs, planes

