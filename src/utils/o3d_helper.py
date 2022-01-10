import numpy as np
import open3d as o3d
import trimesh

from src.utils.geometry_utils import get_homogeneous


STANDARD_COLORS = [
    "Lime", "Blue", 'Green', "Red", 'Yellow', 'Pink', 'IndianRed', 'Purple', 'Peru',
    'Pink', 'Orange', 'OrangeRed', 'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki',
    'DarkOrange', 'Gold', 'GoldenRod']


def align_vector_to_another(a=np.array([0, 0, 1]), b=np.array([1, 0, 0])):
    """
    Aligns vector a to vector b with axis angle rotation
    """
    if np.array_equal(a, b):
        return None, None
    if np.sum(b + a) == 0:  # if b is possite to a
        b += 1e-3
    axis_ = np.cross(a, b)
    axis_ = axis_ / (np.linalg.norm(axis_))
    angle = np.arccos(np.dot(a, b))
    return axis_, angle


def normalized(a, axis=-1, order=2):
    """Normalizes a numpy array of points"""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis), l2


class LineMesh(object):
    def __init__(self, points, lines=None, colors=[0, 1, 0], radius=0.15):
        """Creates a line represented as sequence of cylinder triangular meshes

        Arguments:
            points {ndarray} -- Numpy array of ponts Nx3.

        Keyword Arguments:
            lines {list[list] or None} -- List of point index pairs denoting line segments. If None, implicit lines from ordered pairwise points. (default: {None})
            colors {list} -- list of colors, or single color of the line (default: {[0, 1, 0]})
            radius {float} -- radius of cylinder (default: {0.15})
        """
        self.points = np.array(points)
        self.lines = np.array(
            lines) if lines is not None else self.lines_from_ordered_points(self.points)
        self.colors = np.array(colors)
        self.radius = radius
        self.cylinder_segments = []

        self.create_line_mesh()

    @staticmethod
    def lines_from_ordered_points(points):
        lines = [[i, i + 1] for i in range(0, points.shape[0] - 1, 1)]
        return np.array(lines)

    def create_line_mesh(self):
        first_points = self.points[self.lines[:, 0], :]
        second_points = self.points[self.lines[:, 1], :]
        line_segments = second_points - first_points
        line_segments_unit, line_lengths = normalized(line_segments)

        z_axis = np.array([0, 0, 1])
        # Create triangular mesh cylinder segments of line
        for i in range(line_segments_unit.shape[0]):
            line_segment = line_segments_unit[i, :]
            line_length = line_lengths[i]
            # get axis angle rotation to allign cylinder with line segment
            axis, angle = align_vector_to_another(z_axis, line_segment)
            # Get translation vector
            translation = first_points[i, :] + line_segment * line_length * 0.5
            # create cylinder and apply transformations
            cylinder_segment = o3d.geometry.TriangleMesh.create_cylinder(
                self.radius, line_length)
            cylinder_segment = cylinder_segment.translate(
                translation, relative=False)
            if axis is not None:
                axis_a = axis * angle
                cylinder_segment = cylinder_segment.rotate(
                    R=o3d.geometry.get_rotation_matrix_from_axis_angle(axis_a), center=True)
                # cylinder_segment = cylinder_segment.rotate(
                #   axis_a, center=True, type=o3d.geometry.RotationType.AxisAngle)
            # color cylinder
            color = self.colors if self.colors.ndim == 1 else self.colors[i, :]
            cylinder_segment.paint_uniform_color(color)

            self.cylinder_segments.append(cylinder_segment)

    def add_line(self, vis):
        """Adds this line to the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.add_geometry(cylinder)

    def remove_line(self, vis):
        """Removes this line from the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.remove_geometry(cylinder)


def lineset_from_pc(point_cloud, colors, orders=None):
    """ open3d lineset from numpy point cloud

    Args:
        point_cloud ([N, 3] np.ndarray): corner points of a 3D bounding box
        colors ([1, 3] np.ndarray): color of the lineset
        orders (): reorder the point cloud to build a valid 3D bbox

    Returns:
        line_set (open3d.geometry.Lineset)
    """
    # vertex order is consistent with get_corner_pts() in Object class
    if orders is None:
        lines = [
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
    else:
        lines = orders
    colors_tmp = np.zeros((len(lines), 3))
    colors_tmp += colors
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(point_cloud),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors_tmp)
    return line_set


def linemesh_from_pc(point_cloud, colors, orders=None):
    if orders is None:
        lines = [
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
    else:
        lines = orders

    colors_tmp = np.zeros((len(lines), 3))
    colors_tmp += colors

    line_mesh = LineMesh(point_cloud, lines, colors_tmp, radius=0.02)
    return line_mesh.cylinder_segments


def load_scene_mesh(path, trans_mat=None, open_3d=True):
    scene_mesh = trimesh.load(path)
    if trans_mat is not None:
        scene_mesh.vertices = np.dot(get_homogeneous(scene_mesh.vertices), trans_mat.T)[:, :3]
    if open_3d:
        scene_mesh_o3d = trimesh2o3d(scene_mesh)
        return scene_mesh_o3d
    else:
        return scene_mesh


def trimesh2o3d(mesh):
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    mesh_o3d.triangles = o3d.utility.Vector3iVector(mesh.faces)
    mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(mesh.visual.vertex_colors[:, :3] / 255.)
    return mesh_o3d


def np2pc(points, colors=None):
    """ convert numpy colors point cloud to o3d point cloud
    
    Args:
        points (np.ndarray): [n_pts, 3]
        colors (np.ndarray): [n_pts, 3]
    Return:
        pts_o3d (o3d.geometry.PointCloud)
    """
    pts_o3d = o3d.geometry.PointCloud()
    pts_o3d.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pts_o3d.colors = o3d.utility.Vector3dVector(colors)
    return pts_o3d