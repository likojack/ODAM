import numpy as np
import scipy
import warnings
import torch
import torch.nn.functional as F
import pickle

import src.utils.box_utils as box_utils
from learnable_primitives.sampling import EqualDistanceSamplerSQ, sample_points_on_surface
import src.utils.geometry_utils as geo_utils


CLASS_MAPPER = {
    0: "03211117",   #"display",
    1: "04379243",   #"table",
    2: "02808440",   #"bathtub",
    3: "02747177",   #"trashbin",
    4: "04256520",   #"sofa",
    5: "03001627",   #"chair",
    6: "02933112",   #"cabinet",
    7: "02871439",   #"bookshelf",
}



def squashing(shape, min_=0.2, max_=1.6):
    return torch.sigmoid(shape) * (max_ - min_) + min_


def compute_quadric_svd(plane_vecs):
    Sigma = np.concatenate(plane_vecs, axis=0)
    A = Sigma.T @ Sigma
    d, V = np.linalg.eig(A)
    indices = np.argsort(d)
    Q_est_vec = V[:, indices[0]]
    return Q_est_vec


class QuadricOptimizer():

    def __init__(self, translate, quat, scale):
        """
        only need the first three coefficient for quaternion. 
        Fourth is just normalization
        """

        self.translate = torch.tensor(
            translate, dtype=torch.float32,
            requires_grad=True)
        self.quat = torch.tensor(
            quat, dtype=torch.float32,
            requires_grad=True)
        self.scale_factor = torch.tensor(
            1., dtype=torch.float32,
            requires_grad=True)
        self.scale = torch.tensor(
            scale/2, dtype=torch.float32,
            requires_grad=False)  # input scale is length of 3D bbox
        self.Q_init = self.params2mat(self.translate, self.quat, (self.scale_factor * self.scale) ** 2)
        self.Q_init = DualQuadric(self.Q_init.detach().numpy())

        self.optimizer = torch.optim.Adam(
            [self.translate, self.quat, self.scale_factor],
            lr=0.01
        )
        self.loss_log = []

    def params2mat(self, translate, quat, scale):
        zeros = torch.zeros(3).float()
        Q_o = torch.diag(torch.cat([scale, torch.tensor([-1.])], dim=0))
        T_wo = torch.eye(4)
        # R_wo = self.quat2mat(torch.unsqueeze(quat, 0))  # [3] -> [1, 3]
        # R_wo = R_wo[0, :, :]  # [1, 3, 3] -> [3, 3]
        R_wo = self.rotz(quat)
        T_wo = torch.cat([R_wo, translate[None, :].T], dim=1)
        T_wo = torch.cat([T_wo, torch.tensor([[0., 0., 0., 1.]])], dim=0)

        return T_wo @ Q_o @ T_wo.T

    def quat2mat(self, quat):
        """Convert quaternion coefficients to rotation matrix.
        Args:
            quat: first three coeff of quaternion of rotation. fourht is then computed to have a norm of 1 -- size = [B, 3]
        Returns:
            Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
        """
        norm_quat = torch.cat([quat[:,:1].detach()*0 + 1, quat], dim=1)
        norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
        w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

        B = quat.size(0)

        w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
        wx, wy, wz = w*x, w*y, w*z
        xy, xz, yz = x*y, x*z, y*z

        rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                            2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                            2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
        return rotMat

    def rotz(self, angle):
        """Convert euler angles to rotation matrix.
        Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
        Args:
            angle: rotation angle along 3 axis (in radians) -- size = [B, 3]
        Returns:
            Rotation matrix corresponding to the euler angles -- size = [B, 3, 3]
        """
        z = angle

        cosz = torch.cos(z)
        sinz = torch.sin(z)

        zeros = z.detach()*0
        ones = zeros.detach()+1
        zmat = torch.stack([cosz, -sinz, zeros,
                            sinz,  cosz, zeros,
                            zeros, zeros,  ones], dim=0).reshape(3, 3)

        return zmat

    def compute_projected_lines(self, C):
        """ 4 lines in this order: x_min, x_max, y_min, y_max
        """

        # extract bounding box of the dual conic
        b_x = torch.sqrt(4 * C[:, 0, 2] ** 2 - 4 * C[:, 0, 0] * C[:, 2, 2])
        assert not torch.isnan(b_x).any()
        x_0 = 0.5 / C[:, 2, 2] * (2 * C[:, 0, 2] + b_x)
        x_1 = 0.5 / C[:, 2, 2] * (2 * C[:, 0, 2] - b_x)
        x_min = torch.min(torch.stack([x_0, x_1], dim=0), dim=0).values
        x_max = torch.max(torch.stack([x_0, x_1], dim=0), dim=0).values

        b_y = torch.sqrt(4 * C[:, 1, 2] ** 2 - 4 * C[:, 1, 1] * C[:, 2, 2])
        assert not torch.isnan(b_y).any()
        y_0 = 0.5 / C[:, 2, 2] * (2 * C[:, 1, 2] + b_y)
        y_1 = 0.5 / C[:, 2, 2] * (2 * C[:, 1, 2] - b_y)
        y_min = torch.min(torch.stack([y_0, y_1], dim=0), dim=0).values
        y_max = torch.max(torch.stack([y_0, y_1], dim=0), dim=0).values
        out = {
            "x_min": -x_min,
            "y_min": -y_min,
            "x_max": -x_max,
            "y_max": -y_max
        }  # take negative for line equation
        return out

    def constraint_2d(
        self, Q_w, Ms, gt_lines_all_direction, gt_masks_all_direction, names):
        """ projection line constraint
        """

        n_frames = Ms.shape[0]
        Q_w = Q_w.repeat(n_frames, 1, 1)
        C = Ms @ Q_w @ Ms.permute(0, 2, 1)
        pred_lines = self.compute_projected_lines(C)
        loss_2d = 0
        n_loss = 0
        for name in names:
            losses = F.l1_loss(
                pred_lines[name], gt_lines_all_direction[name],
                reduction='none')
            losses = torch.where(
                torch.isnan(losses), torch.zeros_like(losses), losses)
            losses = losses * gt_masks_all_direction[name]
            loss_2d += torch.mean(losses)
        return loss_2d

    def constraint_3d(self, Q_w, planes):
        """ 3D plane constraint

        Args:
            planes (torch.tensor): [n_planes, 4], line in the form of 
                [normal (3), offset (1)]
        """

        n_planes = planes.shape[0]
        Q_w = Q_w.repeat(n_planes, 1, 1)

        normals = planes[:, :3].unsqueeze(-1)  # [n_planes, 3, 1]
        d_gt = planes[:, 3:].unsqueeze(-1)  # [n_planes, 1, 1]
        Q_upper =  Q_w[:, :3, :3] # upper left 3x3 of the quadric mat
        t = -Q_w[:, :3, 3:]  # [n_planes, 3, 1]
        B = torch.sqrt((2 * t.permute(0, 2, 1) @ normals)**2 + 4 * normals.permute(0, 2, 1) @ Q_upper @ normals)
        d1 = -(2 * t.permute(0, 2, 1) @ normals + B) / 2
        d2 = -(2 * t.permute(0, 2, 1) @ normals - B) / 2
        loss_1 = F.l1_loss(d_gt, d1, reduction='none').flatten()
        loss_2 = F.l1_loss(d_gt, d2, reduction='none').flatten()
        loss_3d = torch.min(torch.stack([loss_1, loss_2], dim=0), dim=0).values
        loss_3d = torch.mean(loss_3d)
        return loss_3d

    def run(self, gt_lines, Ms):
        """ run iterative optimization using geometric loss

        Args:
            gt_lines (list): a list of line constraints given by 2D
                bbox. Each element is dict indicating the line params for
                'x_min', 'x_max', 'y_min', y_max'.
            gt_planes (list): a list of 3D plane constraints from the
                3D bbox.
            Ms (list): a list of [3*4] np.ndarray for transform and projection.

        """

        if isinstance(Ms, np.ndarray):
            Ms = torch.tensor(Ms).float()
        # if isinstance(gt_planes, np.ndarray):
        #     gt_planes = torch.tensor(gt_planes).float()

        names = ['x_min', 'x_max', 'y_min', 'y_max']
        gt_masks_all_direction = {}
        gt_lines_all_direction = {}
        for name in names:
            gt_mask = torch.ones((len(gt_lines))).float()
            gt = torch.zeros((len(gt_lines))).float()
            for idx, gt_line in enumerate(gt_lines):
                if name not in gt_line:
                    gt_mask[idx] = 0
                    gt[idx] = 0
                else:
                    gt[idx] = gt_line[name][-1]
            gt_masks_all_direction[name] = gt_mask
            gt_lines_all_direction[name] = gt

        for i in range(500):
            self.optimizer.zero_grad()
            scale = (self.scale_factor * self.scale) ** 2
            Q_w = self.params2mat(self.translate, self.quat, scale)
            loss_2d = self.constraint_2d(
                Q_w, Ms, gt_lines_all_direction, gt_masks_all_direction, names)
            # loss_3d = self.constraint_3d(Q_w, gt_planes)
            loss = loss_2d #+ loss_3d * 0
            loss.backward()
            self.loss_log.append([loss_2d])
            self.optimizer.step()
            print(self.loss_log[-1])
        Q_w = self.params2mat(self.translate, self.quat, (self.scale_factor * self.scale) ** 2)
        Q_w = DualQuadric(Q_w.detach().numpy())
        return Q_w


class DualQuadric:
    def __init__(self, Q):
        self.Q = Q

    def projection(self, P, if_vectorize):
        """
        Args:
            P: transformation matrix
            if_vectorize: whether the P is in vectorization form
        """

        return P @ self.Q @ P.T

    def get_srt(self):
        """ get rotation, translation and scale from dual quadric matrix
        
        Returns:
            scale: diagonal elements of the scale matrix in dual form
        """

        t_wo = -self.Q[:3, 3:]  # [3, 1]
        t_t_transpose = t_wo @ t_wo.T  # [3, 3]
        
        # the matrix to be decomposed to get scale and R
        A = self.Q[:3, :3] + t_t_transpose
        scale, R_wo = scipy.linalg.eig(A)
        scale = scale.astype(np.float32)
        if np.linalg.det(R_wo) < 0:
            R_wo *= -1
        is_ellipsoid = True
        if (scale < 0).any():
            is_ellipsoid = False
            print("[warning]: quadric is not ellipsoid")
        for i in range(len(scale)):
            if scale[i] < 0:
                scale[i] *= -1
        return scale, R_wo, t_wo, is_ellipsoid
    
    def transform(self, T_cw):
        """ transform dual quadric from coordinate w to coordinate c
        """

        Q_c = T_cw @ self.Q @ T_cw.T
        return Q_c

    def get_bbox(self, P, if_vectorize, line_form):
        # project to image plane using projection matrix
        C_dual = self.projection(P, if_vectorize)

        # extract bounding box of the dual conic
        b_x = np.sqrt(4 * C_dual[0, 2] ** 2 - 4 * C_dual[0, 0] * C_dual[2, 2])
        x_0 = 0.5 / C_dual[2, 2] * (2 * C_dual[0, 2] + b_x)
        x_1 = 0.5 / C_dual[2, 2] * (2 * C_dual[0, 2] - b_x)
        x_min = min(x_0, x_1)
        x_max = max(x_0, x_1)

        b_y = np.sqrt(4 * C_dual[1, 2] ** 2 - 4 * C_dual[1, 1] * C_dual[2, 2])
        y_0 = 0.5 / C_dual[2, 2] * (2 * C_dual[1, 2] + b_y)
        y_1 = 0.5 / C_dual[2, 2] * (2 * C_dual[1, 2] - b_y)
        y_min = min(y_0, y_1)
        y_max = max(y_0, y_1)
        if line_form:
            lines = [
                np.array([1, 0, -x_min]),
                np.array([0, 1, -y_min]),
                np.array([1, 0, -x_max]),
                np.array([0, 1, -y_max]),
            ]
            return lines
        else:
            return np.array([x_min, y_min, x_max, y_max])

    def compute_ellipsoid_points(self, use_numpy=None):
        """Compute 3D points"""

        axes, R, centre, is_ellipsoid = self.get_srt()

        axes = np.sqrt(axes)

        centre = centre.flatten()

        size_side = 50  # Number of points for plotting one curve on the ellipsoid.

        # Compute the set of all spherical angles:
        u = np.linspace(0, 2 * np.pi, size_side)
        v = np.linspace(0, np.pi, size_side)

        # Compute the Cartesian coordinates of the surface points of the ellipsoid aligned with the axes and
        # centred at the origin:
        # (this is the equation of an ellipsoid):
        x = axes[0] * np.outer(np.cos(u), np.sin(v))
        y = axes[1] * np.outer(np.sin(u), np.sin(v))
        z = axes[2] * np.outer(np.ones_like(u), np.cos(v))

        # Rotate the points according to R.
        x, y, z = np.tensordot(R, np.vstack((x, y, z)).reshape((3, size_side, size_side)), axes=1)

        # Apply the translation.
        x = x + centre[0]
        y = y + centre[1]
        z = z + centre[2]
        n_points = x.shape[0] * x.shape[1]
        points = np.hstack((x.reshape(n_points, 1), y.reshape(n_points, 1), z.reshape(n_points, 1)))
        points = points.astype(np.float32)
        return points, is_ellipsoid


class SuperQuadricOptimizer():

    def __init__(self, translate, quat, scales, obj_class, representation, prior):
        """
        only need the first three coefficient for quaternion. 
        Fourth is just normalization
        """

        # input scale is the 3D dimensions of the bounding box
        # change it to sqrt of half of the dimension
        scales = np.sqrt(scales / 2)
        self.use_prior = prior
        assert representation in ["cube", "super_quadric", "quadric"]        
        if representation == "cube":
            self.Q_init = SuperQuadric(
                translate, quat, scales, shapes=np.array([-10000., -10000.]))
        else:
            self.Q_init = SuperQuadric(
                translate, quat, scales, shapes=np.array([-0., -0.]))
        
        self.Q_init.obj_class = obj_class

        if representation == "super_quadric":
            self.optimizer = torch.optim.Adam(
                [
                    {"params": [self.Q_init.translate, self.Q_init.angle, self.Q_init.scales]},
                    {"params": [self.Q_init.shapes], "lr": 0.1}
                ],
                lr=0.01
            )
        else:
            self.optimizer = torch.optim.Adam(
                [
                    {"params": [self.Q_init.translate, self.Q_init.angle, self.Q_init.scales]},
                ],
                lr=0.01
            )
        with open("./src/super_quadric/scale_prior", "rb") as f:
            scale_prior = pickle.load(f)
            for k in scale_prior:
                scale_prior[k] = torch.tensor(scale_prior[k]).float()
            self.scale_prior = scale_prior
        self.loss_log = []

    def constraint_2d(self, pts_w, Ms, gt_lines_all_direction, gt_masks_all_direction, names):
        n_frames = Ms.shape[0]
        pts_w = pts_w.repeat(n_frames, 1, 1)
        pixels = geo_utils.get_homogeneous(pts_w) @ Ms.permute(0, 2, 1)
        valid_pts = pixels[:, :, 2] > 0.5
        pixels = pixels[:, :, :2] / (torch.abs(pixels[:, :, 2:])+1e-6)
        
        x_min = torch.min(
            torch.where(valid_pts, pixels[:, :, 0], torch.ones_like(pixels[:, :, 0]) * 1000000),
            dim=1).values
        x_max = torch.max(
            torch.where(valid_pts, pixels[:, :, 0], torch.ones_like(pixels[:, :, 0]) * -1000000),
            dim=1).values
        y_min = torch.min(
            torch.where(valid_pts, pixels[:, :, 1], torch.ones_like(pixels[:, :, 1]) * 1000000),
            dim=1).values
        y_max = torch.max(
            torch.where(valid_pts, pixels[:, :, 1], torch.ones_like(pixels[:, :, 1]) * -1000000),
            dim=1).values

        pred_lines = {}
        pred_lines['x_min'] = x_min
        pred_lines['x_max'] = x_max
        pred_lines['y_min'] = y_min
        pred_lines['y_max'] = y_max
        loss_2d = 0
        n_loss = 0
        for name in names:
            losses = F.l1_loss(
                pred_lines[name], -gt_lines_all_direction[name],
                reduction='none')
            losses = torch.where(
                torch.isnan(losses), torch.zeros_like(losses), losses)
            losses = losses * gt_masks_all_direction[name]
            loss_2d += torch.mean(losses)
        return loss_2d

    def run(self, gt_lines, gt_planes, Ms, n_iters=200):
        if isinstance(Ms, np.ndarray):
            Ms = torch.tensor(Ms).float()
        if isinstance(gt_planes, np.ndarray):
            gt_planes = torch.tensor(gt_planes).float()

        names = ['x_min', 'x_max', 'y_min', 'y_max']
        gt_masks_all_direction = {}
        gt_lines_all_direction = {}
        for name in names:
            gt_mask = torch.ones((len(gt_lines))).float()
            gt = torch.zeros((len(gt_lines))).float()
            for idx, gt_line in enumerate(gt_lines):
                if name not in gt_line:
                    gt_mask[idx] = 0
                    gt[idx] = 0
                else:
                    gt[idx] = gt_line[name][-1]
            gt_masks_all_direction[name] = gt_mask
            gt_lines_all_direction[name] = gt

        translate_init = self.Q_init.translate.detach().clone()
        scales_init = self.Q_init.scales.detach().clone()
        for i in range(n_iters):
            torch.autograd.set_detect_anomaly(True)
            self.optimizer.zero_grad()
            pts_w, _ = self.Q_init.compute_ellipsoid_points(use_numpy=False)
            loss_2d = self.constraint_2d(
                pts_w, Ms, gt_lines_all_direction, gt_masks_all_direction, names)
            # loss_3d = self.constraint_3d(pts_w, ground_plane)
            loss = loss_2d
            if self.use_prior:
                scale_inv_var = self.scale_prior[CLASS_MAPPER[self.Q_init.obj_class]]
                loss_3d = (scales_init - self.Q_init.scales)[None, :] @ scale_inv_var @ (scales_init - self.Q_init.scales)[None, :].T
                loss += loss_3d[0, 0] * 20
            # if self.Q_init.obj_class in [3, 5]:#[1, 2, 12, 14, 16]:
            #     loss_3d = 10 * F.l1_loss(translate_init, self.Q_init.translate) + 1000 * F.l1_loss(scales_init, self.Q_init.scales)
            #     loss += loss_3d
            loss.backward()
            self.loss_log.append([loss_2d])
            self.optimizer.step()
            # pred_points, _ = self.Q_est.compute_ellipsoid_points(use_numpy=True)
            # bbox_qc = box_utils.compute_oriented_bbox(pred_points)
        return self.Q_init


    def run_with_intermediate(self, gt_lines, gt_planes, Ms, n_iters=200):
        if isinstance(Ms, np.ndarray):
            Ms = torch.tensor(Ms).float()
        if isinstance(gt_planes, np.ndarray):
            gt_planes = torch.tensor(gt_planes).float()

        names = ['x_min', 'x_max', 'y_min', 'y_max']
        gt_masks_all_direction = {}
        gt_lines_all_direction = {}
        for name in names:
            gt_mask = torch.ones((len(gt_lines))).float()
            gt = torch.zeros((len(gt_lines))).float()
            for idx, gt_line in enumerate(gt_lines):
                if name not in gt_line:
                    gt_mask[idx] = 0
                    gt[idx] = 0
                else:
                    gt[idx] = gt_line[name][-1]
            gt_masks_all_direction[name] = gt_mask
            gt_lines_all_direction[name] = gt

        translate_init = self.Q_init.translate.detach().clone()
        scales_init = self.Q_init.scales.detach().clone()
        step_results = []
        for i in range(n_iters):
            torch.autograd.set_detect_anomaly(True)
            self.optimizer.zero_grad()
            pts_w, _ = self.Q_init.compute_ellipsoid_points(use_numpy=False)
            loss_2d = self.constraint_2d(
                pts_w, Ms, gt_lines_all_direction, gt_masks_all_direction, names)
            # loss_3d = self.constraint_3d(pts_w, ground_plane)
            loss = loss_2d
            if self.use_prior:
                scale_inv_var = self.scale_prior[CLASS_MAPPER[self.Q_init.obj_class]]
                loss_3d = (scales_init - self.Q_init.scales)[None, :] @ scale_inv_var @ (scales_init - self.Q_init.scales)[None, :].T
                loss += loss_3d[0, 0] * 20
            # if self.Q_init.obj_class in [3, 5]:#[1, 2, 12, 14, 16]:
            #     loss_3d = 10 * F.l1_loss(translate_init, self.Q_init.translate) + 1000 * F.l1_loss(scales_init, self.Q_init.scales)
            #     loss += loss_3d
            loss.backward()
            self.loss_log.append([loss_2d])
            self.optimizer.step()
            pred_points, _ = self.Q_init.compute_ellipsoid_points(use_numpy=True)
            bbox_qc = box_utils.compute_oriented_bbox(pred_points)
            _out = {
                "bbox_qc":bbox_qc,
                "surface_points": pred_points
            }
            step_results.append(_out)
        return self.Q_init, step_results



class SuperQuadric():
    def __init__(self, translate, angle, scales, shapes):
        self.shapes = torch.tensor(
            shapes, dtype=torch.float32,
            requires_grad=True)
        self.translate = torch.tensor(
            translate, dtype=torch.float32,
            requires_grad=True)
        self.angle = torch.tensor(
            angle, dtype=torch.float32,
            requires_grad=True)
        self.scales = torch.tensor(
            scales, dtype=torch.float32,
            requires_grad=True)  # input scale is length of 3D bbox
        self.sampler = EqualDistanceSamplerSQ(1000)

    def get_bbox(self, P_cw, if_vectorize=False, line_form=False):
        """ get bounding box by projecting surface points onto the image plane"""
        pts_w, _ = self.compute_ellipsoid_points(use_numpy=True)
        pts_c = geo_utils.get_homogeneous(pts_w) @ P_cw.T
        pts_c /= pts_c[:, 2:]
        min_x, min_y, _ = np.min(pts_c, axis=0)
        max_x, max_y, _ = np.max(pts_c, axis=0)
        return np.array([min_x, min_y, max_x, max_y])

    def rotz(self, angle):
        """Convert euler angles to rotation matrix.
        Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
        Args:
            angle: rotation angle along 3 axis (in radians) -- size = [B, 3]
        Returns:
            Rotation matrix corresponding to the euler angles -- size = [B, 3, 3]
        """
        z = angle

        cosz = torch.cos(z)
        sinz = torch.sin(z)

        zeros = z.detach()*0
        ones = zeros.detach()+1
        zmat = torch.stack([cosz, -sinz, zeros,
                            sinz,  cosz, zeros,
                            zeros, zeros,  ones], dim=0).reshape(3, 3)

        return zmat

    def compute_ellipsoid_points(self, use_numpy):
        """ use uniform sampling algorithm for super quadric (BMVC 1995)"""

        R = self.rotz(self.angle)
        _scales = self.scales ** 2
        _scales = _scales.unsqueeze(0).unsqueeze(0)  # [3] -> [1, 1, 3]
        _shape = squashing(self.shapes, min_=0.2, max_=1.6)
        _shape = _shape.unsqueeze(0).unsqueeze(0)  # [3] -> [1, 1, 3]
        pts, normals = sample_points_on_surface(
            _scales,
            _shape,
            self.sampler
        )  # [1, 1, N, 3]
        pts = pts[0, 0]
        pts = pts @ R.T
        pts += self.translate.unsqueeze(0)  # [3] -> [1, 3]
        if use_numpy:
            pts = pts.detach().numpy()
        return pts, None
