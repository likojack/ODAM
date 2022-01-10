import numpy as np
import trimesh
import torch
import torch.nn.functional as F
import open3d as o3d
import OpenGL.GL as gl
import pangolin

from sampling import EqualDistanceSamplerSQ, sample_points_on_surface
from dom.libs.o3d_helper import np2pc
import dom.player.utils as utils


def squashing(shape, min=0.2, max=1.6):
    return torch.sigmoid(shape) * (max - min) + min


if __name__ == "__main__":
    """ test forward/backward with superquadric"""

    width, height = 400, 300
    scam, dcam, dimg, dpred, dtrack, texture = utils.init_panal(
        1000., width, height
    )
    panel = pangolin.CreatePanel('ui')
    panel.SetBounds(height / 768, 1.0, 0.0, 175 / 1024.)
    gt_pos_x = pangolin.VarFloat('ui.gt_pos_x', value=2, min=-3, max=3)
    gt_pos_y = pangolin.VarFloat('ui.gt_pos_y', value=0, min=-3, max=3)
    gt_pos_z = pangolin.VarFloat('ui.gt_pos_z', value=0, min=-3, max=3)
    gt_scale_x = pangolin.VarFloat('ui.gt_scale_x', value=2, min=0.2, max=2)
    gt_scale_y = pangolin.VarFloat('ui.gt_scale_y', value=1, min=0.2, max=2)
    gt_scale_z = pangolin.VarFloat('ui.gt_scale_z', value=1, min=0.2, max=2)
    gt_eps_0 = pangolin.VarFloat('ui.gt_eps_0', value=1, min=0.2, max=1.6)
    gt_eps_1 = pangolin.VarFloat('ui.gt_eps_1', value=1, min=0.2, max=1.6)
    Run = pangolin.VarBool('ui.run', value=False, toggle=False)

    sampler = EqualDistanceSamplerSQ(1000)

    while not pangolin.ShouldQuit():
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(0.95, 0.95, 0.95, 1.0)
        dcam.Activate(scam)
        pangolin.glDrawColouredCube()
        utils.draw_frame_mesh(dcam, scam)

        gt_shape = torch.tensor(
            [[[float(gt_eps_0), float(gt_eps_1)]]],
            dtype=torch.float32)
        gt_size = torch.tensor(
            [[[float(gt_scale_x), float(gt_scale_y), float(gt_scale_z)]]],
            dtype=torch.float32)  # [1, 1, 3]
        gt_pos = torch.tensor(
            [[float(gt_pos_x), float(gt_pos_y), float(gt_pos_z)]],
            dtype=torch.float32
        )  # [1, 3]
        gt_pts, normals = sample_points_on_surface(
            gt_size,
            gt_shape,
            sampler
        )  # [1, 1, N, 3]
        gt_pts = gt_pts[0, 0]
        normals = normals[0, 0]
        gt_pts += gt_pos
        gt_colors = np.zeros_like(gt_pts.numpy())
        gt_colors[:, 0] = 1
        utils.draw_3d_points(dcam, scam, gt_pts.numpy(), gt_colors, pt_size=10)
        
        m = trimesh.Trimesh(vertices=gt_pts).convex_hull
        faces = m.faces
        pts = m.vertices
        normals = m.vertex_normals
        utils.draw_mesh(dcam, scam, pts, faces, normals)
        init_size = torch.tensor(
            [[[1, 1, 1]]], dtype=torch.float32, requires_grad=True)
        init_shape = torch.tensor(
            [[[0, 0]]], dtype=torch.float32, requires_grad=True)
        init_pos = torch.tensor(
            [[0., 0., 0.]], dtype=torch.float32, requires_grad=True)


        if pangolin.Pushed(Run):
            optim = torch.optim.Adam([init_size, init_shape, init_pos], lr=0.001)
            for i in range(1000):
                optim.zero_grad()
                _size = init_size ** 2 # ensure positivity
                _shape = squashing(init_shape, min=0.2, max=1.6)  # maintain convex shape
                pts, normals = sample_points_on_surface(
                    _size,
                    _shape,
                    sampler
                )
                pts = pts[0, 0]
                pts += init_pos
                loss = F.l1_loss(pts, gt_pts)
                print(loss)
                loss.backward()
                optim.step()

            print("optimized size: ")
            print(init_size)
            print("optimized shape: ")
            print(init_shape)
            print("optimized position: ")
            print(init_pos)
        
        pts, normals = sample_points_on_surface(
            init_size**2,
            squashing(init_shape, min=0.2, max=1.6),
            sampler
        )  # [1, 1, N, 3]
        pts = pts[0, 0]
        pts += init_pos
        pts = pts.detach().numpy()
        pred_colors = np.zeros_like(pts)
        pred_colors[:, 1] = 1
        utils.draw_3d_points(dcam, scam, pts, pred_colors, pt_size=10)
        pangolin.FinishFrame()