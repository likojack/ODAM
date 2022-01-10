import math
import numpy as np
import torch
from torchvision.ops.boxes import box_area
from scipy.spatial import ConvexHull


def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def polygon_clip(subjectPolygon, clipPolygon):
   """ Clip a polygon with another polygon.
   Ref: https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python
   Args:
     subjectPolygon: a list of (x,y) 2d points, any polygon.
     clipPolygon: a list of (x,y) 2d points, has to be *convex*
   Note:
     **points have to be counter-clockwise ordered**
   Return:
     a list of (x,y) vertex point for the intersection polygon.
   """
   def inside(p):
      return(cp2[0]-cp1[0])*(p[1]-cp1[1]) > (cp2[1]-cp1[1])*(p[0]-cp1[0])
 
   def computeIntersection():
      dc = [ cp1[0] - cp2[0], cp1[1] - cp2[1] ]
      dp = [ s[0] - e[0], s[1] - e[1] ]
      n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
      n2 = s[0] * e[1] - s[1] * e[0] 
      n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
      return [(n1*dp[0] - n2*dc[0]) * n3, (n1*dp[1] - n2*dc[1]) * n3]
 
   outputList = subjectPolygon
   cp1 = clipPolygon[-1]
 
   for clipVertex in clipPolygon:
      cp2 = clipVertex
      inputList = outputList
      outputList = []
      s = inputList[-1]
 
      for subjectVertex in inputList:
         e = subjectVertex
         if inside(e):
            if not inside(s):
               outputList.append(computeIntersection())
            outputList.append(e)
         elif inside(s):
            outputList.append(computeIntersection())
         s = e
      cp1 = cp2
      if len(outputList) == 0:
          return None
   return(outputList)


def poly_area(x,y):

    """ Ref: http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates """
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))


def convex_hull_intersection(p1, p2):
    """ Compute area of two convex hull's intersection area.
        p1,p2 are a list of (x,y) tuples of hull vertices.
        return a list of (x,y) for the intersection and its volume
    """
    inter_p = polygon_clip(p1,p2)
    if inter_p is not None:
        hull_inter = ConvexHull(inter_p)
        return inter_p, hull_inter.volume
    else:
        return None, 0.0  


def box3d_vol(corners):
    ''' corners: (8,3) no assumption on axis direction '''
    a = np.sqrt(np.sum((corners[0,:] - corners[1,:])**2))
    b = np.sqrt(np.sum((corners[1,:] - corners[2,:])**2))
    c = np.sqrt(np.sum((corners[0,:] - corners[4,:])**2))
    return a*b*c


def box3d_iou(corners1, corners2):
    ''' Compute 3D bounding box IoU.
    Input:
        corners1: numpy array (8,3), assume up direction is negative Y
        corners2: numpy array (8,3), assume up direction is negative Y
    Output:
        iou: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU
    todo (rqi): add more description on corner points' orders.
    '''
    # corner points are in counter clockwise order
    rect1 = [(corners1[i,0], corners1[i,1]) for i in range(3,-1,-1)]
    rect2 = [(corners2[i,0], corners2[i,1]) for i in range(3,-1,-1)] 
    area1 = poly_area(np.array(rect1)[:,0], np.array(rect1)[:,1])
    area2 = poly_area(np.array(rect2)[:,0], np.array(rect2)[:,1])
    inter, inter_area = convex_hull_intersection(rect1, rect2)
    iou_2d = inter_area/(area1+area2-inter_area)
    zmax = min(corners1[0,2], corners2[0,2])
    zmin = max(corners1[4,2], corners2[4,2])
    inter_vol = inter_area * max(0.0, zmax-zmin)
    vol1 = box3d_vol(corners1)
    vol2 = box3d_vol(corners2)
    iou = inter_vol / (vol1 + vol2 - inter_vol)
    return iou, iou_2d


def iou_2d(bboxA: 'np.ndarray', bboxB: 'np.ndarray') -> float:
    """ calculate IoU between two 2D bboxes
    
    Args:
        bboxA ([2, 2] np.ndarray): input bbox A in AABB format
        bboxB ([2, 2] np.ndarray): input bbox B in AABB format
        
    Returns:
        IoU (float): output IoU
    """

    x_min = max(bboxA[0, 0], bboxB[0, 0])
    y_min = max(bboxA[0, 1], bboxB[0, 1])
    x_max = min(bboxA[1, 0], bboxB[1, 0])
    y_max = min(bboxA[1, 1], bboxB[1, 1])
    
    inter_area = max(0, (x_max - x_min)) * max(0, (y_max - y_min))
    area_A = np.prod(bboxA[1] - bboxA[0])
    area_B = np.prod(bboxB[1] - bboxB[0])
    IoU = inter_area / (area_A + area_B - inter_area)
    assert IoU <= 1 and IoU >= 0, "invalid IoU value"
    return IoU


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/
    The boxes should be in [x0, y0, x1, y1] format
    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def get_2d_oriented_bbox(pts_xy):
    """ get an oriented 2D bounding box given a set of 2D points
    
    Args:
        pts_xy: [n_pts, 2] np.ndarray
    
    Returns:
        corner_2d: [4, 2] np.ndarray, corner points of 2D bounding box
    """

    hull = ConvexHull(pts_xy)
    contour = pts_xy[hull.vertices]
    x_mean, y_mean = np.mean(contour, axis=0)
    contour[:, 0] -= x_mean
    contour[:, 1] -= y_mean

    hull_points_2d = contour
    # Compute edges (x2-x1,y2-y1)
    edges = np.zeros( (len(hull_points_2d)-1,2) ) # empty 2 column array
    for i in range( len(edges) ):
        edge_x = hull_points_2d[i+1,0] - hull_points_2d[i,0]
        edge_y = hull_points_2d[i+1,1] - hull_points_2d[i,1]
        edges[i] = [edge_x,edge_y]

    # Calculate edge angles   atan2(y/x)
    edge_angles = np.zeros((len(edges))) # empty 1 column array
    for i in range( len(edge_angles) ):
        edge_angles[i] = math.atan2(edges[i,1], edges[i,0])

    # Check for angles in 1st quadrant
    for i in range(len(edge_angles)):
        edge_angles[i] = abs(edge_angles[i] % (math.pi/2)) # want strictly positive answers

    # Remove duplicate angles
    edge_angles = np.unique(edge_angles)

    # Test each angle to find bounding box with smallest area
    min_bbox = (0, 10000000000, 0, 0, 0, 0, 0, 0) # rot_angle, area, width, height, min_x, max_x, min_y, max_y
    for i in range(len(edge_angles)):

        # Create rotation matrix to shift points to baseline
        # R = [ cos(theta)      , cos(theta-PI/2)
        #       cos(theta+PI/2) , cos(theta)     ]
        R = np.array([
            [math.cos(edge_angles[i]), math.cos(edge_angles[i]-(math.pi/2))],
            [math.cos(edge_angles[i]+(math.pi/2)), math.cos(edge_angles[i])]])

        # Apply this rotation to convex hull points
        rot_points = np.dot(R, np.transpose(hull_points_2d) ) # 2x2 * 2xn
        # Find min/max x,y points
        min_x = np.nanmin(rot_points[0], axis=0)
        max_x = np.nanmax(rot_points[0], axis=0)
        min_y = np.nanmin(rot_points[1], axis=0)
        max_y = np.nanmax(rot_points[1], axis=0)

        # Calculate height/width/area of this bounding rectangle
        width = max_x - min_x
        height = max_y - min_y
        area = width * height

        # Store the smallest rect found first (a simple convex hull might have 2 answers with same area)
        if (area < min_bbox[1]):
            min_bbox = (edge_angles[i], area, width, height, min_x, max_x, min_y, max_y)
        # Bypass, return the last found rect
        #min_bbox = ( edge_angles[i], area, width, height, min_x, max_x, min_y, max_y )

    # Re-create rotation matrix for smallest rect
    angle = min_bbox[0]   
    R = np.array([[math.cos(angle), math.cos(angle-(math.pi/2))], [math.cos(angle+(math.pi/2)), math.cos(angle)]])

    # Project convex hull points onto rotated frame
    proj_points = np.dot(R, np.transpose(hull_points_2d)) # 2x2 * 2xn

    # min/max x,y points are against baseline
    min_x = min_bbox[4]
    max_x = min_bbox[5]
    min_y = min_bbox[6]
    max_y = min_bbox[7]

    corner_2d = np.zeros((4, 2)) # empty 2 column array
    corner_2d[0] = np.dot([max_x, max_y], R)
    corner_2d[1] = np.dot([max_x, min_y], R)
    corner_2d[2] = np.dot([min_x, min_y], R)
    corner_2d[3] = np.dot([min_x, max_y], R)
    corner_2d[:, 0] += x_mean
    corner_2d[:, 1] += y_mean
    return corner_2d


def get_bbox_and_orientation(vertices):
    """ get oriented 3D bounding box for an instance.

    assume positive z is up direction
    """

    z_min = np.min(vertices[:, 2])
    z_max = np.max(vertices[:, 2])
    bbox_2d = get_2d_oriented_bbox(vertices[:, :2])

    axis1 = np.sqrt(np.sum((bbox_2d[0, :] - bbox_2d[1, :]) ** 2))
    axis2 = np.sqrt(np.sum((bbox_2d[0, :] - bbox_2d[3, :]) ** 2))
    if axis1 > axis2:
        long_axis = bbox_2d[0, :] - bbox_2d[1, :]
    else:  # axis1 < axis2
        long_axis = bbox_2d[0, :] - bbox_2d[3, :]
    long_axis = long_axis / np.linalg.norm(long_axis)
    cos_theta = np.dot(long_axis, np.array([1, 0]))
    theta = np.arccos(cos_theta)

    corners_upper_half = np.concatenate(
        [bbox_2d, np.array([[z_max]*4]).T], axis=1)
    corners_lower_half = np.concatenate(
        [bbox_2d, np.array([[z_min]*4]).T], axis=1)
    corners = np.concatenate([corners_upper_half, corners_lower_half], axis=0)
    return corners, theta


def get_3d_box(box_size, rot_mat, center):
    ''' box_size is array(l,w,h), heading_angle is radius clockwise from pos x axis, center is xyz of box center
        output (8,3) array for 3D box cornders
        Similar to utils/compute_orientation_3d

    Args:
        box_size (1D iterator): [l, w, h] for x y z
        rot_mat (np.ndarray): [3, 3] rotation matrix around z axis
        center (np.array): [3] translation vector

    '''

    # assert np.abs(np.abs(rot_mat[2, 2]) - 1) < 1e-1, "the rot mat should be around z axis"
    l,w,h = box_size
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2]
    y_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]
    z_corners = [h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2]
    corners_3d = np.dot(rot_mat, np.vstack([x_corners,y_corners,z_corners]))
    corners_3d[0,:] = corners_3d[0,:] + center[0]
    corners_3d[1,:] = corners_3d[1,:] + center[1]
    corners_3d[2,:] = corners_3d[2,:] + center[2]
    corners_3d = np.transpose(corners_3d)
    return corners_3d


def rotz(t):
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  -s,  0],
                    [s,  c,  0],
                    [0, 0,  1]])


def compute_oriented_bbox(pts):
    """ compute oriented 3D bbox for ellipsoid from points

    Assume positive z is the upright direction

    Args:
        corners (np.ndarray): [n_pts, 3] numpy array for ellipsoid surface
    """
    import matplotlib.pyplot as plt
    z_min = np.min(pts[:, 2])
    z_max = np.max(pts[:, 2])
    pts_xy = pts[:, :2]
    hull = ConvexHull(pts_xy)
    contour = pts_xy[hull.vertices]
    x_mean, y_mean = np.mean(contour, axis=0)
    contour[:, 0] -= x_mean
    contour[:, 1] -= y_mean

    hull_points_2d = contour
    # Compute edges (x2-x1,y2-y1)
    edges = np.zeros( (len(hull_points_2d)-1,2) ) # empty 2 column array
    for i in range( len(edges) ):
        edge_x = hull_points_2d[i+1,0] - hull_points_2d[i,0]
        edge_y = hull_points_2d[i+1,1] - hull_points_2d[i,1]
        edges[i] = [edge_x,edge_y]

    # Calculate edge angles   atan2(y/x)
    edge_angles = np.zeros((len(edges))) # empty 1 column array
    for i in range( len(edge_angles) ):
        edge_angles[i] = math.atan2(edges[i,1], edges[i,0])

    # Check for angles in 1st quadrant
    for i in range(len(edge_angles)):
        edge_angles[i] = abs(edge_angles[i] % (math.pi/2)) # want strictly positive answers

    # Remove duplicate angles
    edge_angles = np.unique(edge_angles)

    # Test each angle to find bounding box with smallest area
    min_bbox = (0, 10000000000, 0, 0, 0, 0, 0, 0) # rot_angle, area, width, height, min_x, max_x, min_y, max_y
    for i in range(len(edge_angles)):

        # Create rotation matrix to shift points to baseline
        # R = [ cos(theta)      , cos(theta-PI/2)
        #       cos(theta+PI/2) , cos(theta)     ]
        R = np.array([
            [math.cos(edge_angles[i]), math.cos(edge_angles[i]-(math.pi/2))],
            [math.cos(edge_angles[i]+(math.pi/2)), math.cos(edge_angles[i])]])

        # Apply this rotation to convex hull points
        rot_points = np.dot(R, np.transpose(hull_points_2d) ) # 2x2 * 2xn
        # Find min/max x,y points
        min_x = np.nanmin(rot_points[0], axis=0)
        max_x = np.nanmax(rot_points[0], axis=0)
        min_y = np.nanmin(rot_points[1], axis=0)
        max_y = np.nanmax(rot_points[1], axis=0)

        # Calculate height/width/area of this bounding rectangle
        width = max_x - min_x
        height = max_y - min_y
        area = width * height

        # Store the smallest rect found first (a simple convex hull might have 2 answers with same area)
        if (area < min_bbox[1]):
            min_bbox = (edge_angles[i], area, width, height, min_x, max_x, min_y, max_y)
        # Bypass, return the last found rect
        #min_bbox = ( edge_angles[i], area, width, height, min_x, max_x, min_y, max_y )

    # Re-create rotation matrix for smallest rect
    angle = min_bbox[0]   
    R = np.array([[math.cos(angle), math.cos(angle-(math.pi/2))], [math.cos(angle+(math.pi/2)), math.cos(angle)]])

    # Project convex hull points onto rotated frame
    proj_points = np.dot(R, np.transpose(hull_points_2d)) # 2x2 * 2xn

    # min/max x,y points are against baseline
    min_x = min_bbox[4]
    max_x = min_bbox[5]
    min_y = min_bbox[6]
    max_y = min_bbox[7]

    corner_2d = np.zeros((4, 2)) # empty 2 column array
    corner_2d[0] = np.dot([max_x, max_y], R)
    corner_2d[1] = np.dot([max_x, min_y], R)
    corner_2d[2] = np.dot([min_x, min_y], R)
    corner_2d[3] = np.dot([min_x, max_y], R)
    corner_2d[:, 0] += x_mean
    corner_2d[:, 1] += y_mean
    corners_upper_half = np.concatenate([corner_2d, np.array([[z_max]*4]).T], axis=1)
    corners_lower_half = np.concatenate([corner_2d, np.array([[z_min]*4]).T], axis=1)
    corners = np.concatenate([corners_upper_half, corners_lower_half], axis=0)
    return corners    


def dim2corners(dimensions, orientation):
    l, w, h = dimensions
    rot_mat = rotz(orientation)
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2]
    y_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]
    z_corners = [h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2]
    corners_3d = np.dot(
        rot_mat, np.vstack([x_corners,y_corners,z_corners])).T
    return corners_3d


def iou_3d(bboxA: 'np.ndarray', bboxB: 'np.ndarray') -> float:
    """ calculate 3D IoU between two 3D bboxes

    Args:
        bboxA ([2, 3] np.ndarray): input bbox A in AABB format
        bboxB ([2, 3] np.ndarray): input bbox B in AABB format
        
    Returns:
        IoU (float): 3D IoU
    """

    x_min = max(bboxA[0, 0], bboxB[0, 0])
    y_min = max(bboxA[0, 1], bboxB[0, 1])
    z_min = max(bboxA[0, 2], bboxB[0, 2])
    x_max = min(bboxA[1, 0], bboxB[1, 0])
    y_max = min(bboxA[1, 1], bboxB[1, 1])
    z_max = min(bboxA[1, 2], bboxB[1, 2])
    
    inter_volume = max(0, (x_max - x_min)) * max(0, (y_max - y_min)) * max(0, (z_max - z_min))
    volume_A = np.prod(bboxA[1] - bboxA[0])
    volume_B = np.prod(bboxB[1] - bboxB[0])
    IoU = inter_volume / (volume_A + volume_B - inter_volume)
    assert IoU <= 1 and IoU >= 0, "invalid IoU value"
    return IoU


def box_cxcywh_to_xyxy(bbox):
    """convert xyxy representation to center_x, center_y, width, height

    Args:
        bbox ([N, 4] np.ndarray): bbox [x_min, y_min, x_max, y_max]

    Returns:
        bbox ([N, 4] np.ndarray): bbox [center_x, center_y, width, height]
    """

    cx, cy, w, h = bbox[:, :1], bbox[:, 1:2], bbox[:, 2:3], bbox[:, 3:]
    assert (cx >= 0).all()
    assert (cy >= 0).all()
    assert (w >= 0).all()
    assert (h >= 0).all()
    bbox = np.concatenate([(cx - w/2.), (cy - h/2),
        (cx + w/2), (cy + h/2)], axis=1)
    return bbox


def box_xyxy_to_cxcywh(bbox):
    """convert xyxy representation to center_x, center_y, width, height

    Args:
        bbox ([N, 4] np.ndarray): bbox [x_min, y_min, x_max, y_max]

    Returns:
        bbox ([N, 4] np.ndarray): bbox [center_x, center_y, width, height]
    """

    x0, y0, x1, y1 = bbox[:, :1], bbox[:, 1:2], bbox[:, 2:3], bbox[:, 3:]
    assert (x0 >= 0).all()
    assert (y0 >= 0).all()
    assert (x1 >= x0).all()
    assert (y1 >= y0).all()
    bbox = np.concatenate([(x0 + x1) / 2, (y0 + y1) / 2,
        (x1 - x0), (y1 - y0)], axis=1)
    return bbox


def box_cxcywh_to_xyxy_pytorch(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh_pytorch(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)