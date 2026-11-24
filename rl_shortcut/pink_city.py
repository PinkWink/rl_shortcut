"""City of Pink geometry: the outer ring road, the floor label map and what the Pinky Pro camera sees of the road.

The policy never gets the robot position. It gets a small "road view": the road mask of the lower camera image
shrunk to 8 x 16 cells (1 = road, 0 = not road).
  training  the view is computed from the floor label map and the robot pose (no rendering, thousands of robots)
  play      the same view is made from the YOLO segmentation of the rendered camera image (road_seg.py)
"""
import math

import numpy as np
import torch
from PIL import Image

from rl_shortcut import ROOT

CITY_USD = str(ROOT / "assets/pink_city/pink_city.usd")
FLOOR_USD = str(ROOT / "assets/pink_city/pink_city_floor.usd")
LABELS_PNG = ROOT / "assets/pink_city/textures/floor_labels.png"   # 0 other, 1 sidewalk, 2 road, 3 centre line, 4 crosswalk

SIZE = 3.0                  # the city is 3 x 3 m, origin in the middle
ROAD = 0.40                 # road width (one lane 0.20)
RING = 1.05                 # centre line of the ring road: a square of half size 1.05 m ...
RING_CORNER = 0.30          # ... with rounded corners of this radius
LANE = 0.10                 # the lane we drive in: this far outside the centre line (anticlockwise = right lane)

# Pinky Pro camera (from the URDF): 6.3 cm above the floor, 3.3 cm ahead of the wheel axle, looking 8 deg down
CAM_X, CAM_HEIGHT, CAM_PITCH, CAM_HFOV = 0.0332, 0.0634, math.radians(8.0), math.radians(66.0)
IMAGE = (640, 360)          # width, height
VIEW_TOP = 0.45             # the view uses the image from 45 % of its height down to the bottom (floor up to 0.6 m ahead)
VIEW_ROWS, VIEW_COLS = 8, 16


def ring_distance(xy):
    """Signed distance [m] from the ring centre line (+ = outside) and the anticlockwise driving direction there."""
    s = RING - RING_CORNER
    q = xy.abs() - s
    outside = q.clamp(min=0.0)
    dist = outside.norm(dim=-1) + q.max(dim=-1).values.clamp(max=0.0) - RING_CORNER
    corner = (q > 0).all(dim=-1, keepdim=True)
    axis = torch.where(q[..., :1] > q[..., 1:], torch.tensor([1.0, 0.0], device=xy.device),
                       torch.tensor([0.0, 1.0], device=xy.device))
    normal = torch.where(corner, outside / outside.norm(dim=-1, keepdim=True).clamp(min=1e-6), axis) * xy.sign()
    tangent = torch.stack([-normal[..., 1], normal[..., 0]], dim=-1)       # normal turned 90 deg to the left
    return dist, tangent


def ring_point(angle, offset):
    """Point at signed distance `offset` from the ring centre line, in the direction `angle` from the city centre."""
    far = 3.0 * torch.stack([angle.cos(), angle.sin()], dim=-1)
    dist, tangent = ring_distance(far)
    normal = torch.stack([tangent[..., 1], -tangent[..., 0]], dim=-1)
    return far - (dist - offset).unsqueeze(-1) * normal, torch.atan2(tangent[..., 1], tangent[..., 0])


def view_points(sub=2):
    """Floor points (robot frame: x ahead, y left) seen by the view cells: (ROWS, COLS, sub * sub, 2)."""
    width, height = IMAGE
    fx = width / 2 / math.tan(CAM_HFOV / 2)                       # focal length in pixels
    top = VIEW_TOP * height
    v = top + (torch.arange(VIEW_ROWS * sub) + 0.5) * (height - top) / (VIEW_ROWS * sub)
    u = (torch.arange(VIEW_COLS * sub) + 0.5) * width / (VIEW_COLS * sub)
    v, u = torch.meshgrid(v, u, indexing="ij")
    right, down = (u - width / 2) / fx, (v - height / 2) / fx     # ray of the pixel, 1 m ahead of the lens
    t = CAM_HEIGHT / (math.sin(CAM_PITCH) + down * math.cos(CAM_PITCH))      # where the ray meets the floor
    pts = torch.stack([CAM_X + t * (math.cos(CAM_PITCH) - down * math.sin(CAM_PITCH)), -t * right], dim=-1)
    pts = pts.view(VIEW_ROWS, sub, VIEW_COLS, sub, 2).permute(0, 2, 1, 3, 4)
    return pts.reshape(VIEW_ROWS, VIEW_COLS, sub * sub, 2)


class RoadMap:
    """The floor label map on the GPU: is this point of the city road?"""

    def __init__(self, device):
        labels = torch.from_numpy(np.array(Image.open(LABELS_PNG))).to(device)
        self.road = (labels >= 2).float()                         # road, centre line and crosswalk are all drivable
        self.n = labels.shape[0]
        self.points = view_points().to(device)

    def is_road(self, xy):
        col = ((xy[..., 0] + SIZE / 2) / SIZE * self.n).long()
        row = ((SIZE / 2 - xy[..., 1]) / SIZE * self.n).long()
        inside = (col >= 0) & (col < self.n) & (row >= 0) & (row < self.n)
        return self.road[row.clamp(0, self.n - 1), col.clamp(0, self.n - 1)] * inside

    def view(self, xy, yaw):
        """Road view (N, ROWS * COLS) of robots at city positions xy (N, 2) heading yaw (N,)."""
        c, s = yaw.cos().view(-1, 1, 1, 1), yaw.sin().view(-1, 1, 1, 1)
        px, py = self.points[..., 0], self.points[..., 1]
        world = torch.stack([xy[:, 0].view(-1, 1, 1, 1) + c * px - s * py,
                             xy[:, 1].view(-1, 1, 1, 1) + s * px + c * py], dim=-1)
        return self.is_road(world).mean(dim=-1).flatten(1)
