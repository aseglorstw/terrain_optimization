import numpy as np
import torch
from pytorch3d.structures import Pointclouds


def load_point_cloud(device, desired_center=np.array([5, 5, 5])):
    points = np.load('point_clouds/test_robot.npy')
    current_center = np.mean(points, axis=0)
    translation_vector = desired_center - current_center
    translated_points = points + translation_vector
    return Pointclouds(points=[move_point_cloud_to_VRAM(translated_points, device)])


def move_point_cloud_to_VRAM(point_cloud, device):
    points_tensor = torch.tensor(point_cloud, dtype=torch.float32)
    return points_tensor.to(device)
