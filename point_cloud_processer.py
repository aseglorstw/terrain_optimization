import numpy as np
import torch
from pytorch3d.structures import Pointclouds

import visualizer


def load_point_cloud(device, path_to_point_cloud, translation_vector, rotation_matrix):
    point_cloud = np.load(path_to_point_cloud)
    transformed_point_cloud = transform_point_cloud(point_cloud, translation_vector, rotation_matrix)
    return Pointclouds(points=[move_point_cloud_to_VRAM(transformed_point_cloud, device)])


def transform_point_cloud(point_cloud, translation_vector, rotation_matrix):
    centered_point_cloud = point_cloud - np.mean(point_cloud, axis=0) + translation_vector
    rotated_point_cloud = rotation_matrix @ centered_point_cloud.T
    transformed_point_cloud = rotated_point_cloud.T
    return transformed_point_cloud


def move_point_cloud_to_VRAM(point_cloud, device):
    points_tensor = torch.tensor(point_cloud, dtype=torch.float32)
    return points_tensor.to(device)


def combine_point_clouds(point_clouds):
    packed_point_clouds = []
    for point_cloud in point_clouds:
        packed_point_clouds.append(point_cloud.points_packed())
    combined_points = torch.cat(packed_point_clouds, dim=0)
    return Pointclouds(points=[combined_points])

