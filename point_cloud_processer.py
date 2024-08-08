import numpy as np
import torch
from pytorch3d.structures import Pointclouds


def load_point_cloud(device, path_to_point_cloud, desired_center=np.array([25, 25, 5])):
    points = np.load(path_to_point_cloud)
    current_center = np.mean(points, axis=0)
    translation_vector = desired_center - current_center
    translated_points = points + translation_vector
    return Pointclouds(points=[move_point_cloud_to_VRAM(translated_points, device)])


def move_point_cloud_to_VRAM(point_cloud, device):
    points_tensor = torch.tensor(point_cloud, dtype=torch.float32)
    return points_tensor.to(device)


def combine_point_clouds(point_clouds):
    packed_point_clouds = []
    for point_cloud in point_clouds:
        packed_point_clouds.append(point_cloud.points_packed())
    combined_points = torch.cat(packed_point_clouds, dim=0)
    return Pointclouds(points=[combined_points])


def generate_point_cloud_model():
    # points = np.array([[-1, -1, 10],
    #                    [-0.7, -1, 10],
    #                    [-1, -0.7, 10],
    #                    [-0.7, -0.7, 10],
    #                    [1, -1, 10],
    #                    [0.7, -1, 10],
    #                    [1, -0.7, 10],
    #                    [0.7, -0.7, 10],
    #                    [-1, 1, 10],
    #                    [-0.7, 1, 10],
    #                    [-1, 0.7, 10],
    #                    [-0.7, 0.7, 10],
    #                    [1, 1, 10],
    #                    [0.7, 1, 10],
    #                    [1, 0.7, 10],
    #                    [0.7, 0.7, 10]])
    # np.save('point_clouds/test_robot_without_roof.npy', points)
    roof_points = []
    for x in np.linspace(-1, 1, 7):
        for y in np.linspace(-1, 1, 7):
            roof_points.append([x, y, 11])

    additional_points = [[0.85, -0.85, 10.5],
                        [-0.85, 0.85, 10.5],
                        [0.85, 0.85, 10.5],
                        [-0.85, -0.85, 10.5]]
    np.save('point_clouds/test_robot_roof.npy', roof_points + additional_points)


if __name__ == '__main__':
    generate_point_cloud_model()
