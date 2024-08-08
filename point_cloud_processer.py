import numpy as np
import torch
from pytorch3d.structures import Pointclouds


def load_point_cloud(device, desired_center=np.array([25, 25, 5])):
    points = np.load('point_clouds/test_robot_without_roof.npy')
    current_center = np.mean(points, axis=0)
    translation_vector = desired_center - current_center
    translated_points = points + translation_vector
    return Pointclouds(points=[move_point_cloud_to_VRAM(translated_points, device)])


def move_point_cloud_to_VRAM(point_cloud, device):
    points_tensor = torch.tensor(point_cloud, dtype=torch.float32)
    return points_tensor.to(device)


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
    for x in np.linspace(-1, 1, 10):
        for y in np.linspace(-1, 1, 10):
            roof_points.append([x, y, 11])
    np.save('point_clouds/test_robot_roof.npy', roof_points)


if __name__ == '__main__':
    generate_point_cloud_model()
