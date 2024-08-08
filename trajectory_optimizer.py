import argparse
import os
import sys
import numpy as np
import torch
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from visualizer import visualize_height_map_mayavi
from visualizer import visualize_mesh_open3d
from visualizer import visualize_mesh_open3d_with_points


def check_input(path_to_mesh):
    if not os.path.isfile(path_to_mesh):
        print(f"The file {path_to_mesh} does not exists")
        sys.exit("Exiting the program due to missing file.")


def choose_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        print("WARNING: CPU only, this will be slow!")
    return device


def is_device_GPU(device):
    return device.type == 'cuda'


def move_data_to_VRAM(vertices, faces, device):
    return vertices.to(device), faces.verts_idx.to(device)


def normalize_mesh(vertices):
    return (vertices - vertices.mean(0)) / max(vertices.abs().max(0)[0])


def create_mesh_object(path_to_obj_file, device):
    vertices, faces, aux = load_obj(path_to_obj_file)
    if is_device_GPU(device):
        vertices, faces = move_data_to_VRAM(vertices, faces, device)
    vertices = normalize_mesh(vertices)
    return Meshes(verts=[vertices], faces=[faces])


def generate_init_height_map(width, height):
    return np.full((height, width), 0)


def height_map_to_mesh(height_map):
    height, width = height_map.shape
    x = np.arange(0, width)
    y = np.arange(0, height)
    x, y = np.meshgrid(x, y)
    z = height_map
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    vertices = np.column_stack((x, y, z))
    faces = []
    for i in range(height - 1):
        for j in range(width - 1):
            v1 = i * width + j
            v2 = v1 + 1
            v3 = v1 + width
            v4 = v3 + 1
            faces.append([v1, v2, v4])
            faces.append([v1, v4, v3])
    vertices_tensor = torch.tensor(vertices, dtype=torch.float32)
    faces_tensor = torch.tensor(faces, dtype=torch.int64)
    mesh = Meshes(verts=[vertices_tensor], faces=[faces_tensor])
    return mesh


def load_point_cloud(desired_center=np.array([5, 5, 5])):
    points = np.load('point_clouds/test_robot.npy')
    current_center = np.mean(points, axis=0)
    translation_vector = desired_center - current_center
    translated_points = points + translation_vector
    return translated_points


def main(arguments):
    check_input(arguments.mesh_path)
    device = choose_device()
    #mesh = create_mesh_object(arguments.mesh_path, device)
    cuboid_points = load_point_cloud()
    height_map = generate_init_height_map(10, 10)
    mesh_height_map = height_map_to_mesh(height_map)
    visualize_mesh_open3d_with_points(mesh_height_map, cuboid_points)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_path", default='', type=str, help="Path to the .obj file")
    main(parser.parse_args())
