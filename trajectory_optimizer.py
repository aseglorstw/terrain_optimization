import argparse
import os
import sys
import numpy as np
from noise import pnoise2
import torch
from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
from visualizer import visualize_mesh_matplotlib
from visualizer import visualize_mesh_open3d
from visualizer import visualize_height_map_mayavi


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


def generate_height_map(width, height, scale, octaves, persistence, lacunarity):
    height_map = np.zeros((height, width))
    for y in range(height):
        for x in range(width):
            height_map[y][x] = pnoise2(x / scale,
                                       y / scale,
                                       octaves=octaves,
                                       persistence=persistence,
                                       lacunarity=lacunarity,
                                       repeatx=1024,
                                       repeaty=1024,
                                       base=42)
    height_map -= height_map.min()
    height_map /= height_map.max()
    height_map *= 50
    return height_map


def main(arguments):
    check_input(arguments.mesh_path)
    device = choose_device()
    mesh = create_mesh_object(arguments.mesh_path, device)
    height_map = generate_height_map(100, 100, 10, 6, 0.5, 2.0)
    visualize_height_map_mayavi(height_map)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_path", default='', type=str, help="Path to the .obj file")
    main(parser.parse_args())
