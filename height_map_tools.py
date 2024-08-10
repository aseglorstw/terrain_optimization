import numpy as np
import torch
from pytorch3d.structures import Meshes
import mesh_tools
import device_tools


def generate_init_height_map(width, height):
    return np.full((height, width), 0)


def height_map_to_mesh(height_map, device):
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
    vertices = torch.tensor(vertices, dtype=torch.float32)
    faces = torch.tensor(faces, dtype=torch.int64)
    if device_processer.is_device_GPU(device):
        vertices, faces = move_mesh_to_VRAM(vertices, faces, device)
    mesh = Meshes(verts=[vertices], faces=[faces])
    return mesh


def move_mesh_to_VRAM(vertices, faces, device):
    return vertices.to(device), faces.to(device)
