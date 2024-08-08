from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
import device_processer


def create_mesh_object(path_to_obj_file, device):
    vertices, faces, aux = load_obj(path_to_obj_file)
    if device_processer.is_device_GPU(device):
        vertices, faces = move_mesh_to_VRAM(vertices, faces, device)
    vertices = normalize_mesh(vertices)
    return Meshes(verts=[vertices], faces=[faces])


def move_mesh_to_VRAM(vertices, faces, device):
    return vertices.to(device), faces.verts_idx.to(device)


def normalize_mesh(vertices):
    return (vertices - vertices.mean(0)) / max(vertices.abs().max(0)[0])
