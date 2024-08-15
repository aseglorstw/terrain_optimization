import torch
from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
from tools import device_tools


def load_mesh(path_to_obj_file, device):
    vertices, faces, aux = load_obj(path_to_obj_file)
    if device_tools.is_device_GPU(device):
        vertices, faces = move_mesh_to_VRAM(vertices, faces, device)
    return Meshes(verts=[vertices], faces=[faces])


def save_mesh(mesh, path_to_file):
    vertices = mesh.verts_packed().detach().cpu().numpy()
    faces = mesh.faces_packed().detach().cpu().numpy()
    mesh_to_save = Meshes(verts=[torch.tensor(vertices, dtype=torch.float32)],
                          faces=[torch.tensor(faces, dtype=torch.int64)])
    save_obj(path_to_file, mesh_to_save.verts_packed(), mesh_to_save.faces_packed())


def move_mesh_to_VRAM(vertices, faces, device):
    return vertices.to(device), faces.verts_idx.to(device)


def normalize_mesh(vertices):
    return (vertices - vertices.mean(0)) / max(vertices.abs().max(0)[0])
