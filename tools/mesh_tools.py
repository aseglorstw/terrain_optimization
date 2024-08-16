import torch
import pyvista as pv
import numpy as np
from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
from tools import device_tools


def load_mesh(path_to_obj_file, device):
    vertices, faces, aux = load_obj(path_to_obj_file)
    if device_tools.is_device_GPU(device):
        vertices, faces = move_mesh_to_VRAM(vertices, faces, device)
    return Meshes(verts=[vertices], faces=[faces])


def move_mesh_to_VRAM(vertices, faces, device):
    return vertices.to(device), faces.verts_idx.to(device)


def simplify_mesh_and_save(path_to_mesh, output_path, reduction):
    mesh = pv.read(path_to_mesh)
    simplified_mesh = mesh.decimate_pro(reduction)
    simplified_mesh.save(output_path)


def generate_terrain_mesh_and_save(output_path):
    nx, ny = 100, 100
    x = np.linspace(0, 50, nx)
    y = np.linspace(0, 50, ny)
    x, y = np.meshgrid(x, y)
    z = np.sin(x) * np.cos(y)
    grid = pv.StructuredGrid(x, y, z)
    poly_data = grid.extract_surface()
    plotter = pv.Plotter()
    plotter.add_mesh(poly_data, cmap="terrain", show_edges=True)
    poly_data.save(output_path)


def save_mesh(mesh, path_to_file):
    vertices = mesh.verts_packed().detach().cpu().numpy()
    faces = mesh.faces_packed().detach().cpu().numpy()
    mesh_to_save = Meshes(verts=[torch.tensor(vertices, dtype=torch.float32)],
                          faces=[torch.tensor(faces, dtype=torch.int64)])
    save_obj(path_to_file, mesh_to_save.verts_packed(), mesh_to_save.faces_packed())

