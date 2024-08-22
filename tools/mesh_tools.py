import torch
import pyvista as pv
import numpy as np
from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
from tools import device_tools
import fast_simplification
import vtk


def load_mesh(path_to_obj_file, device):
    vertices, faces, aux = load_obj(path_to_obj_file)
    if device_tools.is_device_GPU(device):
        vertices, faces = move_mesh_to_VRAM(vertices, faces, device)
    return Meshes(verts=[vertices], faces=[faces])


def move_mesh_to_VRAM(vertices, faces, device):
    return vertices.to(device), faces.verts_idx.to(device)


def simplify_mesh(input_file, output_path, reduction):
    reader = vtk.vtkOBJReader()
    reader.SetFileName(input_file)
    reader.Update()
    mesh = reader.GetOutput()
    decimator = vtk.vtkQuadricDecimation()
    decimator.SetInputData(mesh)
    decimator.SetTargetReduction(reduction)
    decimator.Update()
    simplified_mesh = decimator.GetOutput()
    writer = vtk.vtkOBJWriter()
    writer.SetFileName(output_path)
    writer.SetInputData(simplified_mesh)
    writer.Write()
    return simplified_mesh


def generate_terrain_mesh_and_save(output_path):
    # Generate grid points
    nx, ny = 150, 150
    x = np.linspace(20, 30, nx)
    y = np.linspace(20, 30, ny)
    x, y = np.meshgrid(x, y)

    # Define terrain heights (clamp to z >= 0)
    z = np.sin(x) * np.cos(y)
    # z[z < 0] = 0  # Ensure z-values are non-negative (clamping)

    # Create the structured grid with clamped heights
    grid = pv.StructuredGrid(x, y, z)

    # Create a flat floor at z = 0
    floor_z = np.zeros_like(z)
    floor = pv.StructuredGrid(x, y, floor_z).extract_surface().triangulate()

    # Combine the terrain (roof) and the floor
    combined_mesh = grid.extract_surface().triangulate()

    # Optional: Plot the combined terrain mesh
    plotter = pv.Plotter()
    plotter.add_mesh(combined_mesh, cmap="terrain", show_edges=True)
    plotter.show()

    # Compute normals for shading and smoothness
    combined_mesh = combined_mesh.compute_normals()
    normals = combined_mesh.point_data["Normals"]
    # Flip the normals manually if their z-component is negative
    for i in range(normals.shape[0]):
        if normals[i, 2] < 0:  # If the z-component of the normal is negative
            normals[i] = -normals[i]  # Flip the normal

    # Update the mesh with the corrected normals
    combined_mesh.point_data["Normals"] = normals
    # Save the combined mesh to the specified file path
    combined_mesh.save(output_path)

def transform_mesh(path_to_mesh, translation, rotation):
    mesh = pv.read(path_to_mesh)
    center = mesh.center
    mesh.translate(-np.array(center), inplace=True)
    mesh.rotate_x(rotation[0], inplace=True)
    mesh.rotate_y(rotation[1], inplace=True)
    mesh.rotate_z(rotation[2], inplace=True)
    mesh.translate(translation, inplace=True)
    return mesh

def save_mesh(mesh, path_to_file):
    vertices = mesh.verts_packed().detach().cpu().numpy()
    faces = mesh.faces_packed().detach().cpu().numpy()
    mesh_to_save = Meshes(verts=[torch.tensor(vertices, dtype=torch.float32)],
                          faces=[torch.tensor(faces, dtype=torch.int64)])
    save_obj(path_to_file, mesh_to_save.verts_packed(), mesh_to_save.faces_packed())

