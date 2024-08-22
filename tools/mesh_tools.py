import pyvista as pv
import numpy as np
import vtk


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


def generate_terrain_mesh_and_save(output_path):
    nx, ny = 150, 150
    x = np.linspace(20, 30, nx)
    y = np.linspace(20, 30, ny)
    x, y = np.meshgrid(x, y)
    z = np.sin(x) * np.cos(y)

    grid = pv.StructuredGrid(x, y, z)
    combined_mesh = grid.extract_surface().triangulate()

    plotter = pv.Plotter()
    plotter.add_mesh(combined_mesh, cmap="terrain", show_edges=True)
    plotter.show()

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
