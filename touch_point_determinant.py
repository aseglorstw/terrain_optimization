import meshlib.mrmeshpy as mr
import torch
from pytorch3d.structures import Meshes
import pyvista as pv
import numpy as np
from meshlib import mrmeshnumpy as mn
import time
from tools import visualize_tools
import vtk


def get_touch_points_meshlib_insideA_or_outsideB(path_to_robot_mesh, path_to_terrain_mesh):
    robot_mesh = mr.loadMesh(path_to_robot_mesh)
    terrain_mesh = mr.loadMesh(path_to_terrain_mesh)
    first_normal = terrain_mesh.normal(mr.FaceId())
    boolean_operation = mr.BooleanOperation.OutsideA if first_normal.z < 0 else mr.BooleanOperation.InsideA
    start = time.time()
    boolean_result = mr.boolean(robot_mesh, terrain_mesh, boolean_operation)
    mesh = boolean_result.mesh
    vertices = mn.getNumpyVerts(mesh)
    faces = mn.getNumpyFaces(mesh.topology)
    vertices_tensor = torch.tensor(vertices, dtype=torch.float32)
    faces_tensor = torch.tensor(faces, dtype=torch.int64)
    mesh = Meshes(verts=[vertices_tensor], faces=[faces_tensor])
    end = time.time()
    print(f"meshlib insideA: {(end - start) * 1000}  milliseconds")

    faces_with_size = np.hstack([np.full((faces.shape[0], 1), 3), faces]).flatten()
    pv_mesh = pv.PolyData(vertices, faces_with_size)
    pv_mesh = pv_mesh.compute_normals(flip_normals=True)
    cells = pv_mesh.cell_centers().points
    nth = 100  # adjust this value to change the density of normals
    subsampled_points = pv_mesh.extract_points(np.arange(0, pv_mesh.n_points, nth))
    pv_robot = pv.read(path_to_robot_mesh)
    pv_terrain = pv.read(path_to_terrain_mesh)
    plotter = pv.Plotter()
    plotter.add_mesh(pv_robot, style='wireframe', color='yellow')
    plotter.add_mesh(pv_terrain, style='wireframe', color='purple')
    plotter.add_mesh(pv_mesh, show_edges=True, color='red')
    plotter.add_points(cells, color='green')
    plotter.add_mesh(subsampled_points.glyph(orient='Normals', scale=False, factor=0.05), color='blue')
    plotter.show()


def get_touch_points_pyvista_intersection(path_to_robot_mesh, terrain_mesh):
    robot_mesh = pv.read(path_to_robot_mesh)
    terrain_mesh = pv.read(terrain_mesh)
    start = time.time()
    result, result1, result2 = robot_mesh.intersection(terrain_mesh)
    end = time.time()
    print(f"pyvista intersection: {(end - start) * 1000}  milliseconds")
    plotter = pv.Plotter()
    plotter.add_mesh(result,  color='r', line_width=10)
    plotter.add_mesh(result1, style='wireframe')
    plotter.add_mesh(result2, style='wireframe')
    plotter.show()


def get_touch_points_pyvista_boolean_intersection(path_to_robot_mesh, terrain_mesh):
    robot_mesh = pv.read(path_to_robot_mesh)
    terrain_mesh = pv.read(terrain_mesh)
    terrain_mesh = terrain_mesh.compute_normals(flip_normals=True)
    start = time.time()
    result = robot_mesh.boolean_intersection(terrain_mesh)
    end = time.time()
    print(f"pyvista boolean intersection: {(end - start) * 1000}  milliseconds")
    plotter = pv.Plotter()
    plotter.add_mesh(result, color='r', line_width=10)
    plotter.show()


def get_touch_points_pyvista_collision(path_to_robot_mesh, terrain_mesh):
    robot_mesh = pv.read(path_to_robot_mesh)
    terrain_mesh = pv.read(terrain_mesh)
    terrain_mesh = terrain_mesh.compute_normals()
    start = time.time()
    collision, n = robot_mesh.collision(terrain_mesh, contact_mode=2, box_tolerance=0.001, cell_tolerance=0)
    end = time.time()
    print(f"pyvista collision: {(end - start) * 1000}  milliseconds")
    plotter = pv.Plotter()
    scalars = np.zeros(collision.n_cells, dtype=bool)
    scalars[collision.field_data['ContactCells']] = True
    plotter.add_mesh(collision, scalars=scalars, show_scalar_bar=False, cmap='bwr')
    plotter.show()


def get_touch_point_pyvista_difference(path_to_robot_mesh, terrain_mesh):
    robot_mesh = pv.read(path_to_robot_mesh)
    terrain_mesh = pv.read(terrain_mesh)
    terrain_mesh = terrain_mesh.compute_normals(flip_normals=True)
    start = time.time()
    result = terrain_mesh.boolean_difference(robot_mesh)
    end = time.time()
    print(f"pyvista difference: {(end - start) * 1000}  milliseconds")
    plotter = pv.Plotter()
    plotter.add_mesh(result)
    plotter.show()


def get_touch_points_vtk_boolean_intersection(obj_file1, obj_file2):
    def read_obj(file_path):
        reader = vtk.vtkOBJReader()
        reader.SetFileName(file_path)
        reader.Update()
        return reader.GetOutput()

    def flip_normals(input_polydata):
        normals_generator = vtk.vtkPolyDataNormals()
        normals_generator.SetInputData(input_polydata)
        normals_generator.FlipNormalsOn()
        normals_generator.Update()
        return normals_generator.GetOutput()
    polydata1 = read_obj(obj_file1)
    polydata2 = read_obj(obj_file2)
    polydata2 = flip_normals(polydata2)
    def clean_polydata(polydata):
        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputData(polydata)
        cleaner.Update()
        return cleaner.GetOutput()
    polydata1 = clean_polydata(polydata1)
    polydata2 = clean_polydata(polydata2)
    boolean_filter = vtk.vtkBooleanOperationPolyDataFilter()
    # boolean_filter.SetTolerance(0.000000001)
    boolean_filter.SetOperationToIntersection()
    boolean_filter.SetInputData(0, polydata1)
    boolean_filter.SetInputData(1, polydata2)
    start = time.time()
    boolean_filter.Update()
    result = boolean_filter.GetOutput()
    end = time.time()
    print(f"vtk bool intersection: {(end - start) * 1000}  milliseconds")
    writer = vtk.vtkOBJWriter()
    writer.SetFileName("vtk.obj")
    writer.SetInputData(result)
    writer.Write()


def main():
    path_to_robot_mesh = "/home/robert/catkin_ws/src/robot_touch_point_detection/robot_models/husky_transformed_45_angle/transformed_model_simplified2.obj"
    path_to_terrain_mesh = "/home/robert/catkin_ws/src/robot_touch_point_detection/terrain_models/terrain_1/terrain_mesh.obj"
    get_touch_points_meshlib_insideA_or_outsideB(path_to_robot_mesh, path_to_terrain_mesh)




if __name__ == '__main__':
    main()
