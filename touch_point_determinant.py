import meshlib.mrmeshpy as mr
import torch
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex
from tools import mesh_tools
import pyvista as pv
import numpy as np
from meshlib import mrmeshnumpy as mn
import time


def get_touch_points_meshlib_insideA_or_outsideB(path_to_robot_mesh, terrain_mesh, output_path):
    robot_mesh = mr.loadMesh(path_to_robot_mesh)
    terrain_mesh = mr.loadMesh(terrain_mesh)
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


def main():
    path_to_robot_mesh = "/home/robert/catkin_ws/src/robot_touch_point_detection/robot_models/husky_transformed_45_angle/transformed_model_simplified.obj"
    path_to_terrain_mesh = "/home/robert/catkin_ws/src/robot_touch_point_detection/terrain_models/terrain_1/terrain_mesh.obj"
    output_path = "/home/robert/catkin_ws/src/robot_touch_point_detection/result.obj"
    get_touch_points_meshlib_insideA_or_outsideB(path_to_robot_mesh, path_to_terrain_mesh, output_path)
    # get_touch_points_pyvista_intersection(path_to_robot_mesh, path_to_terrain_mesh)
    # get_touch_points_pyvista_boolean_intersection(path_to_robot_mesh, path_to_terrain_mesh)
    # get_touch_points_pyvista_collision(path_to_robot_mesh, path_to_terrain_mesh)
    # get_touch_point_pyvista_difference(path_to_robot_mesh, path_to_terrain_mesh)



if __name__ == '__main__':
    main()
