import meshlib.mrmeshpy as mr
import torch
from pytorch3d.structures import Meshes
import pyvista as pv
import numpy as np
from meshlib import mrmeshnumpy as mn
import time
from scipy.spatial import cKDTree


def compute_intersection(path_to_robot_mesh, path_to_terrain_mesh):
    robot_mesh_mr = mr.loadMesh(path_to_robot_mesh)
    terrain_mesh_mr = mr.loadMesh(path_to_terrain_mesh)
    first_normal = terrain_mesh_mr.normal(mr.FaceId())
    boolean_operation = mr.BooleanOperation.OutsideA if first_normal.z < 0 else mr.BooleanOperation.InsideA
    intersection_mesh = mr.boolean(robot_mesh_mr, terrain_mesh_mr, boolean_operation).mesh
    vertices = mn.getNumpyVerts(intersection_mesh)
    faces = mn.getNumpyFaces(intersection_mesh.topology)
    return vertices, faces

def find_touch_points_and_normals(intersection_mesh_pv, terrain_mesh_pv):
    intersection_cells = intersection_mesh_pv.cell_centers().points
    terrain_cells = terrain_mesh_pv.cell_centers().points
    nearest_cells, nearest_cells_indices = find_nearest_neighbors(intersection_cells, terrain_cells)
    terrain_mesh_pv.flip_normals()
    nearest_cells_normals = terrain_mesh_pv.cell_normals[nearest_cells_indices]
    return nearest_cells, nearest_cells_normals

def get_py_vista_mesh(vertices, faces):
    faces_with_size = np.hstack([np.full((faces.shape[0], 1), 3), faces]).flatten()
    mesh = pv.PolyData(vertices, faces_with_size)
    return mesh


def find_nearest_neighbors(robot_cells, terrain_cells):
    kd_tree = cKDTree(terrain_cells)
    start = time.time()
    distances, indices = kd_tree.query(robot_cells)
    end = time.time()
    print(f"finding nearest points: {(end - start) * 1000} milliseconds")
    nearest_points = terrain_cells[indices]
    return nearest_points, indices


def visualize(robot_mesh, terrain_mesh, intersection_mesh, robot_cells, nearest_terrain_cells, terrain_normals):
    plotter = pv.Plotter()
    plotter.add_mesh(robot_mesh, style='wireframe', color='yellow')
    plotter.add_mesh(terrain_mesh, style='wireframe', color='purple')
    plotter.add_mesh(intersection_mesh, show_edges=True, color='red')
    plotter.add_points(robot_cells, color='green', point_size=8)
    plotter.add_arrows(nearest_terrain_cells, terrain_normals, color='blue', mag=0.05)
    plotter.show()


def main():
    path_to_robot_mesh = "/home/robert/catkin_ws/src/robot_touch_point_detection/robot_models/husky_transformed_45_angle/transformed_model_simplified2.obj"
    path_to_terrain_mesh = "/home/robert/catkin_ws/src/robot_touch_point_detection/terrain_models/terrain_1/terrain_mesh_2.obj"

    robot_mesh_pv = pv.read(path_to_robot_mesh)
    terrain_mesh_pv = pv.read(path_to_terrain_mesh)

    intersection_vertices, intersection_faces = compute_intersection(path_to_robot_mesh, path_to_terrain_mesh)
    intersection_mesh_pv = get_py_vista_mesh(intersection_vertices, intersection_faces)
    terrain_touch_cells, terrain_touch_cells_normals = find_touch_points_and_normals(intersection_mesh_pv, terrain_mesh_pv)

    visualize(robot_mesh_pv, terrain_mesh_pv, intersection_mesh_pv, intersection_mesh_pv.cell_centers().points, terrain_touch_cells, terrain_touch_cells_normals)


if __name__ == '__main__':
    main()
