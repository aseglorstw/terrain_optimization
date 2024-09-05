import meshlib.mrmeshpy as mr
import pyvista as pv
import numpy as np
from meshlib import mrmeshnumpy as mn
from scipy.spatial import cKDTree
import argparse


def main(arguments):
    robot_mesh_pv = pv.read(arguments.robot_mesh)
    terrain_mesh_pv = pv.read(arguments.terrain_mesh)

    terrain_touch_cells, terrain_touch_cells_normals = process_boolean_intersection(arguments, robot_mesh_pv, terrain_mesh_pv)


def process_boolean_intersection(arguments, robot_mesh_pv, terrain_mesh_pv):
    intersection_vertices, intersection_faces = compute_boolean_intersection(arguments.robot_mesh, arguments.terrain_mesh)
    intersection_mesh_pv = get_py_vista_mesh(intersection_vertices, intersection_faces)

    terrain_touch_cells, terrain_touch_cells_normals = find_touch_points_and_normals(intersection_mesh_pv, terrain_mesh_pv)

    visualize_boolean_intersection(robot_mesh_pv, terrain_mesh_pv, intersection_mesh_pv, intersection_mesh_pv.cell_centers().points, terrain_touch_cells, terrain_touch_cells_normals)

    return terrain_touch_cells, terrain_touch_cells_normals


def compute_boolean_intersection(path_to_robot_mesh, path_to_terrain_mesh):
    robot_mesh = mr.loadMesh(path_to_robot_mesh)
    terrain_mesh = mr.loadMesh(path_to_terrain_mesh)
    first_normal = terrain_mesh.normal(mr.FaceId())
    boolean_operation = mr.BooleanOperation.OutsideA if first_normal.z < 0 else mr.BooleanOperation.InsideA
    intersection_mesh = mr.boolean(robot_mesh, terrain_mesh, boolean_operation).mesh
    vertices = mn.getNumpyVerts(intersection_mesh)
    faces = mn.getNumpyFaces(intersection_mesh.topology)
    return vertices, faces


def find_touch_points_and_normals(intersection_mesh_pv, terrain_mesh_pv):
    intersection_cells = intersection_mesh_pv.cell_centers().points
    terrain_cells = terrain_mesh_pv.cell_centers().points
    nearest_cells, nearest_cells_indices = find_nearest_neighbors(intersection_cells, terrain_cells)
    terrain_mesh_pv.compute_normals(cell_normals=True, point_normals=False)
    nearest_cells_normals = terrain_mesh_pv.cell_normals[nearest_cells_indices]
    positive_nearest_cells_normals = flip_negative_normals(nearest_cells_normals)
    return nearest_cells, positive_nearest_cells_normals


def find_nearest_neighbors(robot_cells, terrain_cells):
    kd_tree = cKDTree(terrain_cells)
    distances, indices = kd_tree.query(robot_cells)
    nearest_points = terrain_cells[indices]
    return nearest_points, indices


def flip_negative_normals(normals):
    flip_indices = np.where(normals[:, 2] < 0)[0]
    normals[flip_indices] *= -1
    return normals


def get_py_vista_mesh(vertices, faces):
    faces_with_size = np.hstack([np.full((faces.shape[0], 1), 3), faces]).flatten()
    mesh = pv.PolyData(vertices, faces_with_size)
    return mesh


def visualize_boolean_intersection(robot_mesh, terrain_mesh, intersection_mesh, robot_cells, nearest_terrain_cells, terrain_normals):
    plotter = pv.Plotter()
    plotter.add_mesh(robot_mesh, style='wireframe', color='yellow')
    plotter.add_mesh(terrain_mesh, style='wireframe', color='purple')
    plotter.add_mesh(intersection_mesh, show_edges=True, color='red')
    plotter.add_points(robot_cells, color='green', point_size=8)
    plotter.add_arrows(nearest_terrain_cells, terrain_normals, color='blue', mag=0.05)
    plotter.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot_mesh", default='', type=str)
    parser.add_argument("--terrain_mesh", default='', type=str)
    main(parser.parse_args())
