import meshlib.mrmeshpy as mr
import torch
from pytorch3d.structures import Meshes
import pyvista as pv
import numpy as np
from meshlib import mrmeshnumpy as mn
import time
from scipy.spatial import cKDTree


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
    intersection_mesh = pv.PolyData(vertices, faces_with_size)
    intersection_mesh_pv = intersection_mesh.compute_normals(flip_normals=True)

    robot_mesh_pv = pv.read(path_to_robot_mesh)
    terrain_mesh_pv = pv.read(path_to_terrain_mesh)

    robot_cells = intersection_mesh_pv.cell_centers().points
    terrain_cells = terrain_mesh_pv.cell_centers().points

    nearest_points, nearest_indices = find_nearest_neighbors(robot_cells, terrain_cells)
    terrain_mesh_pv.flip_normals()
    nearest_normals = terrain_mesh_pv.cell_normals[nearest_indices]

    visualize(robot_mesh_pv, terrain_mesh_pv, intersection_mesh_pv, robot_cells, nearest_points, nearest_normals)

def find_nearest_neighbors(robot_cells, terrain_cells):
    kd_tree = cKDTree(terrain_cells)
    start = time.time()
    distances, indices = kd_tree.query(robot_cells)
    end = time.time()
    print(f"finding nearest points: {end - start}")
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
    path_to_terrain_mesh = "/home/robert/catkin_ws/src/robot_touch_point_detection/terrain_models/terrain_1/terrain_mesh.obj"
    get_touch_points_meshlib_insideA_or_outsideB(path_to_robot_mesh, path_to_terrain_mesh)


if __name__ == '__main__':
    main()
