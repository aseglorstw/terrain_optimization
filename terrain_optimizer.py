import argparse
import os
import sys
import torch
import numpy as np
from pytorch3d.loss import (
    point_mesh_face_distance,
    point_mesh_edge_distance,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
    mesh_edge_loss
)
from tqdm.auto import tqdm
import device_processer, heigh_map_processer, point_cloud_processer, visualizer
import mesh_processer
from pytorch3d.structures import Pointclouds


def check_input(path_to_mesh):
    if not os.path.isfile(path_to_mesh):
        print(f"The file {path_to_mesh} does not exists")
        sys.exit("Exiting the program due to missing file.")


def optimization_process(wheels_point_cloud, roof_point_cloud, init_terrain_mesh, device):
    deform_vertices = torch.full(init_terrain_mesh.verts_packed().shape, 0.0, device=device, requires_grad=True)
    optimizer = torch.optim.SGD([deform_vertices], lr=1.0, momentum=0.9)
    for i in tqdm(range(500), ncols=80, ascii=True, desc='Optimization'):
        optimizer.zero_grad()
        terrain_mesh = init_terrain_mesh.offset_verts(deform_vertices)
        loss_face_distance_wheels = point_mesh_face_distance(terrain_mesh, wheels_point_cloud)
        loss_face_distance_roof = point_mesh_face_distance(terrain_mesh, roof_point_cloud)
        loss_edge_distance_wheels = point_mesh_edge_distance(terrain_mesh, wheels_point_cloud)
        loss_edge_distance_roof = point_mesh_edge_distance(terrain_mesh, roof_point_cloud)
        loss_laplacian = mesh_laplacian_smoothing(terrain_mesh, method="uniform")
        loss_edge = mesh_edge_loss(terrain_mesh)
        loss = 0.75 * (loss_face_distance_wheels + loss_edge_distance_wheels) + 0.5 * loss_laplacian + 1.5 * loss_edge - 0.001 * (loss_face_distance_roof + loss_edge_distance_roof)
        loss.backward()
        optimizer.step()
        if i % 50 == 0 and i >= 200:
            visualizer.visualize_mesh_open3d_with_points(terrain_mesh, point_cloud_processer.combine_point_clouds([wheels_point_cloud, roof_point_cloud]))


def main(arguments):
    check_input(arguments.mesh_path)
    device = device_processer.choose_device()
    rotation_matrix = np.array([
        [np.cos(-45), 0, np.sin(-45)],
        [0, 1, 0],
        [-np.sin(-45), 0, np.cos(-45)]
    ])
    wheels_point_cloud = point_cloud_processer.load_point_cloud(device, "point_clouds/test_robot_without_roof.npy", [30, 30, 0], rotation_matrix)
    roof_point_cloud = point_cloud_processer.load_point_cloud(device, "point_clouds/test_robot_roof.npy", [30, 30, 1], rotation_matrix)
    init_height_map = heigh_map_processer.generate_init_height_map(50, 50)
    init_mesh_height_map = heigh_map_processer.height_map_to_mesh(init_height_map, device)
    optimization_process(wheels_point_cloud, roof_point_cloud, init_mesh_height_map, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_path", default='', type=str, help="Path to the .obj file")
    main(parser.parse_args())
