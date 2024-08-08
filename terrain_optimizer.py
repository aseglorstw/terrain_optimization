import argparse
import os
import sys
import torch
from pytorch3d.structures import Meshes
from pytorch3d.loss import point_mesh_face_distance
from pytorch3d.structures import Pointclouds
from pytorch3d.loss import (
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)
from tqdm.auto import tqdm
import device_processer, heigh_map_processer, point_cloud_processer, visualizer


def check_input(path_to_mesh):
    if not os.path.isfile(path_to_mesh):
        print(f"The file {path_to_mesh} does not exists")
        sys.exit("Exiting the program due to missing file.")


def optimization_process(robot_point_cloud, terrain_mesh, device):
    if len(terrain_mesh.verts_packed().shape) == 2:
        terrain_mesh = Meshes(verts=[terrain_mesh.verts_packed()], faces=[terrain_mesh.faces_packed()])
    if len(robot_point_cloud.shape) == 2:
        robot_point_cloud = Pointclouds(points=[robot_point_cloud])
    deform_vertices = torch.full(terrain_mesh.verts_packed().shape, 0.0, device=device, requires_grad=True)
    optimizer = torch.optim.SGD([deform_vertices], lr=1.0, momentum=0.9)
    loop = tqdm(range(20000))
    new_src_mesh = None
    for i in tqdm(range(1000), ncols=80, ascii=True, desc='Total'):
        optimizer.zero_grad()
        new_src_mesh = terrain_mesh.offset_verts(deform_vertices)
        loss_distance = 0.05 * point_mesh_face_distance(new_src_mesh, robot_point_cloud)
        loss_laplacian = mesh_laplacian_smoothing(new_src_mesh, method="uniform")
        loss_normal = mesh_normal_consistency(new_src_mesh)
        loss = 0.1 * loss_distance + 0.8 * loss_laplacian
        loss.backward()
        optimizer.step()
    visualizer.visualize_mesh_open3d_with_points(new_src_mesh, robot_point_cloud)


def main(arguments):
    check_input(arguments.mesh_path)
    device = device_processer.choose_device()
    robot_point_cloud = point_cloud_processer.load_point_cloud(device)
    height_map = heigh_map_processer.generate_init_height_map(10, 10)
    mesh_height_map = heigh_map_processer.height_map_to_mesh(height_map, device)
    optimization_process(robot_point_cloud, mesh_height_map, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_path", default='', type=str, help="Path to the .obj file")
    main(parser.parse_args())
