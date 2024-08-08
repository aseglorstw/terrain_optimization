import argparse
import os
import sys
import torch
from pytorch3d.loss import point_mesh_face_distance
from pytorch3d.loss import (
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
    mesh_edge_loss
)
from tqdm.auto import tqdm
import device_processer, heigh_map_processer, point_cloud_processer, visualizer
import mesh_processer


def check_input(path_to_mesh):
    if not os.path.isfile(path_to_mesh):
        print(f"The file {path_to_mesh} does not exists")
        sys.exit("Exiting the program due to missing file.")


def optimization_process(robot_point_cloud, init_terrain_mesh, device):
    deform_vertices = torch.full(init_terrain_mesh.verts_packed().shape, 0.0, device=device, requires_grad=True)
    optimizer = torch.optim.SGD([deform_vertices], lr=1.0, momentum=0.9)
    terrain_mesh = None
    for i in tqdm(range(10000), ncols=80, ascii=True, desc='Optimization'):
        optimizer.zero_grad()
        terrain_mesh = init_terrain_mesh.offset_verts(deform_vertices)
        loss_distance_wheels = 0.05 * point_mesh_face_distance(terrain_mesh, robot_point_cloud)
        loss_distance_roof = 0.05 * point_mesh_face_distance(terrain_mesh, robot_point_cloud)
        loss_laplacian = mesh_laplacian_smoothing(terrain_mesh, method="uniform")
        loss_normal = mesh_normal_consistency(terrain_mesh)
        loss_edge = mesh_edge_loss(terrain_mesh)
        loss = 0.5 * loss_distance_wheels + 2 * loss_laplacian + loss_edge + 0.5 * loss_normal
        loss.backward()
        optimizer.step()
    visualizer.visualize_mesh_open3d_with_points(terrain_mesh, robot_point_cloud)
    #mesh_processer.save_mesh(terrain_mesh)



def main(arguments):
    check_input(arguments.mesh_path)
    device = device_processer.choose_device()
    robot_point_cloud = point_cloud_processer.load_point_cloud(device)
    init_height_map = heigh_map_processer.generate_init_height_map(50, 50)
    init_mesh_height_map = heigh_map_processer.height_map_to_mesh(init_height_map, device)
    optimization_process(robot_point_cloud, init_mesh_height_map, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_path", default='', type=str, help="Path to the .obj file")
    main(parser.parse_args())
