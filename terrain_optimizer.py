import argparse
import torch
from tools import (
    device_tools,
    height_map_tools,
    point_cloud_tools,
    visualize_tools,
    mesh_tools,
    math_tools
)


def transform_robot_model(robot_model, robot_pose, terrain_shape):
    map_pose = terrain_shape.verts_packed().mean(0)
    robot_vertices = robot_model.verts_packed()
    # print(robot_vertices[:, 2].min().item())
    device = robot_vertices.device
    num_verts = robot_vertices.shape[0]
    homogeneous_robot_vertices = torch.cat([robot_vertices, torch.ones((num_verts, 1), device=device)], dim=1)
    transformed_vertices = torch.matmul(robot_pose.to(device), homogeneous_robot_vertices.T).T
    transformed_vertices = transformed_vertices[:, :3] + map_pose
    new_robot_model = robot_model.update_padded(transformed_vertices.unsqueeze(0))
    visualize_tools.visualize_two_py_torch3d_meshes(new_robot_model, terrain_shape)
    return new_robot_model


def main(arguments):
    device = device_tools.choose_device()
    init_height_map = height_map_tools.generate_init_height_map(51, 51)
    init_mesh_height_map = height_map_tools.height_map_to_mesh(init_height_map, device)
    robot_mesh = mesh_tools.load_mesh("../URDF2mesh/meshes_extracted/husky.obj", device)
    robot_pose = math_tools.get_transformation_matrix([0, 0, 0], [0, 0, 0.1449791043996811])
    transformed_robot_model = transform_robot_model(robot_mesh, robot_pose, init_mesh_height_map)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_path", default='', type=str, help="Path to the .obj file")
    main(parser.parse_args())
