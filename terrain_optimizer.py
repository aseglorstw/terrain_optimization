import argparse
import os
import sys
from tools import (
    device_tools,
    height_map_tools,
    point_cloud_tools,
    visualize_tools,
    mesh_tools,
    math_tools
)


def check_input(path_to_mesh):
    if not os.path.isfile(path_to_mesh):
        print(f"The file {path_to_mesh} does not exists")
        sys.exit("Exiting the program due to missing file.")


def determine_touch_points(robot_model, robot_pose, terrain_shape):
    map_pose = terrain_shape.verts_packed().mean(0)
    robot_vertices = robot_model.verts_packed()
    translated_vertices = robot_vertices + map_pose
    new_robot_model = robot_model.update_padded(translated_vertices.unsqueeze(0))
    visualize_tools.visualize_two_py_torch3d_meshes(terrain_shape, new_robot_model)
    return new_robot_model


def main(arguments):
    check_input(arguments.mesh_path)
    device = device_tools.choose_device()
    init_height_map = height_map_tools.generate_init_height_map(50, 50)
    init_mesh_height_map = height_map_tools.height_map_to_mesh(init_height_map, device)
    robot_mesh = mesh_tools.load_mesh("../URDF2mesh/meshes_extracted/husky.obj", device)
    robot_pose = math_tools.get_transformation_matrix([0, 0, 0], [0, 0, 0])
    determine_touch_points(robot_mesh, robot_pose, init_mesh_height_map)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_path", default='', type=str, help="Path to the .obj file")
    main(parser.parse_args())
