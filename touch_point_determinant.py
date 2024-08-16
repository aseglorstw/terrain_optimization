import meshlib.mrmeshpy as mr
from tools import mesh_tools
from tools import visualize_tools
from tools import device_tools
from tools import math_tools
from math import pi
import time


def determine_touch_points_meshlib(path_to_robot_mesh, terrain_mesh, output_path):
    robot_mesh = mr.loadMesh(path_to_robot_mesh)
    terrain_mesh = mr.loadMesh(terrain_mesh)
    start = time.time()
    intersection = mr.boolean(robot_mesh, terrain_mesh, mr.BooleanOperation.OutsideA)
    end = time.time()
    print(end - start)
    mr.saveMesh(intersection.mesh, output_path)


def main():
    path_to_robot_mesh = "/home/robert/catkin_ws/src/robot_touch_point_detection/experiments/experiment_5/transformed_robot.obj"
    path_to_terrain_mesh = "/home/robert/catkin_ws/src/robot_touch_point_detection/experiments/experiment_5/terrain_mesh.obj"
    output_path = "/home/robert/catkin_ws/src/robot_touch_point_detection/experiments/experiment_5/meshlib/inside.stl"
    # determine_touch_points_meshlib(path_to_robot_mesh, path_to_terrain_mesh, output_path)
    # device = device_tools.choose_device()
    # robot_mesh = mesh_tools.load_mesh("/home/robert/catkin_ws/src/URDF2mesh/meshes_extracted/husky.obj", device)
    # terrain_mesh = mesh_tools.load_mesh(path_to_terrain_mesh, device)
    # transform_matrix = math_tools.get_transformation_matrix([0, -pi / 4, 0], [0.001, 0, 0.05])
    # transformed_robot = math_tools.transform_robot_model(robot_mesh, transform_matrix, terrain_mesh)
    # mesh_tools.save_mesh(transformed_robot, "/home/robert/catkin_ws/src/robot_touch_point_detection/experiments/experiment_5/transformed_robot.obj")
    # mesh_tools.simplify_mesh_and_save("/home/robert/catkin_ws/src/robot_touch_point_detection/experiments/experiment_1/transformed_robot.stl", "/home/robert/catkin_ws/src/robot_touch_point_detection/experiments/experiment_4/transformed_robot_simplified.stl", 0.99)
    # mesh_tools.generate_terrain_mesh_and_save(output_path)
    determine_touch_points_meshlib(path_to_robot_mesh, path_to_terrain_mesh, output_path)
    # visualize_tools.visualize_two_meshes(output_path, path_to_robot_mesh)


if __name__ == '__main__':
    main()
