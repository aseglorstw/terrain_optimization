import argparse
import torch
import meshlib.mrmeshpy as mr
from tools import (
    device_tools,
    height_map_tools,
    point_cloud_tools,
    visualize_tools,
    mesh_tools,
    math_tools
)
import time


def determine_touch_points():
    mesh = mr.loadMesh("experiments/transformed_robot.obj")
    terrain_mesh = mr.loadMesh("experiments/terrain_mesh.obj")
    start = time.time()
    diff = mr.boolean(terrain_mesh, mesh, mr.BooleanOperation.Intersection)
    end = time.time()
    print(end - start)
    mr.saveMesh(diff.mesh, mr.Path("intersection.obj"))


def main(arguments):
    determine_touch_points()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_path", default='', type=str, help="Path to the .obj file")
    main(parser.parse_args())
