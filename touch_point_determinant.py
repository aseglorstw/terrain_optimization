import meshlib.mrmeshpy as mr
import os
import time


def determine_touch_points_meshlib():
    mesh = mr.loadMesh("/home/robert/catkin_ws/src/robot_touch_point_detection/experiments/transformed_robot.obj")
    terrain_mesh = mr.loadMesh("/home/robert/catkin_ws/src/robot_touch_point_detection/experiments/terrain_mesh.obj")
    start = time.time()
    intersection = mr.boolean(terrain_mesh, mesh, mr.BooleanOperation.Intersection)
    end = time.time()
    print(end - start)
    mr.saveMesh(intersection.mesh, "/home/robert/catkin_ws/src/robot_touch_point_detection/experiments/meshlib/intersection.obj")


def main():
    determine_touch_points_meshlib()


if __name__ == '__main__':
    main()
