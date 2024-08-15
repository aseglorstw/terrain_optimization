import argparse
import os
import sys
import torch
from pytorch3d.loss import (
    point_mesh_face_distance,
    point_mesh_edge_distance,
    mesh_laplacian_smoothing,
    mesh_edge_loss
)
import meshlib.mrmeshpy as mr
from tqdm.auto import tqdm
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


def optimization_process(wheels_point_cloud, body_point_cloud, init_terrain_mesh, device):
    deform_vertices = torch.full(init_terrain_mesh.verts_packed().shape, 0.0, device=device, requires_grad=True)
    optimizer = torch.optim.SGD([deform_vertices], lr=1.0, momentum=0.9)
    for i in tqdm(range(500), ncols=80, ascii=True, desc='Optimization'):
        optimizer.zero_grad()
        terrain_mesh = init_terrain_mesh.offset_verts(deform_vertices)
        loss_face_distance_wheels = point_mesh_face_distance(terrain_mesh, wheels_point_cloud)
        loss_edge_distance_wheels = point_mesh_edge_distance(terrain_mesh, wheels_point_cloud)
        loss_distance_wheels = loss_face_distance_wheels + loss_edge_distance_wheels
        loss_face_distance_body = point_mesh_face_distance(terrain_mesh, body_point_cloud)
        loss_edge_distance_body = point_mesh_edge_distance(terrain_mesh, body_point_cloud)
        loss_distance_body = loss_face_distance_body + loss_edge_distance_body
        loss_laplacian = mesh_laplacian_smoothing(terrain_mesh, method="uniform")
        loss_edge = mesh_edge_loss(terrain_mesh)
        loss = 0.75 * loss_distance_wheels + 0.5 * loss_laplacian + 1.5 * loss_edge - 0.001 * loss_distance_body
        loss.backward()
        optimizer.step()
        if i % 50 == 0:
            visualize_tools.visualize_mesh_open3d_with_points(
                terrain_mesh, point_cloud_tools.combine_point_clouds([wheels_point_cloud, body_point_cloud]))


def determine_touch_points(robot_model, robot_pose, terrain_shape):
    sparams = mr.SphereParams()
    smallSphere = mr.makeSphere(sparams)
    sparams.numMeshVertices = 400
    sparams.radius = 2
    bigSphere = mr.makeSphere(sparams)
    transVector = mr.Vector3f()
    transVector.z = 2
    diffXf = mr.AffineXf3f.translation(transVector)
    smallSphere.transform(diffXf)
    diff = mr.boolean(bigSphere, smallSphere, mr.BooleanOperation.DifferenceAB)
    mr.saveMesh(diff.mesh, mr.Path("diffSpheres.stl"))


def main(arguments):
    check_input(arguments.mesh_path)
    device = device_tools.choose_device()
    init_height_map = height_map_tools.generate_init_height_map(50, 50)
    init_mesh_height_map = height_map_tools.height_map_to_mesh(init_height_map, device)
    robot_mesh = mesh_tools.load_mesh("../URDF2mesh/meshes_extracted/husky.obj", device)
    robot_pose = math_tools.get_transformation_matrix([0, 0, 0], [0, 0, 0])
    print(robot_pose)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_path", default='', type=str, help="Path to the .obj file")
    main(parser.parse_args())
