import torch
from pytorch3d.transforms import euler_angles_to_matrix, Transform3d


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
    return new_robot_model


def get_transformation_matrix(rotation_angles, translation_vector):
    rotation_tensor = torch.tensor(rotation_angles)
    translation_tensor = torch.tensor(translation_vector)
    rotation_matrix = euler_angles_to_matrix(rotation_tensor, convention="XYZ")
    transformation_matrix = torch.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = translation_tensor
    return transformation_matrix
