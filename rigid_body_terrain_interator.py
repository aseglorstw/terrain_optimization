import torch
from pytorch3d.io import load_objs_as_meshes
import warnings
warnings.filterwarnings("ignore")



def robot_terrain_iterate(robot_config, initial_robot_state, terrain_config, simulation_config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    robot_mesh = get_robot_mesh(robot_config, initial_robot_state, device)


def get_robot_mesh(robot_config, initial_robot_state, device):
    robot_mesh = load_objs_as_meshes([robot_config["mesh_file_path"]], device)
    translation_vector = torch.tensor(initial_robot_state["translation_vector"]).to(device)
    rotation_matrix = torch.tensor(initial_robot_state["rotation_matrix"]).to(device)
    vertices = robot_mesh.verts_packed()
    rotated_vertices = torch.matmul(vertices, rotation_matrix.T)
    translated_vertices = rotated_vertices + translation_vector
    robot_mesh.offset_verts_(translated_vertices - vertices)
    return robot_mesh


if __name__ == '__main__':
    robot_config = {
                    "weight": 10.0,
                    "inertia_matrix": 100. * torch.eye(3),
                    "mesh_file_path": "robot_models/husky/husky.obj"
                   }

    initial_robot_state = {
                            "translation_vector": torch.tensor([1.0, 0.0, 3.0]),
                            "rotation_matrix": torch.eye(3),
                            "velocity": torch.tensor([0.0, 0.0, 0.0]),
                            "angle_velocity": torch.tensor([0.0, 0.0, 0.0])
                           }

    terrain_config = {
                      "terrain_file_path": "terrain_models/terrain_1/terrain_mesh_2.obj",
                      "k_stiffness": 1000.0,
                      "k_friction": 0.02
                     }

    simulation_config = {
                        "time_step_size": 0.01,
                        "simulation_time": 5.0,
                        "free_fall_acceleration": 9.81
                        }

    robot_terrain_iterate(robot_config, initial_robot_state, terrain_config, simulation_config)
