import torch


def robot_terrain_iterate(robot_config, initial_robot_state, terrain_config, simulation_config):
    pass


if __name__ == '__main__':
    robot_config = {
                    "weight": 10.0,
                    "inertia_matrix": 100. * torch.eye(3),
                    "mesh_file_path": "robot_models/husky/husky.obj"
                   }

    initial_robot_state = {
                            "translation": torch.tensor([1.0, 0.0, 3.0]),
                            "rotation": torch.eye(3),
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

