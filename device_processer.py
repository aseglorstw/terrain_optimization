import torch


def choose_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        print("WARNING: CPU only, this will be slow!")
    return device


def is_device_GPU(device):
    return device.type == 'cuda'
