import os
import torch

def create_folders_if_necessary(path):
    dirname = os.path.dirname(path)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)

def get_storage_dir():
    if "RL_STORAGE" in os.environ:
        return os.environ["RL_STORAGE"]
    return "storage"

def get_model_dir(model_name):
    return os.path.join(get_storage_dir(), model_name)

def get_status_path(model_dir):
    return os.path.join(model_dir, "status.pt")

def get_status(model_dir):
    path = get_status_path(model_dir)
    return torch.load(path)

def save_status(status, model_dir):
    path = get_status_path(model_dir)
    create_folders_if_necessary(path)
    torch.save(status, path)

def get_config(model_dir):
    return get_status(model_dir)["config"]

def get_model_state(model_dir):
    return get_status(model_dir)["model_state"]