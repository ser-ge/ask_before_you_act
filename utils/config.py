import yaml

from utils import Config

def load_yaml_config(path_to_yaml):
    try:
        with open (path_to_yaml, 'r') as file:
            config = yaml.safe_load(file)
    except Exception as e:
        print('Error reading the config file')

    return Config(**config)
