import os

module_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(module_dir, os.pardir, os.pardir))
DATA_DIR = os.path.join(project_root, "resources")
os.makedirs(DATA_DIR, exist_ok=True)


def get_resource_path(filename):
    return os.path.join(DATA_DIR, filename)
