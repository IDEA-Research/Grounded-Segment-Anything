import os
import sys

def make_folder(folder_name):
    os.makedirs(folder_name, exist_ok=True)
    # if not os.path.exists(folder_name):
    #     os.makedirs(folder_name)

def add_pypath(path):
    if path not in sys.path:
        sys.path.insert(0, path)

