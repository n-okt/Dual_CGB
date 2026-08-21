import os
from glob import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from importlib import import_module

def load_modules(path, cls_name):
    files = glob(f'{path}/*.py')
    root = path.replace('/', '.')

    modules = [
        import_module(f'{root}.{os.path.splitext(os.path.basename(file_))[0]}')
        for file_ in files
    ]

    for module in modules:
        cls_ = getattr(module, cls_name, None)
        if cls_ is not None: 
            break

    if cls_ is None: 
        raise ValueError(f'Model {cls_name} is not found.')

    return cls_

def create_model(model_name, kwargs):
    model_cls = load_modules('models', model_name)
    return model_cls(**kwargs)