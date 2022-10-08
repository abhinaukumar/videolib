import os
from types import ModuleType
import numpy as np


# Borrowed from Netflix/vmaf.
def import_python_file(filepath: str) -> ModuleType:
    '''
    Import a python file as a module.

    Args:
        filepath: Path to Python file.

    Returns:
        ModuleType: Loaded module.
    '''
    filename = os.path.basename(filepath).rsplit('.', 1)[0]
    try:
        from importlib.machinery import SourceFileLoader
        ret = SourceFileLoader(filename, filepath).load_module()
    except ImportError:
        import imp
        ret = imp.load_source(filename, filepath)
    return ret


def apply_transfer_mat(img: np.ndarray, transfer_mat: np.ndarray) -> np.ndarray:
    '''
    Apply given color transformation matrix to data.

    Args:
        img: Data to be transformed.
        transfer_mat: Transfer matrix to be applied.

    Returns:
        np.ndarray: Transformed data.
    '''
    return img @ transfer_mat.T
