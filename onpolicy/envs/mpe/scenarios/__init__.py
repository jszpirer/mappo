import importlib.util
import importlib.machinery
import os.path as osp

def load(name):
    pathname = osp.join(osp.dirname(__file__), name)
    loader = importlib.machinery.SourceFileLoader('', pathname)
    spec = importlib.util.spec_from_file_location('', pathname, loader=loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module



