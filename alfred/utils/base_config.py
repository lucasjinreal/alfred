
import importlib
from yacs.config import CfgNode as CN
import argparse


class Config:
    @classmethod
    def load_from_args(cls):
        parser = argparse.ArgumentParser()
        parser.add_argument('--cfg', type=str, default='config/base.yml')
        parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
        args = parser.parse_args()
        return cls.load(filename=args.cfg, opts=args.opts)

    @classmethod
    def load(cls, filename=None, opts=[]) -> CN:
        cfg = CN()
        cfg = cls.init(cfg)
        if filename is not None:
            cfg.merge_from_file(filename)
        if len(opts) > 0:
            cfg.merge_from_list(opts)
        cls.parse(cfg)
        cls.print(cfg)
        return cfg

    @staticmethod
    def init(cfg):
        return cfg

    @staticmethod
    def parse(cfg):
        pass

    @staticmethod
    def print(cfg):
        print('[Info] --------------')
        print('[Info] Configuration:')
        print('[Info] --------------')
        print(cfg)


def load_object(module_name, module_args):
    module_path = '.'.join(module_name.split('.')[:-1])
    module = importlib.import_module(module_path)
    name = module_name.split('.')[-1]
    obj = getattr(module, name)(**module_args)
    return obj
