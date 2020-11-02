import os
import json
import torch

__all__ = ["get_gpu_prop", "collect_gpu_info"]


dirname = os.path.dirname(__file__)


def get_gpu_devices_count():
    return torch.cuda.device_count()


def get_gpu_prop(show=True):
    ngpus = torch.cuda.device_count()

    properties = []
    for dev in range(ngpus):
        prop = torch.cuda.get_device_properties(dev)
        properties.append({
            "name": prop.name,
            "capability": [prop.major, prop.minor],
            # unit GB
            "total_momory": round(prop.total_memory / 1073741824, 2),
            "sm_count": prop.multi_processor_count
        })

    if show:
        print("cuda: {}".format(torch.cuda.is_available()))
        print("available GPU(s): {}".format(ngpus))
        for i, p in enumerate(properties):
            print("{}: {}".format(i, p))
    return properties


def sort(d, tmp={}):
    for k in sorted(d.keys()):
        if isinstance(d[k], dict):
            tmp[k] = {}
            sort(d[k], tmp[k])
        else:
            tmp[k] = d[k]
    return tmp
