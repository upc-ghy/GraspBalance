import torch
from torch.nn.modules import Module
from torch.nn.parallel.scatter_gather import gather
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.nn.parallel import DataParallel,DistributedDataParallel
from torch.nn.parallel._functions import Scatter
from torch.autograd import Variable


def scatter(inputs, target_gpus, dim=0):
    def scatter_map(obj):
        if isinstance(obj, torch.Tensor):
            return Scatter.apply(target_gpus, None, dim, obj)
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            return list_scatter(obj,target_gpus)
        if isinstance(obj, dict) and len(obj) > 0:
            return [type(obj)(i) for i in zip(*map(scatter_map, obj.items()))]
        return [obj for targets in target_gpus]

    try:
        res = scatter_map(inputs)
    finally:
        scatter_map = None
    return res

def list_scatter(input, devices, streams=None):

    chunk_size = (len(input) - 1) // len(devices) + 1
    for i in range(len(input)):
        for j in range(len(input[i])):
            input[i][j] = input[i][j].to(devices[i//chunk_size])
    outputs = [
        input[i*chunk_size:(i+1)*chunk_size] for i in range(len(devices))
    ]
    return outputs


def scatter_kwargs(inputs, kwargs, target_gpus, dim=0):
    inputs = scatter(inputs, target_gpus, dim) if inputs else []
    kwargs = scatter(kwargs, target_gpus, dim) if kwargs else []
    if len(inputs) < len(kwargs):
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
    elif len(kwargs) < len(inputs):
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs

class ListDataParallel(DataParallel):
    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

class DDP(DistributedDataParallel):
    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)
