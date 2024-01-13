import unittest
import gc
import operator as op
import functools
import torch
from torch.autograd import Variable, Function
# from knn_pytorch import knn_pytorch
from vision_sandbox import _C
knn = _C.knn

# import knn_pytorch
def myknn(ref, query, k=1):
  """ Compute k nearest neighbors for each query point.
  """
  device = ref.device
  ref = ref.float().to(device)
  query = query.float().to(device)
  inds = torch.empty(query.shape[0], 1, query.shape[2]).long().to(device)
  knn(ref, query, inds)
  return inds