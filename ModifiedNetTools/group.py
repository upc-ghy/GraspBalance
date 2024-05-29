import os
import sys
from typing import Tuple
import copy, logging
import torch
import torch.nn as nn
from torch.autograd import Function

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2_batch'))
import pointnet2_batch_cuda as pointnet2_cuda

class KNN(nn.Module):
    def __init__(self, neighbors, transpose_mode=True):
        super(KNN, self).__init__()
        self.neighbors = neighbors

    @torch.no_grad()
    def forward(self, support, query):
        dist = torch.cdist(support, query)
        k_dist = dist.topk(k=self.neighbors, dim=1, largest=False)
        return k_dist.values, k_dist.indices.transpose(1, 2).contiguous().int()

class DenseDilated(nn.Module):
    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super(DenseDilated, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k

    def forward(self, edge_index):
        if self.stochastic:
            if torch.rand(1) < self.epsilon and self.training:
                num = self.k * self.dilation
                randnum = torch.randperm(num)[:self.k]
                edge_index = edge_index[:, :, randnum]
            else:
                edge_index = edge_index[:, :, ::self.dilation]
        else:
            edge_index = edge_index[:, :, ::self.dilation]
        return edge_index.contiguous()


class DilatedKNN(nn.Module):
    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super(DilatedKNN, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k
        self._dilated = DenseDilated(k, dilation, stochastic, epsilon)
        self.knn = KNN(k * self.dilation, transpose_mode=True)

    def forward(self, query):
        _, idx = self.knn(query, query)
        return self._dilated(idx)


class GroupingOperation(Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        assert features.is_contiguous()
        assert idx.is_contiguous()

        B, nfeatures, nsample = idx.size()
        _, C, N = features.size()
        output = torch.cuda.FloatTensor(B, C, nfeatures, nsample, device=features.device)

        pointnet2_cuda.group_points_wrapper(B, C, N, nfeatures, nsample, features, idx, output)

        ctx.for_backwards = (idx, N)
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        idx, N = ctx.for_backwards

        B, C, npoint, nsample = grad_out.size()
        grad_features = torch.zeros([B, C, N], dtype=torch.float, device=grad_out.device, requires_grad=True)
        grad_out_data = grad_out.data.contiguous()
        pointnet2_cuda.group_points_grad_wrapper(B, C, N, npoint, nsample, grad_out_data, idx, grad_features.data)
        return grad_features, None


grouping_operation = GroupingOperation.apply


def torch_grouping_operation(features, idx):
    all_idx = idx.reshape(idx.shape[0], -1)
    all_idx = all_idx.unsqueeze(1).repeat(1, features.shape[1], 1)
    grouped_features = features.gather(2, all_idx)
    return grouped_features.reshape(idx.shape[0], features.shape[1], idx.shape[1], idx.shape[2])


class GatherOperation(Function):
    @staticmethod
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        assert features.is_contiguous()
        assert idx.is_contiguous()

        B, npoint = idx.size()
        _, C, N = features.size()
        output = torch.cuda.FloatTensor(B, C, npoint, device=features.device)

        pointnet2_cuda.gather_points_wrapper(B, C, N, npoint, features, idx, output)

        ctx.for_backwards = (idx, C, N)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        idx, C, N = ctx.for_backwards
        B, npoint = idx.size()

        grad_features = torch.zeros([B, C, N], dtype=torch.float, device=grad_out.device, requires_grad=True)
        grad_out_data = grad_out.data.contiguous()
        pointnet2_cuda.gather_points_grad_wrapper(B, C, N, npoint, grad_out_data, idx, grad_features.data)
        return grad_features, None


gather_operation = GatherOperation.apply


class BallQuery(Function):
    @staticmethod
    def forward(ctx, radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
        assert new_xyz.is_contiguous()
        assert xyz.is_contiguous()

        B, N, _ = xyz.size()
        npoint = new_xyz.size(1)
        idx = torch.cuda.IntTensor(B, npoint, nsample, device=xyz.device).zero_()
        pointnet2_cuda.ball_query_wrapper(B, N, npoint, radius, nsample, new_xyz, xyz, idx)
        return idx

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


ball_query = BallQuery.apply

class QueryAndGroup(nn.Module):
    def __init__(self, radius: float, nsample: int,
                 relative_xyz=True,
                 normalize_dp=False,
                 normalize_by_std=False,
                 normalize_by_allstd=False,
                 normalize_by_allstd2=False,
                 return_only_idx=False,
                 **kwargs
                 ):
        super().__init__()
        self.radius, self.nsample = radius, nsample
        self.normalize_dp = normalize_dp
        self.normalize_by_std = normalize_by_std
        self.normalize_by_allstd = normalize_by_allstd
        self.normalize_by_allstd2 = normalize_by_allstd2
        assert self.normalize_dp + self.normalize_by_std + self.normalize_by_allstd < 2
        self.relative_xyz = relative_xyz
        self.return_only_idx = return_only_idx

    def forward(self, query_xyz: torch.Tensor, support_xyz: torch.Tensor, features: torch.Tensor = None) -> Tuple[
        torch.Tensor]:
        idx = ball_query(self.radius, self.nsample, support_xyz, query_xyz)

        if self.return_only_idx:
            return idx
        xyz_trans = support_xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)
        if self.relative_xyz:
            grouped_xyz = grouped_xyz - query_xyz.transpose(1, 2).unsqueeze(-1)
            if self.normalize_dp:
                grouped_xyz /= self.radius
        grouped_features = grouping_operation(features, idx) if features is not None else None
        return grouped_xyz, grouped_features


class GroupAll(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, new_xyz: torch.Tensor, xyz: torch.Tensor, features: torch.Tensor = None):
        grouped_xyz = xyz.transpose(1, 2).unsqueeze(2)
        grouped_features = features.unsqueeze(2) if features is not None else None
        return grouped_xyz, grouped_features


class KNNGroup(nn.Module):
    def __init__(self, nsample: int,
                 relative_xyz=True,
                 normalize_dp=False,
                 return_only_idx=False,
                 **kwargs
                 ):
        super().__init__()
        self.nsample = nsample
        self.knn = KNN(nsample, transpose_mode=True)
        self.relative_xyz = relative_xyz
        self.normalize_dp = normalize_dp
        self.return_only_idx = return_only_idx

    def forward(self, query_xyz: torch.Tensor, support_xyz: torch.Tensor, features: torch.Tensor = None) -> Tuple[
        torch.Tensor]:
        _, idx = self.knn(support_xyz, query_xyz)
        if self.return_only_idx:
            return idx
        idx = idx.int()
        xyz_trans = support_xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)
        if self.relative_xyz:
            grouped_xyz -= query_xyz.transpose(1, 2).unsqueeze(-1)
        if self.normalize_dp:
            grouped_xyz /= torch.amax(torch.sqrt(torch.sum(grouped_xyz**2, dim=1)), dim=(1, 2)).view(-1, 1, 1, 1)
        if features is not None:
            grouped_features = grouping_operation(features, idx)
            return grouped_xyz, grouped_features
        else:
            return grouped_xyz, None

def get_aggregation_feautres(p, dp, f, fj, feature_type='dp_fj'):
    if feature_type == 'dp_fj':
        fj = torch.cat([dp, fj], 1)
    elif feature_type == 'dp_fj_df':
        df = fj - f.unsqueeze(-1)
        fj = torch.cat([dp, fj, df], 1)
    elif feature_type == 'pi_dp_fj_df':
        df = fj - f.unsqueeze(-1)
        fj = torch.cat([p.transpose(1, 2).unsqueeze(-1).expand(-1, -1, -1, df.shape[-1]), dp, fj, df], 1)
    elif feature_type == 'dp_df':
        df = fj - f.unsqueeze(-1)
        fj = torch.cat([dp, df], 1)
    return fj

def create_grouper(group_args):
    group_args_copy = copy.deepcopy(group_args)
    method = group_args_copy.pop('NAME', 'ballquery')
    radius = group_args_copy.pop('radius', 0.1)
    nsample = group_args_copy.pop('nsample', 20)

    logging.info(group_args)
    if nsample is not None:
        if method == 'ballquery':
            grouper = QueryAndGroup(radius, nsample, **group_args_copy)
        elif method == 'knn':
            grouper = KNNGroup(nsample,  **group_args_copy)
    else:
        grouper = GroupAll()
    return grouper

if __name__ == "__main__":
    import time

    B, C, N = 2, 3, 40960
    K = 16
    device = 'cuda'
    points = torch.randn([B, N, C], device=device, dtype=torch.float)
    print(points.shape, '\n', points)

    from openpoints.models.layers.layer3d import RandomSample, random_sample, furthest_point_sample

    npoints = 10000
    idx = random_sample(points, npoints)
    query = torch.gather(points, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
    print(query.shape, '\n', query)

    idx = furthest_point_sample(points, npoints).to(torch.int64)
    query = torch.gather(points, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
    print(query.shape, '\n', query)
    query_group = QueryAndGroup(0.1, K)

    st = time.time()
    for _ in range(100):
        features = query_group(query, points)
    print(time.time() - st)
    print(features.shape)
