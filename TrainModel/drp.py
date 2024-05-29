import os
import sys
import torch
import torch.nn as nn
from typing import List, Type
import logging

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'PointNet'))
from pointnet2_modules import PointnetSAModuleVotes, PointnetFPModule
sys.path.append(os.path.join(ROOT_DIR, 'ModifiedNetTools'))
from conv import create_convblock1d, create_convblock2d
from activation import create_act, CHANNEL_MAP
from group import create_grouper, get_aggregation_feautres
from subsample import furthest_point_sample, random_sample


def get_reduction_fn(reduction):
    reduction = 'mean' if reduction.lower() == 'avg' else reduction
    assert reduction in ['sum', 'max', 'mean']
    if reduction == 'max':
        pool = lambda x: torch.max(x, dim=-1, keepdim=False)[0]
    elif reduction == 'mean':
        pool = lambda x: torch.mean(x, dim=-1, keepdim=False)
    elif reduction == 'sum':
        pool = lambda x: torch.sum(x, dim=-1, keepdim=False)
    return pool


class LocalAggregation(nn.Module):
    def __init__(self,
                 channels: List[int],
                 norm_args={'norm': 'bn1d'},
                 act_args={'act': 'relu'},
                 group_args={'NAME': 'ballquery', 'radius': 0.1, 'nsample': 16},
                 conv_args=None,
                 feature_type='dp_fj',
                 reduction='max',
                 last_act=True,
                 **kwargs
                 ):
        super().__init__()
        if kwargs:
            logging.warning(f"kwargs: {kwargs} are not used in {__class__.__name__}")

        channels[0] = CHANNEL_MAP[feature_type](channels[0])
        convs = []
        for i in range(len(channels) - 1):
            convs.append(create_convblock2d(channels[i], channels[i + 1],
                                            norm_args=norm_args,
                                            act_args=None if i == (len(channels) - 2) and not last_act else act_args,
                                            **conv_args)
                         )
        self.convs = nn.Sequential(*convs)
        self.grouper = create_grouper(group_args)
        self.reduction = reduction.lower()
        self.pool = get_reduction_fn(self.reduction)
        self.feature_type = feature_type

    def forward(self, pf) -> torch.Tensor:
        p, f = pf
        dp, fj = self.grouper(p, p, f)
        fj = get_aggregation_feautres(p, dp, f, fj, self.feature_type)
        f = self.pool(self.convs(fj))
        return f


class InvResMLP(nn.Module):
    def __init__(self,
                 in_channels,
                 norm_args=None,
                 act_args=None,
                 aggr_args={'feature_type': 'dp_fj', "reduction": 'max'},
                 group_args={'NAME': 'ballquery'},
                 conv_args=None,
                 expansion=1,
                 use_res=True,
                 num_posconvs=2,
                 less_act=False,
                 **kwargs
                 ):
        super().__init__()
        self.use_res = use_res
        mid_channels = int(in_channels * expansion)
        self.convs = LocalAggregation([in_channels, in_channels],
                                      norm_args=norm_args, act_args=act_args if num_posconvs > 0 else None,
                                      group_args=group_args, conv_args=conv_args,
                                      **aggr_args, **kwargs)
        # 
        if num_posconvs < 1:
            channels = []
        elif num_posconvs == 1:
            channels = [in_channels, in_channels]
        else:
            channels = [in_channels, mid_channels, in_channels]
        pwconv = []
        for i in range(len(channels) - 1):
            pwconv.append(create_convblock1d(channels[i], channels[i + 1],
                                             norm_args=norm_args,
                                             act_args=act_args if
                                             (i != len(channels) - 2) and not less_act else None,
                                             **conv_args)
                          )
        self.pwconv = nn.Sequential(*pwconv)
        self.act = create_act(act_args)

    def forward(self, pf):
        p, f = pf
        identity = f
        f = self.convs([p, f])
        f = self.pwconv(f)
        if f.shape[-1] == identity.shape[-1] and self.use_res:
            f += identity
        f = self.act(f)
        return [p, f]


class ResBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 norm_args=None,
                 act_args=None,
                 aggr_args={'feature_type': 'dp_fj', "reduction": 'max'},
                 group_args={'NAME': 'ballquery'},
                 conv_args=None,
                 expansion=1,
                 use_res=True,
                 **kwargs
                 ):
        super().__init__()
        self.use_res = use_res
        mid_channels = in_channels * expansion
        self.convs = LocalAggregation([in_channels, in_channels, mid_channels, in_channels],
                                      norm_args=norm_args, act_args=None,
                                      group_args=group_args, conv_args=conv_args,
                                      **aggr_args, **kwargs)
        self.act = create_act(act_args)
    def forward(self, pf):
        p, f = pf
        identity = f
        f = self.convs([p, f])
        if f.shape[-1] == identity.shape[-1] and self.use_res:
            f += identity
        f = self.act(f)
        return [p, f]


class DRP(nn.Module):
    def __init__(self, input_feature_dim=0):
        super().__init__()

        self.aggr_args = {'feature_type': 'dp_fj', "reduction": 'max'}
        self.norm_args = {'norm': 'bn'}
        self.act_args = {'act': 'relu'}
        self.conv_args = {'order': 'conv-norm-act'}
        self.use_res = True
        group_args = {'NAME': 'ballquery'}
        self.expansion = 4
        self.sa1 = PointnetSAModuleVotes(
                npoint=2048,
                radius=0.04,
                nsample=64,
                mlp=[input_feature_dim, 64, 64, 128],
                use_xyz=True,
                normalize_xyz=True
            )
        group_args['radius'] = 0.08
        group_args['nsample'] = 64
        blocks = []
        for i in range(3):
            blocks.append(InvResMLP(
                    in_channels=128,
                    aggr_args=self.aggr_args,
                    norm_args=self.norm_args,
                    act_args=self.act_args,
                    group_args=group_args,
                    conv_args=self.conv_args,
                    expansion=self.expansion,
                    use_res=self.use_res 
                    ))
        self.InvResMLP_blocks1 = nn.Sequential(*blocks)

        self.sa2 = PointnetSAModuleVotes(
                npoint=1024,
                radius=0.1,
                nsample=32,
                mlp=[128, 128, 128, 256],
                use_xyz=True,
                normalize_xyz=True
            )

        group_args['radius'] = 0.2
        group_args['nsample'] = 32
        blocks = []
        for i in range(6):
            blocks.append(InvResMLP(
                    in_channels=256,
                    aggr_args=self.aggr_args,
                    norm_args=self.norm_args,
                    act_args=self.act_args,
                    group_args=group_args,
                    conv_args=self.conv_args,
                    expansion=self.expansion,
                    use_res=self.use_res 
                    ))
        self.InvResMLP_blocks2 = nn.Sequential(*blocks)

        self.sa3 = PointnetSAModuleVotes(
                npoint=512,
                radius=0.2,
                nsample=16,
                mlp=[256, 128, 128, 256],
                use_xyz=True,
                normalize_xyz=True
            )

        group_args['radius'] = 0.4
        group_args['nsample'] = 16

        blocks = []
        for i in range(3):
            blocks.append(InvResMLP(
                    in_channels=256,
                    aggr_args=self.aggr_args,
                    norm_args=self.norm_args,
                    act_args=self.act_args,
                    group_args=group_args,
                    conv_args=self.conv_args,
                    expansion=self.expansion,
                    use_res=self.use_res 
                    ))
        self.InvResMLP_blocks3 = nn.Sequential(*blocks)

        self.sa4 = PointnetSAModuleVotes(
                npoint=256,
                radius=0.3,
                nsample=16,
                mlp=[256, 128, 128, 256],
                use_xyz=True,
                normalize_xyz=True
            )
        
        group_args['radius'] = 0.6
        group_args['nsample'] = 16
        blocks = []
        for i in range(3):
            blocks.append(InvResMLP(
                    in_channels=256,
                    aggr_args=self.aggr_args,
                    norm_args=self.norm_args,
                    act_args=self.act_args,
                    group_args=group_args,
                    conv_args=self.conv_args,
                    expansion=self.expansion,
                    use_res=self.use_res 
                    ))
        self.InvResMLP_blocks4 = nn.Sequential(*blocks)

        self.fp1 = PointnetFPModule(mlp=[256+256,256,256])
        self.fp2 = PointnetFPModule(mlp=[256+256,256,256])

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        )
        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor, end_points=None):
        if not end_points: end_points = {}
        batch_size = pointcloud.shape[0]

        xyz, features = self._break_up_pc(pointcloud)
        end_points['input_xyz'] = xyz
        end_points['input_features'] = features

        xyz, features, fps_inds = self.sa1(xyz, features)
        xyz, features = self.InvResMLP_blocks1([xyz, features])
        end_points['sa1_inds'] = fps_inds
        end_points['sa1_xyz'] = xyz
        end_points['sa1_features'] = features
        
        xyz, features, fps_inds = self.sa2(xyz, features)
        xyz, features = self.InvResMLP_blocks2([xyz, features])
        end_points['sa2_inds'] = fps_inds
        end_points['sa2_xyz'] = xyz
        end_points['sa2_features'] = features

        xyz, features, fps_inds = self.sa3(xyz, features)
        xyz, features = self.InvResMLP_blocks3([xyz, features])
        end_points['sa3_xyz'] = xyz
        end_points['sa3_features'] = features

        xyz, features, fps_inds = self.sa4(xyz, features)
        xyz, features = self.InvResMLP_blocks4([xyz, features])
        end_points['sa4_xyz'] = xyz
        end_points['sa4_features'] = features

        features = self.fp1(end_points['sa3_xyz'], end_points['sa4_xyz'], end_points['sa3_features'], end_points['sa4_features'])
        features = self.fp2(end_points['sa2_xyz'], end_points['sa3_xyz'], end_points['sa2_features'], features)
        end_points['fp2_features'] = features
        end_points['fp2_xyz'] = end_points['sa2_xyz']
        num_seed = end_points['fp2_xyz'].shape[1]
        end_points['fp2_inds'] = end_points['sa1_inds'][:,0:num_seed]
        return features, end_points['fp2_xyz'], end_points

