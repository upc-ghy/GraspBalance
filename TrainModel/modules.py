import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'PointNet'))

import pytorch_utils as pt_utils
from pointnet2_utils import CylinderQueryAndGroup, BallQuery, QueryAndGroup
from loss_utils import generate_grasp_views, batch_viewpoint_params_to_matrix
from pointnet2_utils import three_nn, three_interpolate, furthest_point_sample, gather_operation


def ForegroundSampling(end_points):
    batch_seg_res = end_points["seed_cluster"]
    batch_points = end_points['point_clouds']
    batch_features = end_points['up_sample_features'].permute(0, 2, 1)
    B, N = batch_seg_res.shape
    new_fp2_xyz = []
    new_fp2_features = []
    new_fp2_inds = []
    num_points = 1024
    for i in range(B):
        seg_res = batch_seg_res[i]
        points = batch_points[i]
        features = batch_features[i]
        inds = torch.where(seg_res == 1)[0]
        object_points = points[seg_res == 1]
        object_sample_inds = furthest_point_sample(object_points.unsqueeze(0), num_points)[0].long()
        object_inds_scene = torch.gather(inds, 0, object_sample_inds)
        new_fp2_inds.append(object_inds_scene)
        new_fp2_xyz.append(torch.gather(points, 0, object_inds_scene.unsqueeze(1).expand(-1, 3)))
        new_fp2_features.append(torch.gather(features, 0, object_inds_scene.unsqueeze(1).expand(-1, 256)))

    fp2_inds = torch.stack(new_fp2_inds, 0)
    fp2_xyz = torch.stack(new_fp2_xyz, 0)
    fp2_features = torch.stack(new_fp2_features, 0).permute(0, 2, 1)
    end_points['fp2_inds_fps'] = end_points['fp2_inds']
    end_points['fp2_inds'] = fp2_inds.int()
    end_points['fp2_xyz'] = fp2_xyz
    end_points['fp2_features'] = fp2_features
    return end_points

class GraspableDetection(nn.Module):
    def __init__(self, num_view, seed_feature_dim):
        super().__init__()
        self.num_view = num_view
        self.in_dim = seed_feature_dim
        self.conv1 = nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv2 = nn.Conv1d(self.in_dim, 2 + self.num_view, 1)
        self.conv3 = nn.Conv1d(2 + self.num_view, 2 + self.num_view, 1)
        self.bn1 = nn.BatchNorm1d(self.in_dim)
        self.bn2 = nn.BatchNorm1d(2 + self.num_view)

    def forward(self, seed_xyz, seed_features, end_points, record=True):
        B, num_seed, _ = seed_xyz.size()
        features = F.relu(self.bn1(self.conv1(seed_features)), inplace=True)
        features = F.relu(self.bn2(self.conv2(features)), inplace=True)
        features = self.conv3(features)
        if record == False:
            return features
        objectness_score = features[:, :2, :]
        view_score = features[:, 2:2 + self.num_view, :].transpose(1, 2).contiguous()
        end_points['objectness_score'] = objectness_score
        end_points['view_score'] = view_score
        top_view_scores, top_view_inds = torch.max(view_score, dim=2)
        top_view_inds_ = top_view_inds.view(B, num_seed, 1, 1).expand(-1, -1, -1, 3).contiguous()
        template_views = generate_grasp_views(self.num_view).to(features.device)
        template_views = template_views.view(1, 1, self.num_view, 3).expand(B, num_seed, -1, -1).contiguous()

        vp_xyz = torch.gather(template_views, 2, top_view_inds_).squeeze(2)
        vp_xyz_ = vp_xyz.view(-1, 3)

        batch_angle = torch.zeros(vp_xyz_.size(0), dtype=vp_xyz.dtype, device=vp_xyz.device)
        vp_rot = batch_viewpoint_params_to_matrix(-vp_xyz_, batch_angle).view(B, num_seed, 3, 3)

        end_points['grasp_top_view_inds'] = top_view_inds
        end_points['grasp_top_view_score'] = top_view_scores
        end_points['grasp_top_view_xyz'] = vp_xyz
        end_points['grasp_top_view_rot'] = vp_rot
        return end_points  # ,features


class GraspWidthGrouping(nn.Module):
    def __init__(self, nsample, seed_feature_dim, cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01, 0.02, 0.03, 0.04]):
        super().__init__()
        self.nsample = nsample
        self.in_dim = seed_feature_dim
        self.cylinder_radius = cylinder_radius
        mlps = [self.in_dim, 64, 128, 256]

        self.groupers = []
        for hmax in hmax_list:
            self.groupers.append(CylinderQueryAndGroup(
                cylinder_radius, hmin, hmax, nsample, use_xyz=True
            ))
        self.mlps = pt_utils.SharedMLP(mlps, bn=True)

    def forward(self, seed_xyz, pointcloud, vp_rot):
        B, num_seed, _, _ = vp_rot.size()
        num_depth = len(self.groupers)
        grouped_features = []
        for grouper in self.groupers:
            grouped_features.append(grouper(
                pointcloud, seed_xyz, vp_rot
            ))
        grouped_features = torch.stack(grouped_features,
                                       dim=3)
        grouped_features = grouped_features.view(B, -1, num_seed * num_depth,
                                                 self.nsample)

        vp_features = self.mlps(
            grouped_features
        )
        vp_features = F.max_pool2d(
            vp_features, kernel_size=[1, vp_features.size(3)]
        )
        vp_features = vp_features.view(B, -1, num_seed, num_depth)
        return vp_features


class GraspPoseParametersDetection(nn.Module):
    def __init__(self, num_angle, num_depth):
        super().__init__()
        self.num_angle = num_angle
        self.num_depth = num_depth

        self.conv1 = nn.Conv1d(256, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 3 * num_angle, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)

    def forward(self, vp_features, end_points, record=True):
        B, _, num_seed, num_depth = vp_features.size()
        vp_features = vp_features.view(B, -1, num_seed * num_depth)
        vp_features = F.relu(self.bn1(self.conv1(vp_features)), inplace=True)
        vp_features = F.relu(self.bn2(self.conv2(vp_features)), inplace=True)
        vp_features = self.conv3(vp_features)
        vp_features = vp_features.view(B, -1, num_seed, num_depth)
        if record:
            end_points['grasp_score_pred'] = vp_features[:, 0:self.num_angle]
            end_points['grasp_angle_cls_pred'] = vp_features[:, self.num_angle:2 * self.num_angle]
            end_points['grasp_width_pred'] = vp_features[:, 2 * self.num_angle:3 * self.num_angle]
            return end_points
        else:
            return vp_features


class ToleranceNet(nn.Module):
    def __init__(self, num_angle, num_depth):
        super().__init__()
        self.conv1 = nn.Conv1d(256, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, num_angle, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)

    def forward(self, vp_features, end_points, record=True):
        B, _, num_seed, num_depth = vp_features.size()
        vp_features = vp_features.view(B, -1, num_seed * num_depth)
        vp_features = F.relu(self.bn1(self.conv1(vp_features)), inplace=True)
        vp_features = F.relu(self.bn2(self.conv2(vp_features)), inplace=True)
        vp_features = self.conv3(vp_features)
        vp_features = vp_features.view(B, -1, num_seed, num_depth)
        if record:
            end_points['grasp_tolerance_pred'] = vp_features
            return end_points
        else:
            return vp_features


def ObjectBalanceSampling(end_points):
    batch_seg_res = end_points["seed_cluster"]
    batch_points = end_points['point_clouds']
    batch_features = end_points['up_sample_features'].permute(0, 2, 1)
    B, N = batch_seg_res.shape
    new_fp2_xyz = []
    new_fp2_features = []
    new_fp2_inds = []
    for i in range(B):
        seg_res = batch_seg_res[i]
        points = batch_points[i]
        features = batch_features[i]
        idxs = torch.unique(seg_res)
        num_objects = len(idxs) - 1
        points_per_object = [1024 // num_objects for t in range(num_objects)]
        points_per_object[-1] += 1024 % num_objects
        object_inds_scene_list = []
        object_points_scene_list = []
        object_features_scene_list = []
        t = 0
        for j in idxs:
            if j == 0:
                continue
            else:
                inds = torch.where(seg_res == j)[0]
                object_points = points[seg_res == j]
                object_sample_inds = furthest_point_sample(object_points.unsqueeze(0), points_per_object[t])[0].long()
                t += 1
            object_inds_scene = torch.gather(inds, 0, object_sample_inds)
            object_inds_scene_list.append(object_inds_scene)
            object_points_scene_list.append(torch.gather(points, 0, object_inds_scene.unsqueeze(1).expand(-1, 3)))
            object_features_scene_list.append(torch.gather(features, 0, object_inds_scene.unsqueeze(1).expand(-1, 256)))
        new_fp2_inds.append(torch.cat(object_inds_scene_list, 0))
        new_fp2_xyz.append(torch.cat(object_points_scene_list, 0))
        new_fp2_features.append(torch.cat(object_features_scene_list, 0))

    fp2_inds = torch.stack(new_fp2_inds, 0)
    fp2_xyz = torch.stack(new_fp2_xyz, 0)
    fp2_features = torch.stack(new_fp2_features, 0).permute(0, 2, 1)
    end_points['fp2_inds_fps'] = end_points['fp2_inds']
    end_points['fp2_inds'] = fp2_inds.int()
    end_points['fp2_xyz'] = fp2_xyz
    end_points['fp2_features'] = fp2_features
    return end_points