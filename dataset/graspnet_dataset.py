""" GraspNet dataset processing.
    Author: chenxi-wang
"""

import os
import sys
import numpy as np
import scipy.io as scio
from PIL import Image

import torch
# from torch._six import container_abcs  # 已被弃用
from collections.abc import Mapping, Sequence
from torch.utils.data import Dataset
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from data_utils import CameraInfo, transform_point_cloud, create_point_cloud_from_depth_image,\
                            get_workspace_mask, remove_invisible_grasp_points

class GraspNetDataset(Dataset):
    '''
    功能：
    __init__函数是创建GraspNetDataset对象，若是训练数据阶段创建，该对象创建完成后包含所有场景的标签数据，
    以及标签路径列表数据，变量数据等，几乎训练需要用到的真实标签数据都包含在该对象中。

    Args:
        root: 表示数据集的根目录dataset_root
        valid_obj_idxs: 真实标签的索引值列表[1, ..., 17, 18, 20, ..., 88]
        grasp_labels: 抓取的标签（points, offset , scores, tolerance），注和上面索引一样，不包含第19组标签
        camera: 表示选择使用哪类相机拍摄的数据
        split: 
        num_points: 传过来的参数大小为20000，作用为
        remove_outlier: 传过来的参数为True, 剔除异常值
        remove_invisible: 未传参，剔除看不见的值
        augment: 传过来的参数为True,
        load_label: 未传参，
    '''
    def __init__(self, root, valid_obj_idxs, grasp_labels, camera='kinect', split='train', num_points=20000,
                 remove_outlier=False, remove_invisible=True, augment=False, load_label=True):
        assert(num_points<=50000)  # 确保点的数目小于5w
        self.root = root
        self.split = split
        self.num_points = num_points
        self.remove_outlier = remove_outlier   # True
        self.remove_invisible = remove_invisible
        self.valid_obj_idxs = valid_obj_idxs
        self.grasp_labels = grasp_labels
        self.camera = camera
        self.augment = augment   # True
        self.load_label = load_label  # True
        self.collision_labels = {}

        if split == 'train':
            self.sceneIds = list(range(100))  # 生成训练样本的索引列表[0, 1, 2, ..., 99]组
        elif split == 'test':
            self.sceneIds = list(range(100,190)) # 生成测试样本的索引列表[100, 1, 2, ..., 189]组
        elif split == 'test_seen':
            self.sceneIds = list(range(100,130)) # 训练中见过的物体的测试样本索引列表
        elif split == 'test_similar':
            self.sceneIds = list(range(130,160)) # 训练中没见过但是相似的测试样本索引列表
        elif split == 'test_novel':
            self.sceneIds = list(range(160,190)) # 训练中全新的测试样本的索引列表

        self.sceneIds = ['scene_{}'.format(str(x).zfill(4)) for x in self.sceneIds]  # .zfill(4)不全字符串为四位，0000, 0001, ..., 0189

        # 创建了多个列表
        self.colorpath = []  # rgb.png图像的列表，每个列表包括n个场景，每个场景256个拍摄位点的rgb.png图像的路径
        self.depthpath = []  # depth.png图像的列表，同理
        self.labelpath = []  # label.png图像的列表，同理
        self.metapath = []   # meta.mat数据的列表，同理
        self.scenename = []  # 场景索引号，例如训练数据[0, ..., 99]
        self.frameid = []    # 第几个拍摄位点拍摄的数据[0, ..., 255]
        for x in tqdm(self.sceneIds, desc = 'Loading data path and collision labels...'):  # 按着是什么类型场景id来加载数据，例如训练数据是[0, 99]
            for img_num in range(256):  # 每个场景取出四种类型的数据（rgb.png, depth.png, label.png, meta.mat, x.strip(), img_num）的路径到相应类型数据的列表中
                self.colorpath.append(os.path.join(root, 'scenes', x, camera, 'rgb', str(img_num).zfill(4)+'.png'))
                self.depthpath.append(os.path.join(root, 'scenes', x, camera, 'depth', str(img_num).zfill(4)+'.png'))
                self.labelpath.append(os.path.join(root, 'scenes', x, camera, 'label', str(img_num).zfill(4)+'.png'))
                self.metapath.append(os.path.join(root, 'scenes', x, camera, 'meta', str(img_num).zfill(4)+'.mat'))
                self.scenename.append(x.strip())  # 删除字符串前导和尾随的空格
                self.frameid.append(img_num)     # 整个数据列表是一整列，例如训练数据的列表维度是1*(100*256)
            if self.load_label:  # 这个是True, 从数据集中加载collision_label数据到运行内存中
                collision_labels = np.load(os.path.join(root, 'collision_label', x.strip(), 'collision_labels.npz'))
                self.collision_labels[x.strip()] = {}   # 将self.collision_labels字典类型加一对键值，键为场景的索引号，值为空
                for i in range(len(collision_labels)):  # 将加载到内存中的第x个场景的一组collision_labels一个个插入上一行创建的键值对中
                    self.collision_labels[x.strip()][i] = collision_labels['arr_{}'.format(i)]

    def scene_list(self):
        return self.scenename

    def __len__(self):
        return len(self.depthpath)

    def augment_data(self, point_clouds, object_poses_list):
        # Flipping along the YZ plane
        if np.random.random() > 0.5:
            flip_mat = np.array([[-1, 0, 0],
                                [ 0, 1, 0],
                                [ 0, 0, 1]])
            point_clouds = transform_point_cloud(point_clouds, flip_mat, '3x3')
            for i in range(len(object_poses_list)):
                object_poses_list[i] = np.dot(flip_mat, object_poses_list[i]).astype(np.float32)

        # Rotation along up-axis/Z-axis
        rot_angle = (np.random.random()*np.pi/3) - np.pi/6 # -30 ~ +30 degree
        c, s = np.cos(rot_angle), np.sin(rot_angle)
        rot_mat = np.array([[1, 0, 0],
                            [0, c,-s],
                            [0, s, c]])
        point_clouds = transform_point_cloud(point_clouds, rot_mat, '3x3')
        for i in range(len(object_poses_list)):
            object_poses_list[i] = np.dot(rot_mat, object_poses_list[i]).astype(np.float32)

        return point_clouds, object_poses_list

    def __getitem__(self, index):
        if self.load_label:
            return self.get_data_label(index)
        else:
            return self.get_data(index)

    def get_data(self, index, return_raw_cloud=False):
        color = np.array(Image.open(self.colorpath[index]), dtype=np.float32) / 255.0
        depth = np.array(Image.open(self.depthpath[index]))  # 读取的这个深度图片.png转化成numpy数组后是一个维度为(720, 1280)的数组
        seg = np.array(Image.open(self.labelpath[index]))    # 
        meta = scio.loadmat(self.metapath[index])
        scene = self.scenename[index]
        try:
            intrinsic = meta['intrinsic_matrix']
            factor_depth = meta['factor_depth']
        except Exception as e:
            print(repr(e))
            print(scene)
        camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)

        # generate cloud
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

        # get valid points
        depth_mask = (depth > 0)
        seg_mask = (seg > 0)
        if self.remove_outlier:
            camera_poses = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'camera_poses.npy'))
            align_mat = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'cam0_wrt_table.npy'))
            trans = np.dot(align_mat, camera_poses[self.frameid[index]])
            workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
            mask = (depth_mask & workspace_mask)
        else:
            mask = depth_mask
        cloud_masked = cloud[mask]
        color_masked = color[mask]
        seg_masked = seg[mask]
        if return_raw_cloud:
            return cloud_masked, color_masked

        # sample points
        # 
        if len(cloud_masked) >= self.num_points:
            idxs = np.random.choice(len(cloud_masked), self.num_points, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_points-len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]
        
        ret_dict = {}
        ret_dict['point_clouds'] = cloud_sampled.astype(np.float32)
        ret_dict['cloud_colors'] = color_sampled.astype(np.float32)

        return ret_dict

    def get_data_label(self, index):
        # 这个index应该是从100个场景，每个场景256个视角view的也就是25600组数据集colorpath,depthpath等中按索引号取出数据路径下的数据, 索引号index从0~25599
        color = np.array(Image.open(self.colorpath[index]), dtype=np.float32) / 255.0
        depth = np.array(Image.open(self.depthpath[index]))
        seg = np.array(Image.open(self.labelpath[index]))
        meta = scio.loadmat(self.metapath[index])  # 使用scipy.io模块的loadmat()方法来读取MATLAB (.mat)格式的文件中的数组和变量。
        scene = self.scenename[index]  # 场景的索引
        try:
            obj_idxs = meta['cls_indexes'].flatten().astype(np.int32)  # 这个是1*9的数组（维度不唯一，可以是1*10等），88个物体的编号种中的几个编号组合，表示当前场景是哪几个物体，例如:[15, 1, 6, 16, 21, 59, 67, 71, 47]
            poses = meta['poses']  # 对应上面几个物体的对应的（位姿，变换，抓取位姿）3*4矩阵: 3*4*9, 其中9表示9个物体
            intrinsic = meta['intrinsic_matrix']  # 固有矩阵或者称为内部矩阵, 3*3, 是否和相机的内部矩阵有关系？
            factor_depth = meta['factor_depth']   # 一个数: 1000
        except Exception as e:
            print(repr(e))
            print(scene)
        # 例如：CameraInfo(1280.0, 720.0, 927.17, 927.37, 651.32, 349.62, 1000)
        # 3*3的instrinsic矩阵的其它位置除了instrinsic[2][2]是1，其它全是0
        camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)

        # generate cloud, 由深度图depth图来生成场景的点云数据
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

        # get valid points
        depth_mask = (depth > 0)
        seg_mask = (seg > 0)
        if self.remove_outlier:
            camera_poses = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'camera_poses.npy'))
            align_mat = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'cam0_wrt_table.npy'))
            trans = np.dot(align_mat, camera_poses[self.frameid[index]])
            workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
            mask = (depth_mask & workspace_mask)
        else:
            mask = depth_mask
        cloud_masked = cloud[mask]
        color_masked = color[mask]
        seg_masked = seg[mask]

        # sample points
        if len(cloud_masked) >= self.num_points:
            idxs = np.random.choice(len(cloud_masked), self.num_points, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_points-len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]
        seg_sampled = seg_masked[idxs]
        objectness_label = seg_sampled.copy()
        objectness_label[objectness_label>1] = 1
        
        object_poses_list = []
        grasp_points_list = []
        grasp_offsets_list = []
        grasp_scores_list = []
        grasp_tolerance_list = []
        for i, obj_idx in enumerate(obj_idxs):
            if obj_idx not in self.valid_obj_idxs:
                continue
            if (seg_sampled == obj_idx).sum() < 50:
                continue
            object_poses_list.append(poses[:, :, i])  # 从3*4*9变成了9*3*4
            points, offsets, scores, tolerance = self.grasp_labels[obj_idx]
            collision = self.collision_labels[scene][i] #(Np, V, A, D)

            # remove invisible grasp points
            if self.remove_invisible:
                visible_mask = remove_invisible_grasp_points(cloud_sampled[seg_sampled==obj_idx], points, poses[:,:,i], th=0.01)
                points = points[visible_mask]
                offsets = offsets[visible_mask]
                scores = scores[visible_mask]
                tolerance = tolerance[visible_mask]
                collision = collision[visible_mask]

            idxs = np.random.choice(len(points), min(max(int(len(points)/4),300),len(points)), replace=False)
            grasp_points_list.append(points[idxs])
            grasp_offsets_list.append(offsets[idxs])
            collision = collision[idxs].copy()
            scores = scores[idxs].copy()
            scores[collision] = 0
            grasp_scores_list.append(scores)
            tolerance = tolerance[idxs].copy()
            tolerance[collision] = 0
            grasp_tolerance_list.append(tolerance)
        
        if self.augment:
            cloud_sampled, object_poses_list = self.augment_data(cloud_sampled, object_poses_list)
        
        if torch.cuda.is_available():
            device = torch.device('cuda')  # 指定GPU设备
        else:
            device = torch.device('cpu')   # 如果没有GPU，则使用CPU

        
        
        ret_dict = {}
        ret_dict['point_clouds'] = cloud_sampled.astype(np.float32)
        ret_dict['cloud_colors'] = color_sampled.astype(np.float32)
        ret_dict['objectness_label'] = objectness_label.astype(np.int64)
        ret_dict['object_poses_list'] = object_poses_list  # 9个物体的位姿(3*4)的向量，维度为9*3*4
        ret_dict['grasp_points_list'] = grasp_points_list  # grasp_points_list的维度是(9,)
        ret_dict['grasp_offsets_list'] = grasp_offsets_list # grasp_offsets_list的维度是(9,)
        ret_dict['grasp_labels_list'] = grasp_scores_list   # grasp_scores_list的维度是(9,)
        ret_dict['grasp_tolerance_list'] = grasp_tolerance_list # grasp_tolerance_list的维度是(9,)
        
        # ret_dict = {}
        # ret_dict['point_clouds'] = cloud_sampled.astype(np.float32)
        # ret_dict['cloud_colors'] = color_sampled.astype(np.float32)
        # ret_dict['objectness_label'] = objectness_label.astype(np.int64)
        # ret_dict['object_poses_list'] = object_poses_list
        # log_string('------------------object_poses_list2的维度是%s' % str(np.array(object_poses_list).shape))
        # ret_dict['grasp_points_list'] = grasp_points_list
        # log_string('------------------grasp_points_list的维度是%s' % str(np.array(grasp_points_list).shape))
        # ret_dict['grasp_offsets_list'] = grasp_offsets_list
        # log_string('------------------grasp_offsets_list的维度是%s' % str(np.array(grasp_offsets_list).shape))
        # ret_dict['grasp_labels_list'] = grasp_scores_list
        # log_string('------------------grasp_scores_list的维度是%s' % str(np.array(grasp_scores_list).shape))
        # ret_dict['grasp_tolerance_list'] = grasp_tolerance_list
        # log_string('------------------grasp_tolerance_list的维度是%s' % str(np.array(grasp_points_list).shape))
        

        return ret_dict

# grasp_labels总共有88组，加载这88组grasp_labels到运行内存中（大约要21G内存）
# tolerance的真实标签也总共有88组，将这88组tolerance加载到运行内存中（大约要6G内存）
# 返回：
# 1. 真实标签的索引值列表[1, ..., 17, 18, 20, ..., 88]
# 2. grasp_labels是一个四元组的数组，.astype(np.float32)是把通过键值查询到的数据类型转换成32位浮点型
def load_grasp_labels(root):
    obj_names = list(range(88))
    valid_obj_idxs = []
    grasp_labels = {}
    for i, obj_name in enumerate(tqdm(obj_names, desc='Loading grasping labels...')):
        if i == 18: continue
        valid_obj_idxs.append(i + 1) #here align with label png
        label = np.load(os.path.join(root, 'grasp_label', '{}_labels.npz'.format(str(i).zfill(3))))
        tolerance = np.load(os.path.join(BASE_DIR, 'tolerance', '{}_tolerance.npy'.format(str(i).zfill(3))))
        grasp_labels[i + 1] = (label['points'].astype(np.float32), label['offsets'].astype(np.float32),
                                label['scores'].astype(np.float32), tolerance)

    return valid_obj_idxs, grasp_labels

'''
1. 该函数的功能顾名思义就是用于整理一个batch数据的函数
2. 该函数用于数据加载过程中, 当设置好batch size大小后, DataLoader会每次取出设定大小的数据交给自定义的collate_fn()函数
3. 该collate_fn()函数可以将传入的各种不同类型的数据（例如 numpy 数组、字典和列表），并将它们转换为 PyTorch 张量或嵌套的张量列表
4. 这样就可以将单个样本组装成一个mini-batch, 以供模型训练使用。
'''
def collate_fn(batch):
    # 如果它是一个 numpy 数组，那么它会将整个 batch 转换为一个 PyTorch 张量，这里使用了 torch.stack 和 torch.from_numpy 来实现。
    if type(batch[0]).__module__ == 'numpy':
        return torch.stack([torch.from_numpy(b) for b in batch], 0)
    # 如果 batch 中的数据是一个字典，它会递归地调用 collate_fn 函数来对每个键值对进行拼接，张量类型的字典，字典里是这整个batch的每个样本的键值对，
    elif isinstance(batch[0], Mapping):
        return {key:collate_fn([d[key] for d in batch]) for key in batch[0]}
    # 如果 batch 中的数据是一个列表，那么它会将该列表中的每个子列表都转换为一个嵌套的张量列表
    elif isinstance(batch[0], Sequence):
        return [[torch.from_numpy(sample) for sample in b] for b in batch]
    # 其它类型的数据则抛出异常
    raise TypeError("batch must contain tensors, dicts or lists; found {}".format(type(batch[0])))

if __name__ == "__main__":
    root = '/data/Benchmark/graspnet'
    valid_obj_idxs, grasp_labels = load_grasp_labels(root)
    train_dataset = GraspNetDataset(root, valid_obj_idxs, grasp_labels, split='train', remove_outlier=True, remove_invisible=True, num_points=20000)
    print(len(train_dataset))

    end_points = train_dataset[233]
    cloud = end_points['point_clouds']
    seg = end_points['objectness_label']
    print(cloud.shape)
    print(cloud.dtype)
    print(cloud[:,0].min(), cloud[:,0].max())
    print(cloud[:,1].min(), cloud[:,1].max())
    print(cloud[:,2].min(), cloud[:,2].max())
    print(seg.shape)
    print((seg>0).sum())
    print(seg.dtype)
    print(np.unique(seg))