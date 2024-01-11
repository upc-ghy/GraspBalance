""" Tolerance label generation.
    Author: chenxi-wang
"""

import os
import sys
import numpy as np
import time
import argparse
import multiprocessing as mp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 获得当前文件所在文件夹的名
ROOT_DIR = os.path.dirname(BASE_DIR)  # 项目根目录
sys.path.append(os.path.join(ROOT_DIR, 'utils'))  # 将根目录/utils文件夹路劲添加到环境变量中
from data_utils import compute_point_dists  # 加入到环境变量后就可以使用utils中的工具包了

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', required=True, help='Dataset root')
parser.add_argument('--pos_ratio_thresh', type=float, default=0.8, help='Threshold of positive neighbor ratio[default: 0.8]')  # 
parser.add_argument('--mu_thresh', type=float, default=0.55, help='Threshold of friction coefficient[default: 0.55]')  # 摩擦力系数的阈值
parser.add_argument('--num_workers', type=int, default=50, help='Worker number[default: 50]')  # 这是线程数目吗
cfgs = parser.parse_args()

save_path = 'tolerance'

V = 300 # 300个virtual approaching vectors
A = 12  # 对于每个接近向量，末端夹持器的旋转角度被分为12个区间
D = 4   # 对于每一个旋转角度，末端夹持器沿着该approching vector方向要接近的距离(划分为四个区间0.01, 0.02, 0.03, 0.04米)
radius_list = [0.001 * x for x in range(51)]  # [0.001*0, 0.001*1, 0.001*2, 0.001*3, ....., 0.001*50]

def manager(obj_name, pool_size=8):  # 创建一个线程调用的任务函数, 参数001和pool_size=50
    # load models
    label_path = '{}_labels.npz'.format(obj_name)  # 拼接出一个文件名, '000_labels.npz', '001_labels.npz', ,..., '087_labels.npz'
    label = np.load(os.path.join(cfgs.dataset_root, 'grasp_label', label_path))  # 这是标签所在的压缩包文件夹
    points = label['points']   # 取出名为'points'的数组, 维度(3459, 3)
    scores = label['scores']   # 取出名为'scores'的数组, 维度(3459, 300, 12, 4)
    # 300个virtual approaching vectors
    # 对于每个接近向量，末端夹持器的旋转角度被分为12个区间
    # 对于每一个旋转角度，末端夹持器沿着该approching vector方向要接近的距离（接近的距离被划分为四个区间0.01, 0.02, 0.03, 0.04米）, 因此对应四个评价分数

    # create dict
    tolerance = mp.Manager().dict()  # 创建一个可在多进程之间共享的字典
    dists = compute_point_dists(points, points)  # 计算点云points和points任意两点间的欧式距离，每个点用(x,y,z)表示，这是个N*N的数组
    params = (scores, dists)  # 将数组scores和数组dists组成元组赋值给params

    # assign works
    pool = []
    process_cnt = 0
    work_list = [x for x in range(len(points))]  # 从0到3458
    for _ in range(pool_size):  # pool_size=50，创建50个线程
        point_ind = work_list.pop(0)  # 从列表work_list中弹出（不放回）第一个元素，下标为0的元素，并返回该元素的值
        # worker是一个任务函数，obj_name='001'等，point_ind是点的索引，分数和点之间距离组成的元组，tolerance多进程之间共享的字典
        # 从3459个点中，计算下标为0-49的点的方案。我们这里可以根据电脑线程的能力，把pool_size调大。
        pool.append(mp.Process(target=worker, args=(obj_name, point_ind, params, tolerance)))
    [p.start() for p in pool]  # 一个一个的执行线程

    # refill再装满
    while len(work_list) > 0:  # 弹出去了下标0-49的点，还有50-3458
        for ind, p in enumerate(pool):  # 
            if not p.is_alive():  # 使用p.is_alive()判断进程是否仍然在进行。如果该方法返回 True，则表示进程仍在运行；如果返回 False，则表示进程已经结束或被终止
                pool.pop(ind)     # 如果进行执行结束，则弹出这个进程
                point_ind = work_list.pop(0)  # 再弹出点集中的列表首个点的索引
                p = mp.Process(target=worker, args=(obj_name, point_ind, params, tolerance))  # 创建线程
                p.start()
                pool.append(p)
                process_cnt += 1  # 记录运行到哪个线程了
                print('{}/{}'.format(process_cnt, len(points)))
                break
    while len(pool) > 0:   # 如果线程池中的线程还没有运行结束
        for ind, p in enumerate(pool):
            if not p.is_alive():   # 只要是运行结束就把线程从线程池中弹出，直到结束；这个循环一直等着len(pool)等于0，有点像病毒
                pool.pop(ind)
                process_cnt += 1
                print('{}/{}'.format(process_cnt, len(points)))
                break

    # save tolerance
    if not os.path.exists(save_path):  # 在当前文件夹下创建以一个名为'tolerance'的文件
        os.mkdir(save_path)
    saved_tolerance = [None for _ in range(len(points))]  # 创建一个全是None的维度为1*3459的列表，[None, None, ..., None]
    for i in range(len(points)):  # 对于所有3459个点
        saved_tolerance[i] = tolerance[i]  # 第i个点的方案从多进程共享字典中取出
    saved_tolerance = np.array(saved_tolerance)  # 保存3459个节点的方案的列表转化为数组
    np.save('{}/{}_tolerance.npy'.format(save_path, obj_name), saved_tolerance) # 保存3459个节点的方案，每个节点方案是数组的一项

# obj_name='001'等，point_ind是点的索引，分数和点之间距离组成的元组，tolerance多进程之间共享的字典
# 对于任意一个点，求取与该点距离在0.000到0.050半斤范围内的方案数组(数组维度为(300, 12, 4))
# 方案数组的每一项表示：如果是0，则表示拉跨方案；如果是大于的值则表示这个方案按该值大小的半径采取方案
def worker(obj_name, point_ind, params, tolerance):
    scores, dists = params  # 将点云中各点的分数数组和各点间距离数组拆分开，元组类型的数据拆分方法
    tmp_tolerance = np.zeros([V, A, D], dtype=np.float32)  # 构建一个维度为(300, 12, 4)的0数组
    tic = time.time()  # 系统当前时间，是一个浮点数，从1970 年 1 月 1 日 00:00:00 UTC（世界标准时间）以来的秒数
    for r in radius_list: # [0.001*0, 0.001*1, 0.001*2, 0.001*3, ....., 0.001*50]
        # 挑选出两个点距离在0.000，0.001, 0.002,...,0.050范围内的点的下标
        dist_mask = (dists[point_ind] <= r)  # 距离函数的每一行3459个距离值与0.000比较，与0.001比较，..., 与0.05比较，每次比较返回一个1*3459维度array([True, False, False, ....，])数组
        # 挑选出两个点距离在0.000，0.001, 0.002,...,0.050范围内的点的分数
        scores_in_ball = scores[dist_mask]   # 从(3459, 300, 12, 4)的分数数组中，挑选出上面dist_mask数组为True的项，筛选的是按scores的第一项进行筛选(都是3459维)，得到维度为(True的个数, 300, 12, 4)的数组
        # 对于筛选出的每个点的分数(1, 300, 12, 4)，判断每一个值是否在>0, <=0.55，把这个分数数组变成True，False的布尔数组
        # 则((scores_in_ball > 0) & (scores_in_ball <= cfgs.mu_thresh))得到的就是一个维度为(True的个数, 300, 12, 4)布尔数组，
        # mean(axis=0)，就是对于每个点的分数的每一项与其它点的对应项统计出的(True数目)/(True数据+False数目=筛选出的点的数目)的比值
        # 这个比值等于则表示这筛选出来的在半径r范围内的的这些点按找个操作方式，所有点分数都过关了(满足下面的比值)
        # 这个比值较小则表示按这个操作，很多点分数都不过关
        # 这个pos_ratio得到的是一个维度为(300, 12, 4)比值数组
        pos_ratio = ((scores_in_ball > 0) & (scores_in_ball <= cfgs.mu_thresh)).mean(axis=0)
        tolerance_mask = (pos_ratio >= cfgs.pos_ratio_thresh)   # 从比值数组pos_ratio中挑选出比值为0.8及其以上的点的操作方案标记为True, 得到一个方案标记为True和False的数组，维度为(300, 12, 4)的方案好坏的布尔数组
        if tolerance_mask.sum() == 0:   # 如果这片方案都没有一个为True的，比值0.8及其以上的
            break
        tmp_tolerance[tolerance_mask] = r  # 如果有比值0.8及其以上的，则将该半径存入定义的方案数组tmp_tolerance的相应位置，循环结束数组相应位置有半径的是好方案，还是为0的是拉跨方案
    tolerance[point_ind] = tmp_tolerance  # 在多进程的共享字典tolerance中存入键值对{'点的下标值', '这个点的方案矩阵'}
    toc = time.time()
    print("{}: point {} time".format(obj_name, point_ind), toc - tic)

if __name__ == '__main__':
    obj_list = ['%03d' % x for x in range(88)]  # ['000', '001', '002', ...., '087'], 字符串类型, 03d表示宽度为3，不足三位的补0；3d不用补0
    for obj_name in obj_list:  # 依次取出从'000'到'087'的字符串
        p = mp.Process(target=manager, args=(obj_name, cfgs.num_workers))  # p = mp.Process() 是 multiprocessing 模块中用于创建进程对象的函数
        # 它的主要功能是在一个新进程中执行一个可调用对象。这个函数可以接受多个参数，包括：1） target: 表示子进程需要执行的任务函数； 2）args: 表示子进程需要传递给任务函数的参数元组
        # 3）kwargs: 表示子进程需要传递给任务函数的关键字参数字典；4）name: 表示子进程的名称；
        # 需要注意的是，通过调用p.start()方法可以启动新进程。
        # 同时，为了保证所有子进程能够正常退出并释放资源，最好在程序结束前显式地调用 p.join() 方法等待子进程完成任务。
        p.start()
        p.join()
