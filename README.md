# GraspBalance
GraspBalance: 7-DoF Grasp Pose Detection with Multi-scale Object Balanced in Cluttered Scenes


### 1. 创建一个虚拟环境
~~~shell
conda create -n grasp python=3.7
~~~~

### 2. 安装pytorch

```sh
# 进入conda环境
conda activate grasp
# CUDA 11.7
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```

### 3. 查看是否有nvcc指令



### 7. Configure KNN (K-nearest neighbors) accelerated using cuda programming:

In this paper, K-nearest neighbors (KNN) is primarily used to find the n nearest points within the grasp perception field for graspable point.

~~~shell
# cd KNN
cd KNN/
# install
sudo python setup.py install

# you can test if it is successfully installed
# a simple case
python test.py
~~~ 
