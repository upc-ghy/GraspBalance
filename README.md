# GraspBalance
GraspBalance: 7-DoF Grasp Pose Detection with Multi-scale Object Balanced in Cluttered Scenes


### 1. 创建一个虚拟环境
~~~shell
conda create -n grasp python=3.7
~~~~

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
