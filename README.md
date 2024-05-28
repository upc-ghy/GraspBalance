# GraspBalance
GraspBalance: 7-DoF Grasp Pose Detection with Multi-scale Object Balanced in Cluttered Scenes


### 1. Environment
python=3.7
pytorch=1.13.1 
torchvision=0.14.1 
torchaudio=0.13.1 
pytorch-cuda=11.7

### 2. Configure KNN (K-nearest neighbors) accelerated using cuda programming:

In this paper, K-nearest neighbors (KNN) is primarily used to find the n nearest points within the grasp perception field for graspable point.
~~~shell
# cd YourProject/ and clone our knn code
git clone https://github.com/upc-ghy/knn.git
# cd torchKNN
cd torchKNN
# install
sudo python setup.py install

# you can test if it is successfully installed
# a simple case
python test.py
~~~ 

### 3. Configure PointNet++
~~~shell
cd pointnet2
python setup.py install
~~~

### 4. Configure graspnetAPI (graspnetAPI is a tutorial document for data processing, pose visualization, and other functions of the GraspNet-1Billion dataset.)
~~~shell
git clone https://github.com/graspnet/graspnetAPI.git
cd graspnetAPI
pip install .
~~~
