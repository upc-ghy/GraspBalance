# GraspBalance
GraspBalance: 7-DoF Grasp Pose Detection with Multi-scale Object Balanced in Cluttered Scenes


### 1. 创建一个虚拟环境
~~~shell
conda create -n grasp python=3.7
~~~~

### 2. 安装pytorch

```shell
# 进入conda环境
conda activate grasp
# CUDA 11.7
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```

### 3. 查看是否有nvcc指令
~~~shell
# 查看是否有nvcc
nvcc -V

# 如果显示：
[root@Rocky03 graspness]# nvcc -V
bash: nvcc: command not found
# 则表明nvcc没有加入环境变量中，或者没有安装CUDA Toolkit
# 对于上面的情况，首先查看/usr/local/路径下是否存在cuda文件

# 1.如果存在cuda文件，例如/usr/local/cuda、/usr/local/cuda-11.1、/usr/local/cuda-12.3， 则之间将配置相应的环境变量
vim ~/.bashrc 
export LD_LIBRARY_PATH=/usr/local/cuda/lib64  # 注cuda文件夹的名字根据这个local文件夹下存在cuda文件夹的名字来定
export PATH=$PATH:/usr/local/cuda/bin 
source ~/.bashrc
nvcc -V

# 2.如果不存在cuda文件夹，则说明没有安装CUDA Toolkit, 去NVIDA官网下载相应版本的CUDA Toolkit
# 各版本连接：https://developer.nvidia.com/cuda-toolkit-archive
# 可以下载nvidia-smi显示的相同的cuda版本
# 注 nvidia-smi显示的cuda version和nvcc -V显示的cuda版本的区别: 两者之间的区别在于，nvidia-smi显示的是安装在系统上的CUDA工具包的版本号，而nvcc -V显示的是当前系统中使用的CUDA编译器（NVCC）的版本号。因此，两者可能不完全一致。通常情况下，如果你的系统上安装了多个CUDA版本，nvidia-smi会显示最高版本的CUDA，而nvcc -V显示的是默认使用的CUDA版本。这是因为nvidia-smi显示的是系统上安装的所有CUDA版本，而nvcc -V显示的是由系统环境变量决定的默认CUDA版本。
[root@Rocky03 graspness]# nvidia-smi，这个版本不准确，这是安装显卡驱动自带的CUDA Version和真实使用的cuda不同版本也可以存在差异，以nvcc -V显示的为准
Sun Dec  3 18:40:37 2023
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 545.23.08              Driver Version: 545.23.08    CUDA Version: 12.3     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX 4090        On  | 00000000:01:00.0 Off |                  Off |
|  0%   49C    P8              19W / 490W |      3MiB / 24564MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+

+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+
# 我下载的就是11.7版本的CUDA Toolkit, 网络好就选第二个network在线下载
su  # 切换到root账户执行安装步骤
sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
sudo dnf clean all
sudo dnf -y module install nvidia-driver:latest-dkms  # 给的这个是下载最新版本的
sudo dnf -y install cuda  # 给的这个是下载最新版本的cuda的，我们直接手动需修改版本即可，改成下面几行

sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo #根据自己的系统版本
sudo dnf clean all
sudo dnf -y install cuda-toolkit-11-7
export PATH=/usr/local/cuda-12.2/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LIBRARY_PATH
PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin"


Ubuntu的是：
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
sudo apt-get update
sudo apt-get -y install cuda（这一条不是，这一条换成sudo apt-get -y install cuda-toolkit-11-7, 注意！我们下载的cuda中包含cuda toolkit + cuda driver。我们并不想安装driver，因为我们已经有了。之间运行英伟达提供的最后一条命令将会安装cuda toolkit + cuda driver，这是不适用于我们的要求的，不要运行sudo apt-get -y install cuda!）

# 下载安装完是不能直接使用nvcc指令的，还需要重复上面配置环境变量的步骤，才能正确使用nvcc
vim ~/.bashrc 
export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64  # 注cuda文件夹的名字根据这个local文件夹下存在cuda文件夹的名字来定
export PATH=$PATH:/usr/local/cuda-11.7/bin 
source ~/.bashrc
nvcc -V
# 这面这种方式只会在当前的终端会话中生效。一旦关闭了终端窗口或重启了计算机，这些修改就会失效。
# 想要永久保存对 ~/.bashrc 文件的修改，需要将其添加到bash配置文件中。在大多数 Linux 系统中，bash 配置文件可以是 ~/.bash_profile、~/.bash_login 或 ~/.profile 中的任何一个。可以根据你的操作系统和配置习惯选择其中之一。
vim ~/.bash_profile
export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-11.7/bin:$PATH
source ~/.bashrc
~~~

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
