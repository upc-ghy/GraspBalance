# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import glob
import os
ROOT = os.path.dirname(os.path.abspath(__file__))  # /home/gw/Code/graspnet/pointnet2/

_ext_src_root = "_ext_src"  # 这是
_ext_sources = glob.glob("{}/src/*.cpp".format(_ext_src_root)) + glob.glob(
    "{}/src/*.cu".format(_ext_src_root)    # 设置需要调用的CUDA算子源文件，这里只需要用到.cpp文件和.cu文件就行。固定写法
)
_ext_headers = glob.glob("{}/include/*".format(_ext_src_root))

setup(
    name='pointnet2',
    ext_modules=[
        CUDAExtension(
            name='pointnet2._ext',   # 这个name就是我后面导入包时用到的name, import pointnet2._ext
            sources=_ext_sources,
            extra_compile_args={
                "cxx": ["-O2", "-I{}".format("{}/{}/include".format(ROOT, _ext_src_root))],   # -O2是加速优化，-I放在路径前用于一次添加多个文件路劲
                "nvcc": ["-O2", "-I{}".format("{}/{}/include".format(ROOT, _ext_src_root))],
            },
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
