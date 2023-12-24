# coding=utf-8
# Copyright 2022 The IDEA Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------------------------
# Modified from
# https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/setup.py
# https://github.com/facebookresearch/detectron2/blob/main/setup.py
# https://github.com/open-mmlab/mmdetection/blob/master/setup.py
# https://github.com/Oneflow-Inc/libai/blob/main/setup.py
# ------------------------------------------------------------------------------------------------

import glob
import os
import subprocess

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension

# groundingdino version info
version = "0.1.0"
package_name = "groundingdino"
cwd = os.path.dirname(os.path.abspath(__file__))

requirements = ["torch", "torchvision"]

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "csrc")

    main_source = os.path.join(extensions_dir, "vision.cpp")
    sources = glob.glob(os.path.join(extensions_dir, "**", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "**", "*.cu")) + glob.glob(
        os.path.join(extensions_dir, "*.cu")
    )

    sources = [main_source] + sources

    # We need these variables to build with CUDA when we create the Docker image
    # It solves https://github.com/IDEA-Research/Grounded-Segment-Anything/issues/53
    # and https://github.com/IDEA-Research/Grounded-Segment-Anything/issues/84 when running
    # inside a Docker container.
    am_i_docker = os.environ.get('AM_I_DOCKER', '').casefold() in ['true', '1', 't']
    use_cuda = os.environ.get('BUILD_WITH_CUDA', '').casefold() in ['true', '1', 't']

    extension = CppExtension

    extra_compile_args = {"cxx": []}
    define_macros = []

    if (torch.cuda.is_available() and CUDA_HOME is not None) or \
            (am_i_docker and use_cuda):
        print("Compiling with CUDA")
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]
    else:
        print("Compiling without CUDA")
        define_macros += [("WITH_HIP", None)]
        extra_compile_args["nvcc"] = []
        return None

    sources = [os.path.join(extensions_dir, s) for s in sources]
    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            "lyrica._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


if __name__ == "__main__":
    print(f"Building wheel {package_name}-{version}")

    # with open("LICENSE", "r", encoding="utf-8") as f:
    #     license = f.read()


    setup(
        name="deformable detr cuda operator",
        version="0.1.0",
        # author="International Digital Economy Academy, Shilong Liu",
        # url="https://github.com/IDEA-Research/GroundingDINO",
        description="deformable detr cuda operator",
        # license=license,
        packages=find_packages(
            exclude=(
                "configs",
                "tests",
            )
        ),
        ext_modules=get_extensions(),
        cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
    )
