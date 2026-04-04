"""Build script for DEF-cmssm CUDA kernels."""
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="cmssm_cuda_kernels",
    version="0.1.0",
    ext_modules=[
        CUDAExtension(
            "cmssm_cuda_kernels",
            ["cm_interleave.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-gencode=arch=compute_89,code=sm_89",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
