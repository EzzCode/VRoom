from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

setup(
    name="custom_differentiable_rasterizer",
    packages=['cuda_rasterizer'],
    version='0.0.1',
    ext_modules=[
        # ==========================================
        # NEW RASTERIZER
        # ==========================================
        CUDAExtension(
            name="cuda_rasterizer._C",
            sources=[
                "cuda_rasterizer/src/bindings.cpp",
                "cuda_rasterizer/src/rasterizer.cu",
                "cuda_rasterizer/src/orchestrator.cu",
                "cuda_rasterizer/src/fwd.cu",
                "cuda_rasterizer/src/bwd.cu",
            ],
            extra_compile_args={
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    # Pascal (P100 - Kaggle)
                    "-gencode=arch=compute_60,code=sm_60",
                    # Volta (V100 - Colab)
                    "-gencode=arch=compute_70,code=sm_70",
                    # Turing (GTX series, RTX 2000 series, T4 - Kaggle)
                    "-gencode=arch=compute_75,code=sm_75",
                    # Ampere (RTX 3000 series, A100)
                    "-gencode=arch=compute_80,code=sm_80",
                    "-gencode=arch=compute_86,code=sm_86",
                    # Ada Lovelace (RTX 4000 series)
                    "-gencode=arch=compute_89,code=sm_89",
                    # Hopper (H100)
                    "-gencode=arch=compute_90,code=sm_90",
                    # Embed PTX for future architectures
                    "-gencode=arch=compute_90,code=compute_90",
                    # May help GLM inline more aggressively
                    "--expt-relaxed-constexpr",
                    # Inject source lines into Nsight Compute
                    "-lineinfo",
                    # Include thrid party libraries (currently GLM)
                    "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")
                ],
                "cxx": ["-O3"]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
