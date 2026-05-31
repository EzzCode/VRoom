from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

setup(
    name="diff_surfel_rasterization",
    packages=['diff_surfel_rasterization', 'cuda_rasterizer_rewrite'],
    version='0.0.1',
    ext_modules=[
        # ==========================================
        # ORIGINAL RASTERIZER
        # ==========================================
        CUDAExtension(
            name="diff_surfel_rasterization._C",
            sources=[
            "cuda_rasterizer/rasterizer_impl.cu",
            "cuda_rasterizer/forward.cu",
            "cuda_rasterizer/backward.cu",
            "rasterize_points.cu",
            "ext.cpp"],
            extra_compile_args={
                "nvcc": [
                    "-O3",
                    "-lineinfo",  # Inject source lines into Nsight Compute
                    "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")
                ],
                "cxx": ["-O3"]
            }
        ),
        # ==========================================
        # NEW RASTERIZER
        # ==========================================
        CUDAExtension(
            name="cuda_rasterizer_rewrite._C",
            sources=[
                "cuda_rasterizer_rewrite/src/bindings.cpp",
                "cuda_rasterizer_rewrite/src/rasterizer.cu",
                "cuda_rasterizer_rewrite/src/orchestrator.cu",
                "cuda_rasterizer_rewrite/src/fwd.cu",
                "cuda_rasterizer_rewrite/src/bwd.cu",
            ],
            extra_compile_args={
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    # Turing (RTX 2000 series, T4)
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
                    # May help GLM inline more aggressively,
                    "--expt-relaxed-constexpr",
                    # Inject source lines into Nsight Compute
                    "-lineinfo",
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
