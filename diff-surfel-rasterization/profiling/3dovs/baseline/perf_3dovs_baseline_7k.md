# 2DGS Custom CUDA Rasterizer Performance Baseline for 3dovs Dataset (7k Iterations)

This baseline document acts as your hardware-level ground truth. By recording these metrics *before* rewriting or optimizing any CUDA kernels, you establish a mathematically rigorous, scientific control. Any future optimization can be compared directly against this baseline to prove speedup and efficiency to your GP committee and industry interviewers.

---

## 💻 Environment & Hardware Metadata

Fill this out before starting your baseline profiling runs to ensure reproducibility:

| Metadata Field | Value / Description |
| :--- | :--- |
| **Date & Time** | |
| **GPU Model** | NVIDIA GeForce GTX 1650 (WSL2) |
| **GPU Driver Version** | |
| **CUDA / NVCC Version** | nvcc: 11.8 (env compile) / Driver: 13.1 (profiler tools) |
| **PyTorch Version** | |
| **Active Code Branch** | |

---

## 📈 Section 1: Real-World & VRAM Baseline (`run_train_ckpt`)

This represents the clean-room, un-profiled execution of your code under peak Gaussian density (230k+ Gaussians).

> [!TIP]
> Run `nvidia-smi -l 1` in a separate terminal pane to catch the precise peak VRAM usage during the run.

| Metric | Target Baseline Value | Notes / Observations |
| :--- | :--- | :--- |
| **Steady-State Speed (`it/s`)** | 1.09 it/s | Average speed after loading the checkpoint |
| **Time per Iteration (`s/it`)** | - | Seconds per iteration |
| **Peak VRAM Allocation** |  2083MB / 4096MB | e.g., `3.6 GB / 4.0 GB` |

---

## 📊 Section 2: Macro Pipeline & System Timeline (`profile_nsys_ckpt`)

These metrics analyze how well your CPU and GPU coordinate. Gaps in the timeline represent times when your GPU is starved of work.

> [!IMPORTANT]
> The automated `nsys` profile runs for exactly 20 seconds after a 90-second delay. Let it complete automatically to write the `.nsys-rep` report cleanly.

| Timeline Feature / API | Baseline Metric / Observation | Description / What to look for |
| :--- | :--- | :--- |
| **Total Iteration Wall-Time** | 782.11 $ms$ | Combined duration of a full iteration (forward + backward pass) on the timeline. |
| **GPU Core Efficiency (HW Gaps)** | No HW gaps | Look at the top **GPU Core (HW)** row. Are there white vertical gaps (CPU starvation) or solid blocks? |
| **cudaMemcpyHtoD Total Duration** | 302.093 $ms$ | Total time spent copying data from CPU to GPU (host-to-device). |
| **cudaMemcpyDtoD Total Duration** | 102.512 $ms$ | Total time spent copying data within GPU memory (device-to-device). |
| **cudaMemcpyDtoH Total Duration** | 688.541 $\mu s$ | Total time spent copying data from GPU to CPU (device-to-host). |
| **Top 3 CUDA API Calls (Duration)** | `cudaStreamSynchronize`, `cudaMemcpyAsync` , `cudaLaunchKernel` | The three most time-consuming CUDA APIs (e.g. `cudaLaunchKernel`, `cudaStreamSynchronize`). |
| **CPU Utilization (%)** | 8.3% | System CPU utilization percentage during the active training window. |
| **Forward Pass NVTX Block Length** | - | If custom NVTX ranges are present, record the duration of one step. |
| **Backward Pass NVTX Block Length**| - | Duration of a backward step on the timeline. |

---

## 🔬 Section 3: Micro-Kernel Hardware Architecture (`profile_ncu_ckpt`)

Use **Nsight Compute** to inspect the performance of your individual CUDA kernels. You will find these metrics by looking at the 5 captured launches for each kernel.

> [!NOTE]
> For kernels with both Forward and Backward passes (like `renderCUDA` and `preprocessCUDA`), we have split them into separate tables to capture the distinct execution behaviors.

### 1. `renderCUDA` (Forward Pass)
*Main rasterization and color/depth/normal blending loop.*

| Section in NCU | Metric Name | Baseline Value | What it Measures / Why it Matters |
| :--- | :--- | :--- | :--- |
| **Summary** | **Duration** | 48.11 $ms$ | Pure execution time of the kernel ($\mu s$ or $ms$). |
| **SOL Page** | **Compute (SM) SOL %** | 73.89% | Hardware mathematical utilization. |
| **SOL Page** | **Memory SOL %** | 69.39% | VRAM bandwidth saturation percentage. |
| **Launch Stats** | **Local Memory Spill Size** | 0 | Memory spilling to local memory (bytes) - critical for latency. |
| **Scheduler** | **Top Stall Reason** | Stall not selected, Stall wait | The #1 hardware reason warps are blocked. |
| **Launch Stats** | **Registers Per Thread** | 64 | Number of registers allocated per thread (impacts occupancy). |
| **Occupancy** | **Achieved Occupancy** | 99.64% | Percentage of theoretical warps actively scheduled. |
| **Memory** | **Global Load Efficiency %** | 75% | Memory coalescing quality (Target: >85%). |

---

### 2. `renderCUDA` (Backward Pass)
*Accumulation of gradients for colors, opacities, depths, and tile-level parameters.*

| Section in NCU | Metric Name | Baseline Value | What it Measures / Why it Matters |
| :--- | :--- | :--- | :--- |
| **Summary** | **Duration** | 278.60 $ms$ | Pure execution time of the kernel ($\mu s$ or $ms$). |
| **SOL Page** | **Compute (SM) SOL %** | 61.79% | Hardware mathematical utilization. |
| **SOL Page** | **Memory SOL %** | 61.79% | VRAM bandwidth saturation percentage. |
| **Launch Stats** | **Local Memory Spill Size** | 0 | Memory spilling to local memory (bytes) - critical for latency. |
| **Scheduler** | **Top Stall Reason** | Stall Wait | The #1 hardware reason warps are blocked (highly prone to Atomic Contention!). |
| **Launch Stats** | **Registers Per Thread** | 174 | Number of registers allocated per thread (impacts occupancy). |
| **Occupancy** | **Achieved Occupancy** | 23.73% | Percentage of theoretical warps actively scheduled. |
| **Memory** | **Global Load Efficiency %** | 83% | Memory coalescing quality (Target: >85%). |

---

### 3. `preprocessCUDA` (Forward Pass)
*Projects 2D/3D Gaussians, computes normal/depth arrays, and prepares tile ranges.*

| Section in NCU | Metric Name | Baseline Value | What it Measures / Why it Matters |
| :--- | :--- | :--- | :--- |
| **Summary** | **Duration** | 838.5 $\mu s$  | Pure execution time of the kernel ($\mu s$ or $ms$). |
| **SOL Page** | **Compute (SM) SOL %** | 16.58% | Hardware mathematical utilization. |
| **SOL Page** | **Memory SOL %** | 85.12% | VRAM bandwidth saturation percentage. |
| **Launch Stats** | **Local Memory Spill Size** | 0 | Memory spilling to local memory (bytes) - critical for latency. |
| **Scheduler** | **Top Stall Reason** | Stall long scoreboard | The #1 hardware reason warps are blocked. |
| **Launch Stats** | **Registers Per Thread** | 64 | Number of registers allocated per thread (impacts occupancy). |
| **Occupancy** | **Achieved Occupancy** | 88.74% | Percentage of theoretical warps actively scheduled. |
| **Memory** | **Global Load Efficiency %** | 33% | Memory coalescing quality (Target: >85%). |

---

### 4. `preprocessCUDA` (Backward Pass)
*Gradients of loss with respect to original Gaussian dimensions, coordinates, and MLPs.*

| Section in NCU | Metric Name | Baseline Value | What it Measures / Why it Matters |
| :--- | :--- | :--- | :--- |
| **Summary** | **Duration** | 1.10 $ms$ | Pure execution time of the kernel ($\mu s$ or $ms$). |
| **SOL Page** | **Compute (SM) SOL %** | 17.92% | Hardware mathematical utilization. |
| **SOL Page** | **Memory SOL %** | 88.87% | VRAM bandwidth saturation percentage. |
| **Launch Stats** | **Local Memory Spill Size** | 0 | Memory spilling to local memory (bytes) - critical for latency. |
| **Scheduler** | **Top Stall Reason** | Stall long scoreboard | The #1 hardware reason warps are blocked. |
| **Launch Stats** | **Registers Per Thread** | 86 | Number of registers allocated per thread (impacts occupancy). |
| **Occupancy** | **Achieved Occupancy** | 45.22% | Percentage of theoretical warps actively scheduled. |
| **Memory** | **Global Load Efficiency %** | 27% | Memory coalescing quality (Target: >85%). |

---

### 5. `duplicateWithKeysCUDA`
*Duplicates primitives for each overlapping tile to prepare for sorting.*

| Section in NCU | Metric Name | Baseline Value | What it Measures / Why it Matters |
| :--- | :--- | :--- | :--- |
| **Summary** | **Duration** | 1.01 $ms$ | Pure execution time of the kernel ($\mu s$ or $ms$). |
| **SOL Page** | **Compute (SM) SOL %** | 13.29% | Hardware mathematical utilization. |
| **SOL Page** | **Memory SOL %** | 78.11% | VRAM bandwidth saturation percentage. |
| **Launch Stats** | **Local Memory Spill Size** | 0 | Memory spilling to local memory (bytes) - critical for latency. |
| **Scheduler** | **Top Stall Reason** | Stall long scoreboard | The #1 hardware reason warps are blocked. |
| **Launch Stats** | **Registers Per Thread** | 35 | Number of registers allocated per thread (impacts occupancy). |
| **Occupancy** | **Achieved Occupancy** | 70.96% | Percentage of theoretical warps actively scheduled. |
| **Memory** | **Global Load Efficiency %** | 41% | Memory coalescing quality (Target: >85%). |

---

### 6. `identifyTileRanges`
*Calculates search pointers for thread blocks within each tile.*

| Section in NCU | Metric Name | Baseline Value | What it Measures / Why it Matters |
| :--- | :--- | :--- | :--- |
| **Summary** | **Duration** | 205.92 $\mu s$ | Pure execution time of the kernel ($\mu s$ or $ms$). |
| **SOL Page** | **Compute (SM) SOL %** | 24.44% | Hardware mathematical utilization. |
| **SOL Page** | **Memory SOL %** | 92.18% | VRAM bandwidth saturation percentage. |
| **Launch Stats** | **Local Memory Spill Size** | 0 | Memory spilling to local memory (bytes) - critical for latency. |
| **Scheduler** | **Top Stall Reason** | Stall long scoreboard | The #1 hardware reason warps are blocked. |
| **Launch Stats** | **Registers Per Thread** | 16 | Number of registers allocated per thread (impacts occupancy). |
| **Occupancy** | **Achieved Occupancy** | 78.99% | Percentage of theoretical warps actively scheduled. |
| **Memory** | **Global Load Efficiency %** | 47% | Memory coalescing quality (Target: >85%). |

---

## 🔬 Scientific Benchmarking Golden Rules

To guarantee that your comparison figures are flawless:
1. **Match GPU Warmth:** Never compare a "cold boot" run directly against a "warm boot" run. Let the GPU reach a steady state temperature by warming it up before taking final baseline captures.
2. **Control the Inputs:** Keep dataset, configuration, and seeds exactly the same (`DATASET=3dovs SCENE=bed`).
3. **No Profiler Interference for Speed:** Only use `run_train_ckpt` to measure pure iteration speeds (`it/s`). Profiling with `ncu` adds massive kernel replay latency, making raw execution speeds invalid for performance benchmarks.
