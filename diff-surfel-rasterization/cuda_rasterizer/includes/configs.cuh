#pragma once

// Kernel configuration
#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16
#define BLOCK_SIZE (BLOCK_DIM_X * BLOCK_DIM_Y)
#define NUM_THREADS_IN_WARP 32
#define NUM_WARPS (BLOCK_SIZE / NUM_THREADS_IN_WARP)

// Culling boundaries and distortion normalizers
__device__ constexpr float NEAR_PLANE = 0.01f;
__device__ constexpr float FAR_PLANE = 100.0f;
constexpr float DEPTH_NORM_SCALE = FAR_PLANE / (FAR_PLANE - NEAR_PLANE);

// Feature toggles
#define FLIP_NORMALS_TO_CAM 1   // If 1, flip normals to always face camera (important for surfel integrity)
#define RENDER_AUX 1            // If 1, render depth, normal, distortion auxiliary channels

// Render auxiliary channel layout offsets (for rendered_aux_buff indexing)
#define DEPTH_OFFSET 0
#define ALPHA_OFFSET 1
#define NORMAL_OFFSET 2 // 2, 3, 4 are the normals (x, y, z)
#define MEDIAN_DEPTH_OFFSET 5
#define DISTORTION_OFFSET 6 // Forces the optimization to place exactly one opaque surfel per pixel at the true surface depth

// Filter constants. No surfel can have a pixel-space radius smaller than FILTER_SIZE pixels
// FILTER_SIZE MUST NOT BE ZERO. IT WILL BREAK SURFEL RADII COMPUTAIONS.
__device__ constexpr float FILTER_SIZE = 0.707106f; // Half diagonal of unit square. Ensures no surfel is smaller than 1 pixel
__device__ constexpr float FILTER_INV_SQ = 2.0f;    // 1 / FilterSize^2, used for low-pass 2D distance calculation in rendering.