#pragma once

#include "../includes/types.cuh"
#include <cuda.h>
#include <cuda_runtime.h>

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace RasterizerOrchestrator
{
    int forward(
        const int surfel_count,                                       // Surfel / Points count
        const float3 *points_world_space,                             // All points (surfels) in world space
        const float2 *scale_vecs,                                     // Scale vectors
        const float glob_scale_mod,                                   // Global scale modifier
        const float4 *quats,                                          // Quaternions
        const float *opacities,                                       // surfel opacities
        const float *w2cam_mat,                                       // World to Cam space matrix
        const float *w2clip_mat,                                      // World to Clip space matrix
        const int img_W, const int img_H,                             // Image width and height
        const int num_color_feat_channels,                            // Number of channels in the concat of colors + features
        const float *colors_feat,                                     // Concatenation of colors and features per surfel
        const float *background,                                      // Background values
        float *rendered_color_feat,                                   // Rendered concat of colors + features per pixel
        float *rendered_aux,                                          // Rendered depth, normal, distortion auxiliary channels per pixel
        SurfelRasterizerTypes::PreprocessAllocFn preprocessAllocator, // Callback for allocating preprocessing buffers
        SurfelRasterizerTypes::BinningAllocFn binningAllocator,       // Callback for allocating binning buffers
        SurfelRasterizerTypes::ImageAllocFn imageAllocator,           // Callback for allocating image buffers
        const bool debug = false                                      // Debug flag to run synchronization and error checks
    );
} // namespace RasterizerOrchestrator