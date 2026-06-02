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

    void backward(
        // Forward pass saved state
        const int surfel_count,            // Surfel / Points count
        const float3 *points_world_space,  // All points (surfels) in world space
        const float2 *scale_vecs,          // Scale vectors
        const float glob_scale_mod,        // Global scale modifier
        const float4 *quats,               // Quaternions
        const float *w2cam_mat,            // World to Cam space matrix
        const float *w2clip_mat,           // World to Clip space matrix
        const int img_W, const int img_H,  // Image width and height
        const int num_color_feat_channels, // Number of channels in the concat of colors + features
        const float *colors_feat,          // Concatenation of colors and features per surfel
        const float *background,           // Background values
        // Forward pass saved buffers
        // Preprocess buffers
        const float2 *projected_centers,  // Mapped pixel locations of each surfel's center
        const uint32_t *asymmetric_radii, // Computed surfel radii in pixel-space
        const float3 *splat2pix_mats,     // Splat to pixel space matrices buffer for each surfel
        const float4 *normal_opacity,     // Normals (camera space) concatenated with opacity for each surfel
        // Binning buffers
        const uint32_t *sorted_surfel_indices, // sorted surfel indices
        // Image buffers
        const uint2 *tile_ranges,              // Per-tile [start, end) index ranges
        uint32_t *contrib_state,          // Indices of the last and median surfel contributing to a pixel
        float *transmittance_and_moments, // Transmittance, first and seconds moments of depth (for distortion loss)
        // Input gradients (from pytorch)
        const float *grad_rendered_color_feat, // Pytorch rendered pixel's gradients (image rendering loss)
        const float *grad_rendered_aux,        // Pytorch rendered aux outputs gradients
        // Intermediate gradients (computed and used within the rasterizer only)
        float3 *grad_normal, // Computed surfel normals gradients
        // Output gradients (to pytorch)
        float3 *grad_points_world_space, // Computed gradients of points (world space)
        float2 *grad_scale_vecs,         // Computed gradients of scale vectors
        float4 *grad_quats,              // Computed gradients of quaternions
        float2 *grad_projected_centers,  // Computed gradients of projected centers (pix space)
        float3 *grad_splat2pix_mats,     // Computed gradients of splat 2 pixel space matrices
        float *grad_opacity,             // Computed surfel opacity gradients
        float *grad_colors_feat,         // Computed surfel color gradients
        const bool debug = false         // Debug flag to run synchronization and error checks
    );
} // namespace RasterizerOrchestrator