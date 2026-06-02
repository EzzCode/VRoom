#pragma once

#include "device_launch_parameters.h"
#include <cuda.h>
#include <cuda_runtime.h>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace BWD
{
    // Backpropagate gradients through rendering process
    void render(
        const dim3 tile_grid, const dim3 block, // Tile grid and block dimensions for kernels
        const int img_W, const int img_H,       // Image width and height
        const int num_color_feat_channels,      // Number of channels in the concat of colors + features
        const float *colors_feat,               // Concatenation of colors and features per surfel
        const float *background,                // Background values
        // Preprocess buffers
        const float2 *projected_centers, // Mapped pixel locations of each surfel's center
        const float3 *splat2pix_mats,    // Splat to pixel space matrices buffer for each surfel
        const float4 *normal_opacity,    // Normals (camera space) concatenated with opacity for each surfel
        // Binning buffers
        const uint32_t *sorted_surfel_indices, // sorted surfel indices
        // Image buffers
        const uint2 *tile_ranges,               // Per-tile [start, end) index ranges
        const uint32_t *contrib_state,          // Indices of the last and median surfel contributing to a pixel
        const float *transmittance_and_moments, // Transmittance, first and seconds moments of depth (for distortion loss)
        // Input Gradients
        const float *grad_rendered_color_feat, // Pytorch rendered pixel's gradients (image rendering loss)
        const float *grad_rendered_aux,        // Pytorch rendered aux outputs gradients
        // Output Gradients
        float *grad_splat2pix_mats_buff,     // Computed splat to pixel space matrix gradients
        float2 *grad_projected_centers_buff, // Computed projected (pix spcae) surfel center gradients
        float3 *grad_normal_buff,            // Computed surfel normals gradients
        float *grad_opacity_buff,            // Computed surfel opacity gradients
        float *grad_colors_feat_buff         // Computed surfel color gradients
    );

    // Compute per-surfel gradients.
    // Chain: grad_splat2pix_mats -> grad_points_world_space, grad_scale_vecs, grad_quats
    // THEN, grad_projected_centers is re-written with a more stable grad signal.
    void preprocess(
        const int surfel_count,           // Surfel / Points count
        const float3 *points_world_space, // All points (surfels) in world space
        const float2 *scale_vecs,         // Scale vectors
        const float glob_scale_mod,       // Global scale modifier
        const float4 *quats,              // Quaternions
        const float *w2cam_mat,           // World to Cam space matrix
        const float *w2clip_mat,          // World to Clip space matrix
        const int img_W, const int img_H, // Image width and height
        const uint32_t *asymmetric_radii, // Both surfel radii for tighter bounding boxes
        const float3 *splat2pix_mats,     // Splat to pixel space matrices for each surfel
        const float3 *grad_normal,        // Gradients of 3D normals (cam space)
        float3 *grad_points_world_space,  // Computed gradients of points (world space)
        float2 *grad_scale_vecs,          // Computed gradients of scale vectors
        float4 *grad_quats,               // Computed gradients of quaternions
        float2 *grad_projected_centers,   // Computed gradients of projected centers (pix space)
        float3 *grad_splat2pix_mats       // Computed gradients of splat 2 pixel space matrices
    );
} // namespace BWD