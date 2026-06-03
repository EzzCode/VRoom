#pragma once

#include "device_launch_parameters.h"
#include <cuda.h>
#include <cuda_runtime.h>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace FWD
{
    // Preprocessing steps before rendering
    void preprocess(
        const int surfel_count,              // Surfel / Points count
        const float3 *points_world_space,    // All points (surfels) in world space
        const float2 *scale_vecs,            // Scale vectors
        const float glob_scale_mod,          // Global scale modifier
        const float4 *quats,                 // Quaternions
        const float *opacities,              // surfel opacities
        const float *w2cam_mat,              // World to Cam space matrix
        const float *w2clip_mat,             // World to Clip space matrix
        const int img_W, const int img_H,    // Image width and height
        const dim3 tile_grid,                // Grid dimensions for render kernels
        float2 *projected_centers_buff,      // Buffer with mapped pixel locations of each surfel's center
        uint32_t *asymmetric_radii_buff,     // Both surfel radii for tighter bounding boxes
        float *depths_buff,                  // Computed surfel depths as seen from the image (cam space)
        float3 *splat2pix_mats_buff,         // Splat to pixel space matrices buffer for each surfel
        float4 *normal_opacity_buff,         // Normals (camera space) concatenated with opacity for each surfel
        uint32_t *surfels_tiles_touched_buff // Number of tiles touched by each surfel
    );

    // Rendering image (RGB + features) using surfels
    void render(
        const int img_W, const int img_H,  // Image width and height
        const int num_color_feat_channels, // Number of channels in the concat of colors + features
        const float *colors_feat,          // Concatenation of colors and features per surfel
        const float *background,           // Background values
        // Preprocess buffers
        const float2 *projected_centers, // Mapped pixel locations of each surfel's center
        const float3 *splat2pix_mats,    // Splat to pixel space matrices buffer for each surfel
        const float4 *normal_opacity,    // Normals (camera space) concatenated with opacity for each surfel
        // Binning buffers
        const uint32_t *sorted_surfel_indices, // sorted surfel indices
        // Image buffers
        const uint2 *tile_ranges,              // Per-tile [start, end) index ranges
        uint32_t *contrib_state_buff,          // Indices of the last and median surfel contributing to a pixel
        float *transmittance_and_moments_buff, // Transmittance, first and seconds moments of depth (for distortion loss)
        // Remaining outputs
        float *rendered_color_feat_buff, // Rendered concat of colors + features per pixel
        float *rendered_aux_buff         // Rendered depth, normal, distortion auxiliary channels per pixel
    );

    // Frustum culling. Cull surfels that aren't in the current camera's view.
    void frustum_cull(
        const int surfel_count,           // Surfel / Points count
        const float3 *points_world_space, // All points (surfels) in world space
        const float2 *scale_vecs,         // Scale vectors
        const float glob_scale_mod,       // Global scale modifier
        const float4 *quats,              // Quaternions
        const float *w2cam_mat,           // World to Cam space matrix
        const float *w2clip_mat,          // World to Clip space matrix
        const int img_W, const int img_H, // Image width and height
        int *radii_buff                   // Pixel-space surfel max radius. Used as the culling metric (zero for culled surfels)
    );

} // namespace FWD