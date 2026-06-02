#pragma once

#include <torch/extension.h>
#include <tuple>

// Py-CUDA forward first-pass bridge function. Note: P is surfel count.
// ================ General outputs ================
// 1. n_isects: number of surfel-tile intersections
// 2. rendered_color_feat: [C, H, W] float32 - Rendered concat of colors + features per pixel
// 3. rendered_aux: [7, H, W] float32 - Rendered depth, normal, distortion auxiliary channels per pixel
// ================ Preprocessing Intermediates ================
// 4. projected_centers: [P, 2] float32 - mapped center pixel locations of each surfel
// 5. asymmetric_radii: [P] int32 - Packed (x|y) surfel radii in pixel-space
// 6. splat2pix_mats: [P, 3, 3] float32 - Splat to pixel space matrices buffer for each surfel
// 7. normal_opacity: [P, 4] float32 - Normals (camera space) concatenated with opacity for each surfel
// ================ Binning Intermediates ================
// 8. sorted_surfel_indices: [n_isects] int32 - sorted surfel indices (saved for backward)
// ================ Image intermediates ================
// 9. tile_ranges: [grid_size, 2] int32 - Per-tile [start, end) surfel index ranges
// 10. contrib_state: [2 , img_H , img_W] int32 - Indices of the last contributing surfel and median depth surfel per pixel
// 11. transmittance_and_moments: [3 , img_H , img_W] float32 - Final transmittance, first & second moments of depth per pixel
std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
rasterize_surfels_fwd(
    const torch::Tensor &points_world_space, // [P, 3] float32  - All points (surfels) in world space
    const torch::Tensor &scale_vecs,         // [P, 2] float32  - Scale vectors (world-space scales)
    const float glob_scale_mod,              // Scalar          - Global scale modifier
    const torch::Tensor &quats,              // [P, 4] float32  - Quaternions (world-space rotations)
    const torch::Tensor &opacities,          // [P, 1] float32  - surfel opacities
    const torch::Tensor &w2cam_mat,          // [4, 4] float32  - World to Cam space matrix
    const torch::Tensor &w2clip_mat,         // [4, 4] float32  - World to Clip space matrix
    const int img_W, const int img_H,        // Scalars         - Image width and height
    const torch::Tensor &colors_feat,        // [P, C] float32  - Concatenation of colors and features per surfel
    const torch::Tensor &background,         // [C] float32     - Background values
    const bool debug = false                 // Scalar          - Debug flag to run synchronization and error checks
);

// Py-CUDA forward subsequent-passes bridge function.
// Outputs: rendered_color_feat: [C, H, W] float32 - Rendered concat of colors + features per pixel
// Note: P is surfel count.
torch::Tensor rasterize_surfels_fwd_subsequent(
    const int img_W, const int img_H, // Scalars
    const torch::Tensor &colors_feat, // [P, C] float32
    const torch::Tensor &background,  // [C] float32
    // Saved from First Pass
    const torch::Tensor &projected_centers,     // [P, 2] float32
    const torch::Tensor &splat2pix_mats,        // [P, 3, 3] float32
    const torch::Tensor &normal_opacity,        // [P, 4] float32
    const torch::Tensor &sorted_surfel_indices, // [n_isects] int32
    const torch::Tensor &tile_ranges,           // [grid_size, 2] int32  (grid_size = ceil(W/16) * ceil(H/16))
    // Re-used output buffers (saves memory)
    const torch::Tensor &contrib_state,             // [2 , img_H , img_W] int32
    const torch::Tensor &transmittance_and_moments, // [3 , img_H , img_W] float32
    const torch::Tensor &rendered_aux,              // [7, img_H, img_W] float32
    const bool debug = false                        // Scalar
);

// Py-CUDA backward first pass bridge function. Note: P is surfel count.
// Returns:
// 1. grad_points_world_space:  [P, 3] float32 - Computed gradients of points (world space)
// 2. grad_scale_vecs:          [P, 2] float32 - Computed gradients of scale vectors
// 3. grad_quats:               [P, 4] float32 - Computed gradients of quaternions
// 4. grad_projected_centers:   [P, 2] float32 - Computed gradients of projected centers (pix space)
// 5. grad_splat2pix_mats:      [P, 3, 3] float32 - Computed gradients of splat 2 pixel space matrices
// 6. grad_opacity:             [P, 1] float32 - Computed surfel opacity gradients
// 7. grad_colors_feat:         [P, C] float32 - Computed surfel color gradients
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
rasterize_surfels_bwd(
    // Forward pass saved state
    const torch::Tensor &points_world_space, // [P, 3] float32
    const torch::Tensor &scale_vecs,         // [P, 2] float32
    const float glob_scale_mod,              // Scalar
    const torch::Tensor &quats,              // [P, 4] float32
    const torch::Tensor &w2cam_mat,          // [4, 4] float32
    const torch::Tensor &w2clip_mat,         // [4, 4] float32
    const int img_W, const int img_H,        // Scalars
    const torch::Tensor &colors_feat,        // [P, C] float32
    const torch::Tensor &background,         // [C] float32
    // Saved forward pass buffers
    // Preprocess buffers
    const torch::Tensor &projected_centers, // [P, 2] float32
    const torch::Tensor &asymmetric_radii,  // [P] int32
    const torch::Tensor &splat2pix_mats,    // [P, 3, 3] float32
    const torch::Tensor &normal_opacity,    // [P, 4] float32
    // Binning buffers
    const torch::Tensor &sorted_surfel_indices, // [n_isects] int32
    // Image buffers
    const torch::Tensor &tile_ranges,               // [grid_size, 2] int32
    const torch::Tensor &contrib_state,             // [2 , img_H , img_W] int32
    const torch::Tensor &transmittance_and_moments, // [3 , img_H , img_W] float32
    // Input gradients from PyTorch
    const torch::Tensor &grad_rendered_color_feat, // [C, H, W] float32 - Pytorch rendered pixel's gradients (image rendering loss)
    const torch::Tensor &grad_rendered_aux,        // [7, H, W] float32 - Pytorch rendered aux outputs gradients
    const bool debug = false                       // Scalar
);

// Py-CUDA backward subsequent pass bridge function. Note: P is surfel count.
// Returns the same tensors as the first pass, however all of them are dummies
// except grad_colors_feat.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
rasterize_surfels_bwd_subsequent(
    // Forward pass saved state
    const int img_W, const int img_H, // Scalars
    const torch::Tensor &colors_feat, // [P, C] float32
    const torch::Tensor &background,  // [C] float32
    // Saved forward pass buffers
    // Preprocess buffers
    const torch::Tensor &projected_centers, // [P, 2] float32
    const torch::Tensor &splat2pix_mats,    // [P, 3, 3] float32
    const torch::Tensor &normal_opacity,    // [P, 4] float32
    // Binning buffers
    const torch::Tensor &sorted_surfel_indices, // [n_isects] int32
    // Image buffers
    const torch::Tensor &tile_ranges,               // [grid_size, 2] int32
    const torch::Tensor &contrib_state,             // [2 * img_H * img_W] int32
    const torch::Tensor &transmittance_and_moments, // [3 * img_H * img_W] float32
    // Input gradients from PyTorch
    const torch::Tensor &grad_rendered_color_feat, // [C, H, W] float32 - Pytorch rendered pixel's gradients (image rendering loss)
    const torch::Tensor &grad_rendered_aux,        // [7, H, W] float32 - dummy; subsequent passes' calculations are coupled with 1st pass
    const bool debug = false                       // Scalar
);