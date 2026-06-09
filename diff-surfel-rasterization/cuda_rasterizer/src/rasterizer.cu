#include "../includes/bwd.cuh"
#include "../includes/fwd.cuh"
#include "../includes/orchestrator.cuh"
#include "../includes/pytorch_allocators.cuh"
#include "../includes/rasterizer.cuh"
#include "../includes/utils.cuh"
#include "torch/types.h"
#include <cstdint>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <tuple>

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
rasterize_surfels_fwd(
    const torch::Tensor &points_world_space, // [P, 3] float32  - All points (surfels) in world space
    const torch::Tensor &scale_vecs,         // [P, 2] float32  - Scale vectors
    const float glob_scale_mod,              // Scalar          - Global scale modifier
    const torch::Tensor &quats,              // [P, 4] float32  - Quaternions
    const torch::Tensor &opacities,          // [P, 1] float32  - surfel opacities
    const torch::Tensor &w2cam_mat,          // [4, 4] float32  - World to Cam space matrix
    const torch::Tensor &w2clip_mat,         // [4, 4] float32  - World to Clip space matrix
    const int img_W, const int img_H,        // Scalars         - Image width and height
    const torch::Tensor &colors_feat,        // [P, C] float32  - Concatenation of colors and features per surfel
    const torch::Tensor &background,         // [C] float32     - Background values
    const bool debug                         // Scalar          - Debug flag to run synchronization and error checks
)
{
    // Get surfel count and number of colors+features
    const int surfel_count = points_world_space.size(0);
    const int num_color_feat_channels = colors_feat.size(1);

    // Initialize the pytorch allocators
    PytorchAllocators::ForwardAllocationContext ctx(points_world_space.options(), num_color_feat_channels, img_W, img_H);

    // Call forward orchestrator
    int n_isects = 0;
    if (surfel_count > 0)
    {
        // Received tensors must be contigous for correct processing
        n_isects = RasterizerOrchestrator::forward(
            surfel_count,
            reinterpret_cast<float3 *>(points_world_space.contiguous().data_ptr<float>()),
            reinterpret_cast<float2 *>(scale_vecs.contiguous().data_ptr<float>()),
            glob_scale_mod,
            reinterpret_cast<float4 *>(quats.contiguous().data_ptr<float>()),
            opacities.contiguous().data_ptr<float>(),
            w2cam_mat.contiguous().data_ptr<float>(),
            w2clip_mat.contiguous().data_ptr<float>(),
            img_W, img_H,
            num_color_feat_channels,
            colors_feat.contiguous().data_ptr<float>(),
            background.contiguous().data_ptr<float>(),
            ctx.rendered_color_feat.data_ptr<float>(),
            ctx.rendered_aux.data_ptr<float>(),
            ctx.get_preprocess_allocator(),
            ctx.get_binning_allocator(),
            ctx.get_image_allocator(),
            debug);
    }

    return std::make_tuple(
        n_isects, ctx.rendered_color_feat, ctx.rendered_aux,
        ctx.projected_centers, ctx.asymmetric_radii,
        ctx.splat2pix_mats, ctx.normal_opacity,
        ctx.sorted_surfel_indices, ctx.tile_ranges,
        ctx.contrib_state, ctx.transmittance_and_moments);
}

torch::Tensor rasterize_surfels_fwd_subsequent(
    const int img_W, const int img_H, // Scalars
    const torch::Tensor &colors_feat, // [P, C] float32
    const torch::Tensor &background,  // [C] float32
    // ================ Saved from First Pass ================
    const torch::Tensor &projected_centers,     // [P, 2] float32
    const torch::Tensor &splat2pix_mats,        // [P, 3, 3] float32
    const torch::Tensor &normal_opacity,        // [P, 4] float32
    const torch::Tensor &sorted_surfel_indices, // [n_isects] int32
    const torch::Tensor &tile_ranges,           // [grid_size, 2] int32  (grid_size = ceil(W/16) * ceil(H/16))
    // ================ Re-used output buffers (saves memory) ================
    const torch::Tensor &contrib_state,             // [2 * img_H * img_W] int32
    const torch::Tensor &transmittance_and_moments, // [3 * img_H * img_W] float32
    const torch::Tensor &rendered_aux,              // [7, img_H, img_W] float32
    const bool debug                                // Scalar
)
{
    // Get number of colors+features
    const int num_color_feat_channels = colors_feat.size(1);

    // Allocate the rendered colors+features tensor
    torch::Tensor rendered_color_feat(torch::empty({num_color_feat_channels, img_H, img_W},
                                                   colors_feat.options().dtype(torch::kFloat32)));

    // Render image
    CUDA_SAFE_CALL(FWD::render(
                       false,
                       img_W, img_H,
                       num_color_feat_channels,
                       colors_feat.contiguous().data_ptr<float>(),
                       background.contiguous().data_ptr<float>(),
                       reinterpret_cast<float2 *>(projected_centers.data_ptr<float>()),
                       reinterpret_cast<float3 *>(splat2pix_mats.data_ptr<float>()),
                       reinterpret_cast<float4 *>(normal_opacity.data_ptr<float>()),
                       reinterpret_cast<uint32_t *>(sorted_surfel_indices.data_ptr<int32_t>()),
                       reinterpret_cast<uint2 *>(tile_ranges.data_ptr<int32_t>()),
                       reinterpret_cast<uint32_t *>(contrib_state.data_ptr<int32_t>()),
                       transmittance_and_moments.data_ptr<float>(),
                       rendered_color_feat.data_ptr<float>(),
                       rendered_aux.data_ptr<float>()),
                   debug);

    return rendered_color_feat;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
rasterize_surfels_bwd_render(
    // Forward pass saved state
    const bool render_aux,            // Scalar
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
    const torch::Tensor &contrib_state,             // [2 , img_H , img_W] int32
    const torch::Tensor &transmittance_and_moments, // [3 , img_H , img_W] float32
    // Input gradients from PyTorch
    const torch::Tensor &grad_rendered_color_feat, // [C, H, W] float32 - Pytorch rendered pixel's gradients (image rendering loss)
    const torch::Tensor &grad_rendered_aux,        // [7, H, W] float32 - Pytorch rendered aux outputs gradients
    const bool debug                               // Scalar
)
{
    // Get surfel count and number of colors+features
    const int surfel_count = colors_feat.size(0);
    const int num_color_feat_channels = colors_feat.size(1);

    // Allocate ouput gradient tensors
    PytorchAllocators::RenderBackwardAllocationContext ctx(
        colors_feat.options(), num_color_feat_channels, surfel_count);

    if (surfel_count > 0)
    {
        // Backpropagate gradients through rendering process
        CUDA_SAFE_CALL(BWD::render(
                           render_aux,
                           img_W, img_H, num_color_feat_channels,
                           colors_feat.contiguous().data_ptr<float>(),
                           background.contiguous().data_ptr<float>(),
                           // Preprocess buffers
                           reinterpret_cast<float2 *>(projected_centers.data_ptr<float>()),
                           reinterpret_cast<float3 *>(splat2pix_mats.data_ptr<float>()),
                           reinterpret_cast<float4 *>(normal_opacity.data_ptr<float>()),
                           /// Binning buffers
                           reinterpret_cast<uint32_t *>(sorted_surfel_indices.data_ptr<int32_t>()),
                           // Image buffers
                           reinterpret_cast<uint2 *>(tile_ranges.data_ptr<int32_t>()),
                           reinterpret_cast<uint32_t *>(contrib_state.data_ptr<int32_t>()),
                           transmittance_and_moments.data_ptr<float>(),
                           // Input gradients (from pytorch)
                           grad_rendered_color_feat.contiguous().data_ptr<float>(),
                           grad_rendered_aux.contiguous().data_ptr<float>(),
                           // Output gradients (to pytorch).
                           ctx.grad_splat2pix_mats.data_ptr<float>(),
                           reinterpret_cast<float2 *>(ctx.grad_projected_centers.data_ptr<float>()),
                           reinterpret_cast<float3 *>(ctx.grad_normal.data_ptr<float>()),
                           ctx.grad_opacity.data_ptr<float>(),
                           ctx.grad_colors_feat.data_ptr<float>()),
                       debug);
    }

    return std::make_tuple(
        ctx.grad_projected_centers,
        ctx.grad_splat2pix_mats,
        ctx.grad_normal,
        ctx.grad_opacity,
        ctx.grad_colors_feat);
}

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
rasterize_surfels_bwd_preprocess(
    // Forward pass saved state
    const torch::Tensor &points_world_space, // [P, 3] float32
    const torch::Tensor &scale_vecs,         // [P, 2] float32
    const float glob_scale_mod,              // Scalar
    const torch::Tensor &quats,              // [P, 4] float32
    const torch::Tensor &w2cam_mat,          // [4, 4] float32
    const torch::Tensor &w2clip_mat,         // [4, 4] float32
    const int img_W, const int img_H,        // Scalars
    // Saved forward pass buffers
    // Preprocess buffers
    const torch::Tensor &asymmetric_radii, // [P] int32
    const torch::Tensor &splat2pix_mats,   // [P, 3, 3] float32
    // Input gradients from PyTorch (accumulated from all bwd rendering passes)
    const torch::Tensor &grad_normal,            // [P, 3] float32
    const torch::Tensor &grad_projected_centers, // [P, 2] float32
    const torch::Tensor &grad_splat2pix_mats,    // [P, 3, 3] float32
    const bool debug                             // Scalar
)
{
    // Get surfel count
    const int surfel_count = points_world_space.size(0);

    // Allocate ouput gradient tensors
    PytorchAllocators::PreprocessBackwardAllocationContext ctx(
        points_world_space.options(),
        grad_projected_centers,
        surfel_count);

    if (surfel_count > 0)
    {
        // Backpropagate gradients through rendering process
        CUDA_SAFE_CALL(BWD::preprocess(
                           surfel_count,
                           reinterpret_cast<float3 *>(points_world_space.contiguous().data_ptr<float>()),
                           reinterpret_cast<float2 *>(scale_vecs.contiguous().data_ptr<float>()),
                           glob_scale_mod,
                           reinterpret_cast<float4 *>(quats.contiguous().data_ptr<float>()),
                           w2cam_mat.contiguous().data_ptr<float>(),
                           w2clip_mat.contiguous().data_ptr<float>(),
                           img_W, img_H,
                           // Forward pass saved buffers
                           // Preprocess buffers
                           reinterpret_cast<uint32_t *>(asymmetric_radii.data_ptr<int32_t>()),
                           reinterpret_cast<float3 *>(splat2pix_mats.data_ptr<float>()),
                           // Intermediate gradients (computed and used within the rasterizer only)
                           reinterpret_cast<float3 *>(grad_normal.contiguous().data_ptr<float>()),
                           // Output gradients (to pytorch)
                           reinterpret_cast<float3 *>(ctx.grad_points_world_space.data_ptr<float>()),
                           reinterpret_cast<float2 *>(ctx.grad_scale_vecs.data_ptr<float>()),
                           reinterpret_cast<float4 *>(ctx.grad_quats.data_ptr<float>()),
                           reinterpret_cast<float2 *>(ctx.grad_mod_projected_centers.data_ptr<float>()),
                           reinterpret_cast<float3 *>(grad_splat2pix_mats.contiguous().data_ptr<float>())),
                       debug);
    }

    return std::make_tuple(
        ctx.grad_points_world_space,
        ctx.grad_scale_vecs,
        ctx.grad_quats,
        ctx.grad_mod_projected_centers);
}

torch::Tensor frustum_cull_surfels(
    const torch::Tensor &points_world_space, // [P, 3] float32
    const torch::Tensor &scale_vecs,         // [P, 2] float32
    const float glob_scale_mod,              // Scalar
    const torch::Tensor &quats,              // [P, 4] float32
    const torch::Tensor &w2cam_mat,          // [4, 4] float32
    const torch::Tensor &w2clip_mat,         // [4, 4] float32
    const int img_W, const int img_H,        // Scalars
    const bool debug                         // Scalar
)
{
    // Get surfel count
    const int surfel_count = points_world_space.size(0);

    // Allocate radii tensor
    const torch::TensorOptions &base_opts = points_world_space.options();
    torch::Tensor radii = torch::zeros({surfel_count}, base_opts.dtype(torch::kInt32));

    if (surfel_count > 0)
    {
        // Call surfel culling kernel
        CUDA_SAFE_CALL(FWD::frustum_cull(
                           surfel_count,
                           reinterpret_cast<float3 *>(points_world_space.contiguous().data_ptr<float>()),
                           reinterpret_cast<float2 *>(scale_vecs.contiguous().data_ptr<float>()),
                           glob_scale_mod,
                           reinterpret_cast<float4 *>(quats.contiguous().data_ptr<float>()),
                           w2cam_mat.contiguous().data_ptr<float>(),
                           w2clip_mat.contiguous().data_ptr<float>(),
                           img_W, img_H,
                           radii.data_ptr<int32_t>()),
                       debug);
    }

    return radii;
}