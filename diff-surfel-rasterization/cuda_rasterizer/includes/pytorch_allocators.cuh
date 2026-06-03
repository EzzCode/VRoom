#include "torch/types.h"
#include "types.cuh"
#include <cstdint>
#include <torch/extension.h>

namespace PytorchAllocators
{

    // This struct handles all PyTorch tensor allocations in forward pass.
    struct ForwardAllocationContext
    {
        // Data type options
        torch::TensorOptions base_opts;

        // The tensors that will be allocated and returned
        torch::Tensor projected_centers, asymmetric_radii, depths, splat2pix_mats, normal_opacity; // preprocessor
        torch::Tensor sorted_surfel_indices;                                                       // binning
        torch::Tensor tile_ranges, contrib_state, transmittance_and_moments;                       // image
        torch::Tensor rendered_color_feat, rendered_aux;                                           // outputs

        // Internal preprocessing, binning and image tensors that we don't need to return to Python
        torch::Tensor tiles_touched, tiles_touched_prefix_sum, keys_unsorted, keys_sorted, unsorted_surfel_indices;

        ForwardAllocationContext(const torch::TensorOptions &base_opts, int num_color_feat_channels,
                                 int img_W, int img_H)
            : base_opts(base_opts),
              rendered_color_feat(torch::empty({num_color_feat_channels, img_H, img_W}, base_opts.dtype(torch::kFloat32))),
              rendered_aux(torch::empty({7, img_H, img_W}, base_opts.dtype(torch::kFloat32)))
        {
        }

        SurfelRasterizerTypes::PreprocessAllocFn get_preprocess_allocator()
        {
            return [this](int surfel_count) -> SurfelRasterizerTypes::PreprocessBuffers {
                SurfelRasterizerTypes::PreprocessBuffers bufs = {};
                if (surfel_count > 0)
                {
                    // Allocate tensors.
                    // Tensors allocated as zeros can be used in culling surfels that don't change
                    // the value.
                    projected_centers = torch::empty({surfel_count, 2}, base_opts.dtype(torch::kFloat32));
                    asymmetric_radii = torch::zeros({surfel_count}, base_opts.dtype(torch::kInt32));
                    depths = torch::empty({surfel_count}, base_opts.dtype(torch::kFloat32));
                    splat2pix_mats = torch::empty({surfel_count, 3, 3}, base_opts.dtype(torch::kFloat32));
                    normal_opacity = torch::empty({surfel_count, 4}, base_opts.dtype(torch::kFloat32));
                    tiles_touched = torch::zeros({surfel_count}, base_opts.dtype(torch::kInt32));
                    tiles_touched_prefix_sum = torch::empty({surfel_count}, base_opts.dtype(torch::kInt32));

                    // Generate ptrs
                    bufs.projected_centers = reinterpret_cast<float2 *>(projected_centers.data_ptr<float>());
                    bufs.asymmetric_radii = reinterpret_cast<uint32_t *>(asymmetric_radii.data_ptr<int32_t>());
                    bufs.depths = depths.data_ptr<float>();
                    bufs.splat2pix_mats = reinterpret_cast<float3 *>(splat2pix_mats.data_ptr<float>());
                    bufs.normal_opacity = reinterpret_cast<float4 *>(normal_opacity.data_ptr<float>());
                    bufs.tiles_touched = reinterpret_cast<uint32_t *>(tiles_touched.data_ptr<int32_t>());
                    bufs.tiles_touched_prefix_sum = reinterpret_cast<uint32_t *>(tiles_touched_prefix_sum.data_ptr<int32_t>());
                }
                return bufs;
            };
        }

        SurfelRasterizerTypes::BinningAllocFn get_binning_allocator()
        {
            return [this](int n_isects) -> SurfelRasterizerTypes::BinningBuffers {
                SurfelRasterizerTypes::BinningBuffers bufs = {};
                if (n_isects > 0)
                {
                    // Allocate tensors
                    keys_unsorted = torch::empty({n_isects}, base_opts.dtype(torch::kInt64));
                    keys_sorted = torch::empty({n_isects}, base_opts.dtype(torch::kInt64));
                    unsorted_surfel_indices = torch::empty({n_isects}, base_opts.dtype(torch::kInt32));
                    sorted_surfel_indices = torch::empty({n_isects}, base_opts.dtype(torch::kInt32));

                    // Generate ptrs
                    bufs.keys_unsorted = reinterpret_cast<uint64_t *>(keys_unsorted.data_ptr<int64_t>());
                    bufs.keys_sorted = reinterpret_cast<uint64_t *>(keys_sorted.data_ptr<int64_t>());
                    bufs.unsorted_surfel_indices = reinterpret_cast<uint32_t *>(unsorted_surfel_indices.data_ptr<int32_t>());
                    bufs.sorted_surfel_indices = reinterpret_cast<uint32_t *>(sorted_surfel_indices.data_ptr<int32_t>());
                }
                return bufs;
            };
        }

        SurfelRasterizerTypes::ImageAllocFn get_image_allocator()
        {
            return [this](int grid_size, int img_size) -> SurfelRasterizerTypes::ImageBuffers {
                SurfelRasterizerTypes::ImageBuffers bufs = {};
                if (grid_size > 0 && img_size > 0)
                {
                    // Allocate tensors
                    tile_ranges = torch::zeros({grid_size, 2}, base_opts.dtype(torch::kInt32));
                    contrib_state = torch::empty({2 * img_size}, base_opts.dtype(torch::kInt32));
                    transmittance_and_moments = torch::empty({3 * img_size}, base_opts.dtype(torch::kFloat32));

                    // Generate ptrs
                    bufs.tile_ranges = reinterpret_cast<uint2 *>(tile_ranges.data_ptr<int32_t>());
                    bufs.contrib_state = reinterpret_cast<uint32_t *>(contrib_state.data_ptr<int32_t>());
                    bufs.transmittance_and_moments = transmittance_and_moments.data_ptr<float>();
                }
                return bufs;
            };
        }
    }; // struct ForwardAllocationContext

    // This struct handles all PyTorch tensor allocations in backward pass.
    struct BackwardAllocationContext
    {
        // Intermediate tensor gradients
        torch::Tensor grad_normal;

        // Output tensor gradients
        torch::Tensor grad_points_world_space;
        torch::Tensor grad_scale_vecs;
        torch::Tensor grad_quats;
        torch::Tensor grad_projected_centers;
        torch::Tensor grad_splat2pix_mats;
        torch::Tensor grad_opacity;
        torch::Tensor grad_colors_feat;

        BackwardAllocationContext(const torch::TensorOptions &base_opts, int num_color_feat_channels,
                                  int surfel_count)
        {
            // Allocate intermediate tensor
            grad_normal = torch::zeros({surfel_count, 3}, base_opts.dtype(torch::kFloat32));

            // Allocate output tensors
            grad_points_world_space = torch::zeros({surfel_count, 3}, base_opts.dtype(torch::kFloat32));
            grad_scale_vecs = torch::zeros({surfel_count, 2}, base_opts.dtype(torch::kFloat32));
            grad_quats = torch::zeros({surfel_count, 4}, base_opts.dtype(torch::kFloat32));
            grad_projected_centers = torch::zeros({surfel_count, 2}, base_opts.dtype(torch::kFloat32));
            grad_splat2pix_mats = torch::zeros({surfel_count, 3, 3}, base_opts.dtype(torch::kFloat32));
            grad_opacity = torch::zeros({surfel_count, 1}, base_opts.dtype(torch::kFloat32));
            grad_colors_feat = torch::zeros({surfel_count, num_color_feat_channels}, base_opts.dtype(torch::kFloat32));
        }
    }; // struct BackwardAllocationContext

} // namespace PytorchAllocators
