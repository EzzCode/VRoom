#include "../includes/orchestrator.cuh"
#include "../includes/types.cuh"

#include <cuda.h>
#include <cuda_runtime.h>

#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include <c10/cuda/CUDACachingAllocator.h>

#include "../includes/bwd.cuh"
#include "../includes/fwd.cuh"
#include "../includes/utils.cuh"

// Generate the keys to be used to sort surfels front to back per tile.
__global__ void key_gen_kernel(
    const int surfel_count,                                // Total surfel count
    const float2 *__restrict__ projected_centers,          // Buffer with mapped pixel locations of each surfel
    const float *__restrict__ depths,                      // Computed surfel depths as seen from the image (cam space)
    const uint32_t *__restrict__ tiles_touched_prefix_sum, // For sorting of surfels
    const uint32_t *__restrict__ asymmetric_radii,         // Both surfel radii
    const dim3 tile_grid,                                  // Tile grid dimensions
    uint64_t *__restrict__ keys_unsorted,                  // [Tile ID | Depth]
    uint32_t *__restrict__ unsorted_surfel_indices         // surfel IDs being sorted
)
{
    auto surfel_idx = cg::this_grid().thread_rank();

    // Verify the thread corresponds to a surfel
    if (surfel_idx >= surfel_count)
        return;

    // Skip culled surfels
    if (asymmetric_radii[surfel_idx] > 0) // Modified the default zero value of allocation
    {
        // Tiles touched prefix sum offset
        uint32_t offset = (surfel_idx != 0) ? tiles_touched_prefix_sum[surfel_idx - 1] : 0;

        // Unpack radii
        int2 radii;
        {
            uint32_t packed_radii = asymmetric_radii[surfel_idx];
            radii = {(int)(packed_radii & 0xFFFF), (int)(packed_radii >> 16)};
        }
        // Compute tile bounds
        uint2 min_tile_coord, max_tile_coord;
        get_tile_bounds(projected_centers[surfel_idx], radii, min_tile_coord, max_tile_coord,
                        tile_grid, BLOCK_DIM_X, BLOCK_DIM_Y);

        // Emit one key & value per tile
        for (int tile_y = min_tile_coord.y; tile_y < max_tile_coord.y; tile_y++)
        {
            for (int tile_x = min_tile_coord.x; tile_x < max_tile_coord.x; tile_x++)
            {
                uint64_t key = tile_y * tile_grid.x + tile_x;              // Tile ID
                key = (key << 32) | __float_as_uint((depths[surfel_idx])); // Tile ID | Surfel Depth
                keys_unsorted[offset] = key;
                unsorted_surfel_indices[offset] = surfel_idx;
                offset++;
            }
        }
    }
}

// Find where each tile's block of surfels begins and ends
__global__ void compute_tile_ranges_kernel(
    const int n_isects,                       // Total number of surfel-tile intersections (tiles touched)
    const uint64_t *__restrict__ keys_sorted, // [Tile ID | Depth] sorted
    uint2 *__restrict__ tile_ranges           // Per-tile [start, end) index ranges
)
{
    auto isect_idx = cg::this_grid().thread_rank();

    // Verify the thread corresponds to a valid intersection
    if (isect_idx >= n_isects)
        return;

    uint32_t curr_tile_id = keys_sorted[isect_idx] >> 32;

    if (isect_idx == 0)
        // Open first tile's range
        tile_ranges[curr_tile_id].x = 0;
    else
    {
        uint32_t prev_tile_id = keys_sorted[isect_idx - 1] >> 32;
        if (curr_tile_id != prev_tile_id)
        {
            // Boundary detected between two populated tiles.
            // Note: empty tiles are skipped and rely on allocation to set them to {0,0}
            tile_ranges[prev_tile_id].y = isect_idx; // Close previous tile's range
            tile_ranges[curr_tile_id].x = isect_idx; // Open current tile's range
        }
    }

    if (isect_idx == n_isects - 1)
        // Close final tile's range
        tile_ranges[curr_tile_id].y = n_isects;
}

// Compute the minimum number of bits needed to represent the total number of tiles.
// This is a basic binary search.
uint32_t compute_bit_width(uint32_t num_tiles)
{
    uint32_t msb = sizeof(num_tiles) * 4; // = 16. Start in the middle
    uint32_t step = msb;

    while (step > 1)
    {
        step /= 2;
        if (num_tiles >> msb)
            msb += step; // msb too small; increase it
        else
            msb -= step; // msb too large; decrease it
    }

    if (num_tiles >> msb)
        msb++;

    return msb;
}

int RasterizerOrchestrator::forward(
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
    const bool debug                                              // Debug flag to run synchronization and error checks
)
{
    // Sanity guard
    if (surfel_count == 0)
        return 0;

    // Define kernel dims
    dim3 tile_grid(DIV_CEIL(img_W, BLOCK_DIM_X), DIV_CEIL(img_H, BLOCK_DIM_Y), 1);
    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y, 1);

    // Allocate preprocessing buffers via callback
    SurfelRasterizerTypes::PreprocessBuffers prep_buffers = preprocessAllocator(surfel_count);

    // Run per-surfel preprocessing
    CUDA_SAFE_CALL(FWD::preprocess(
                       surfel_count,
                       points_world_space,
                       scale_vecs, glob_scale_mod,
                       quats,
                       opacities,
                       w2cam_mat,
                       w2clip_mat,
                       img_W, img_H,
                       tile_grid,
                       prep_buffers.projected_centers,
                       prep_buffers.asymmetric_radii,
                       prep_buffers.depths,
                       prep_buffers.splat2pix_mats,
                       prep_buffers.normal_opacity,
                       prep_buffers.tiles_touched),
                   debug);

    {
        // Find how much mem needed for prefix sum
        size_t scan_bytes = 0;
        cub::DeviceScan::InclusiveSum(nullptr, scan_bytes, prep_buffers.tiles_touched,
                                      prep_buffers.tiles_touched_prefix_sum, surfel_count);

        // Allocate scratch space for prefix sum
        auto scan_scratch = c10::cuda::CUDACachingAllocator::get()->allocate(scan_bytes);
        // Run the prefix sum
        CUDA_SAFE_CALL(cub::DeviceScan::InclusiveSum(
                           scan_scratch.get(), scan_bytes,
                           prep_buffers.tiles_touched,
                           prep_buffers.tiles_touched_prefix_sum,
                           surfel_count),
                       debug);
    }
    // Find the total number of surfel-tile intersections.
    // We use cudaMemcpy since we are in a host function trying to access VRAM.
    int n_isects;
    CUDA_SAFE_CALL(cudaMemcpy(
                       &n_isects, prep_buffers.tiles_touched_prefix_sum + surfel_count - 1,
                       sizeof(int), cudaMemcpyDeviceToHost),
                   debug);

    // Allocate binning buffers via callback
    SurfelRasterizerTypes::BinningBuffers bin_buffers = binningAllocator(n_isects);

    // Allocate image buffers via callback
    SurfelRasterizerTypes::ImageBuffers img_buffers = imageAllocator(tile_grid.x * tile_grid.y, img_W * img_H);

    if (n_isects > 0)
    {
        // Generate surfel sorting keys.
        CUDA_SAFE_CALL((key_gen_kernel<<<DIV_CEIL(surfel_count, BLOCK_SIZE), BLOCK_SIZE>>>(
                           surfel_count, prep_buffers.projected_centers, prep_buffers.depths,
                           prep_buffers.tiles_touched_prefix_sum, prep_buffers.asymmetric_radii,
                           tile_grid, bin_buffers.keys_unsorted, bin_buffers.unsorted_surfel_indices)),
                       debug);

        {
            // Get the bit width required for efficient tile sorting
            int msb = compute_bit_width(tile_grid.x * tile_grid.y);

            // Find how much mem needed for sorting
            size_t sort_bytes = 0;
            CUDA_SAFE_CALL(cub::DeviceRadixSort::SortPairs(
                               nullptr, sort_bytes,
                               bin_buffers.keys_unsorted, bin_buffers.keys_sorted,
                               bin_buffers.unsorted_surfel_indices, bin_buffers.sorted_surfel_indices,
                               n_isects, 0, msb + 32),
                           debug);

            // Allocate scratch space for sorting
            auto sort_scratch = c10::cuda::CUDACachingAllocator::get()->allocate(sort_bytes);
            // Run the sorting
            CUDA_SAFE_CALL(cub::DeviceRadixSort::SortPairs(
                               sort_scratch.get(), sort_bytes,
                               bin_buffers.keys_unsorted, bin_buffers.keys_sorted,
                               bin_buffers.unsorted_surfel_indices, bin_buffers.sorted_surfel_indices,
                               n_isects, 0, msb + 32),
                           debug);
        }

        // Compute tile ranges in the sorted key list
        CUDA_SAFE_CALL((compute_tile_ranges_kernel<<<DIV_CEIL(n_isects, BLOCK_SIZE), BLOCK_SIZE>>>(
                           n_isects, bin_buffers.keys_sorted, img_buffers.tile_ranges)),
                       debug);
    }

    // Render image
    CUDA_SAFE_CALL(FWD::render(
                       img_W, img_H, num_color_feat_channels, colors_feat, background,
                       prep_buffers.projected_centers, prep_buffers.splat2pix_mats,
                       prep_buffers.normal_opacity, bin_buffers.sorted_surfel_indices,
                       img_buffers.tile_ranges, img_buffers.contrib_state,
                       img_buffers.transmittance_and_moments,
                       rendered_color_feat, rendered_aux),
                   debug);

    return n_isects;
}

void RasterizerOrchestrator::backward(
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
    const uint2 *tile_ranges,         // Per-tile [start, end) index ranges
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
    const bool debug                 // Debug flag to run synchronization and error checks
)
{
    // Sanity guard
    if (surfel_count == 0)
        return;

    // Backpropagate gradients through rendering process
    CUDA_SAFE_CALL(BWD::render(
                       img_W, img_H, num_color_feat_channels,
                       colors_feat, background,
                       projected_centers, splat2pix_mats, normal_opacity,
                       sorted_surfel_indices,
                       tile_ranges, contrib_state, transmittance_and_moments,
                       grad_rendered_color_feat, grad_rendered_aux,
                       reinterpret_cast<float *>(grad_splat2pix_mats),
                       grad_projected_centers, grad_normal, grad_opacity, grad_colors_feat),
                   debug);

    // Backpropagate gradients through preprocessing
    CUDA_SAFE_CALL(BWD::preprocess(
                       surfel_count,
                       points_world_space, scale_vecs, glob_scale_mod,
                       quats, w2cam_mat, w2clip_mat,
                       img_W, img_H, asymmetric_radii, splat2pix_mats,
                       grad_normal,
                       grad_points_world_space, grad_scale_vecs, grad_quats,
                       grad_projected_centers, grad_splat2pix_mats),
                   debug);
}