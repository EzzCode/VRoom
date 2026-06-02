#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <functional>

namespace SurfelRasterizerTypes
{
    // Callback type for allocating preprocessing arrays after surfel_count is known.
    struct PreprocessBuffers
    {
        float2 *projected_centers;          // Buffer with mapped pixel locations of each surfel
        uint32_t *asymmetric_radii;         // Computed surfel radii in pixel-space
        float *depths;                      // Computed surfel depths as seen from the image (cam space)
        float3 *splat2pix_mats;             // Splat to pixel space matrices buffer for each surfel
        float4 *normal_opacity;             // Normals (camera space) concatenated with opacity for each surfel
        uint32_t *tiles_touched;            // Number of tiles touched by each surfel
        uint32_t *tiles_touched_prefix_sum; // For sorting of surfels front to back per tile
    };

    // Callback that allocates these tensors and returns their pointers
    using PreprocessAllocFn = std::function<PreprocessBuffers(int surfel_count)>;

    // Callback type for allocating binning arrays after n_isects (total number of tiles touched) is known.
    struct BinningBuffers
    {
        uint64_t *keys_unsorted;           // [Tile ID | Depth]
        uint64_t *keys_sorted;             // [Tile ID | Depth] sorted
        uint32_t *unsorted_surfel_indices; // surfel IDs being sorted
        uint32_t *sorted_surfel_indices;   // sorted surfel indices (saved for backward)
    };

    // Callback that allocates these tensors and returns their pointers
    using BinningAllocFn = std::function<BinningBuffers(int n_isects)>;

    // Callback type for allocating image arrays after grid_size (total number of tiles) is known.
    struct ImageBuffers
    {
        // Per-tile [start, end) surfel index ranges (sized grid_x * grid_y)
        uint2 *tile_ranges;

        // Backend State for Backward Pass Sized: (2 * img_H * img_W)
        // offset 0: Index of the last surfel that contributed to this pixel
        // offset 1: Index of the surfel that represents the median depth
        uint32_t *contrib_state;

        // Backend State for Backward Pass (Sized 3 * img_H * img_W)
        // offset 0: Final remaining transmittance (T) of the pixel
        // offset 1: M1 (First moment of depth) for distortion loss -> E[Depth]
        // offset 2: M2 (Second moment of depth) for distortion loss -> E[Depth^2]
        // Both moments are used to calculate depth variance: Var[Depth] = M2 - M1^2
        float *transmittance_and_moments;
    };

    // Callback that allocates these tensors and returns their pointers
    using ImageAllocFn = std::function<ImageBuffers(int grid_size, int img_size)>;
} // namespace SurfelRasterizerTypes