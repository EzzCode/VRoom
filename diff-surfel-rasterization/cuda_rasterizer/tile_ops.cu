/*
 * Tile intersection and offset computation for gsplat-style rasterization.
 * Replaces duplicateWithKeys + identifyTileRanges from Inria's pipeline.
 */

#include "tile_ops.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// ===================================================================
// fill_isect_ids_kernel
// For each Gaussian, scatter (tile_id << 32 | depth_bits, gaussian_idx)
// into the intersection arrays using the cumsum-based offsets.
//
// This replaces Inria's duplicateWithKeys kernel.
// Key differences from Inria:
//   - Uses int64_t keys (like gsplat) instead of uint64_t
//   - Gets offsets from torch::cumsum output, not CUB InclusiveSum
//   - Encodes tile_id in the upper 32 bits using tile_n_bits
// ===================================================================
__global__ void fill_isect_ids_kernel(
	int P,
	const float2* __restrict__ means2D,
	const float* __restrict__ depths,
	const int64_t* __restrict__ cum_tiles_per_gauss,
	const int* __restrict__ radii,
	int64_t* __restrict__ isect_ids,
	int32_t* __restrict__ flatten_ids,
	dim3 tile_grid,
	uint32_t tile_n_bits)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Skip invisible Gaussians
	if (radii[idx] <= 0)
		return;

	// Get write offset from cumsum. cum_tiles_per_gauss is an inclusive
	// prefix sum, so the first Gaussian writes at offset 0 (or cum[idx-1]).
	int64_t cur_idx = (idx == 0) ? 0 : cum_tiles_per_gauss[idx - 1];

	// Compute tile bounding rect (same as preprocessCUDA)
	uint2 rect_min, rect_max;
	getRect(means2D[idx], radii[idx], rect_min, rect_max, tile_grid);

	// Reinterpret depth as uint32 for bitwise sorting
	uint32_t depth_bits = *((uint32_t*)&depths[idx]);

	// For each tile this Gaussian overlaps, emit (key, value)
	for (uint32_t y = rect_min.y; y < rect_max.y; y++)
	{
		for (uint32_t x = rect_min.x; x < rect_max.x; x++)
		{
			int64_t tile_id = (int64_t)(y * tile_grid.x + x);
			// Key format: [tile_id (tile_n_bits)] [depth (32 bits)]
			// This ensures sorting by tile first, then by depth within tile
			isect_ids[cur_idx] = (tile_id << 32) | (int64_t)depth_bits;
			flatten_ids[cur_idx] = (int32_t)idx;
			cur_idx++;
		}
	}
}

// ===================================================================
// compute_tile_offsets_kernel
// After sorting isect_ids by (tile_id, depth), find where each tile's
// range starts in the sorted list.
//
// This replaces Inria's identifyTileRanges kernel.
// Key difference: produces int32_t offsets[tile_id] (gsplat style)
// instead of uint2 ranges[tile_id] = {start, end} (Inria style).
// The end of tile T is offsets[T+1] (or n_isects for the last tile).
// ===================================================================
__global__ void compute_tile_offsets_kernel(
	int n_isects,
	const int64_t* __restrict__ sorted_isect_ids,
	int32_t* __restrict__ tile_offsets,
	uint32_t n_tiles,
	uint32_t tile_n_bits)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= n_isects)
		return;

	// Extract tile_id from the upper bits of the key
	int64_t key = sorted_isect_ids[idx];
	uint32_t currtile = (uint32_t)(key >> 32);

	if (idx == 0)
	{
		// Fill all tiles up to and including currtile with offset 0
		for (uint32_t i = 0; i <= currtile; i++)
			tile_offsets[i] = 0;
	}
	else
	{
		uint32_t prevtile = (uint32_t)(sorted_isect_ids[idx - 1] >> 32);
		if (currtile != prevtile)
		{
			// Fill tiles between prevtile+1 and currtile (inclusive) 
			// with the current index
			for (uint32_t i = prevtile + 1; i <= currtile; i++)
				tile_offsets[i] = (int32_t)idx;
		}
	}

	if (idx == n_isects - 1)
	{
		// Fill all remaining tiles with n_isects
		for (uint32_t i = currtile + 1; i < n_tiles; i++)
			tile_offsets[i] = (int32_t)n_isects;
	}
}


// ===================================================================
// Host-side launch wrappers
// ===================================================================

void TILE_OPS::fill_isect_ids(
	int P,
	const float2* means2D,
	const float* depths,
	const int64_t* cum_tiles_per_gauss,
	const int* radii,
	int64_t* isect_ids,
	int32_t* flatten_ids,
	dim3 tile_grid,
	uint32_t tile_n_bits)
{
	fill_isect_ids_kernel<<<(P + 255) / 256, 256>>>(
		P, means2D, depths, cum_tiles_per_gauss, radii,
		isect_ids, flatten_ids, tile_grid, tile_n_bits);
}

void TILE_OPS::compute_tile_offsets(
	int n_isects,
	const int64_t* sorted_isect_ids,
	int32_t* tile_offsets,
	uint32_t n_tiles,
	uint32_t tile_n_bits)
{
	if (n_isects > 0)
		compute_tile_offsets_kernel<<<(n_isects + 255) / 256, 256>>>(
			n_isects, sorted_isect_ids, tile_offsets, n_tiles, tile_n_bits);
}
