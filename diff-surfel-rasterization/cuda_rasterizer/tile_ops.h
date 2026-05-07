/*
 * Tile intersection and offset computation for gsplat-style rasterization.
 * Replaces duplicateWithKeys + identifyTileRanges from Inria's pipeline.
 */

#ifndef CUDA_RASTERIZER_TILE_OPS_H_INCLUDED
#define CUDA_RASTERIZER_TILE_OPS_H_INCLUDED

#include <cuda.h>
#include <cstdint>

namespace TILE_OPS
{
	// For each visible Gaussian, write its (tile_id << 32 | depth_bits) key
	// and Gaussian index into the intersection arrays.
	// Uses cum_tiles_per_gauss (from torch::cumsum) for scatter offsets.
	void fill_isect_ids(
		int P,
		const float2* means2D,
		const float* depths,
		const int64_t* cum_tiles_per_gauss,  // [P], output of cumsum on tiles_touched
		const int* radii,
		int64_t* isect_ids,       // [n_isects], output
		int32_t* flatten_ids,     // [n_isects], output
		dim3 tile_grid,           // grid of tiles
		uint32_t tile_n_bits);

	// After sorting isect_ids, find where each tile's range starts in the
	// sorted array. Produces offsets[tile_id] such that tile's Gaussians
	// are at flatten_ids[offsets[tile_id] .. offsets[tile_id+1]).
	// The last tile extends to n_isects.
	void compute_tile_offsets(
		int n_isects,
		const int64_t* sorted_isect_ids,
		int32_t* tile_offsets,    // [n_tiles], output
		uint32_t n_tiles,
		uint32_t tile_n_bits);
}

#endif
