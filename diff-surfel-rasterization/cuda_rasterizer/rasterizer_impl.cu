/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "rasterizer.h"
#include "rasterizer_impl.h"
#include <cuda.h>
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"

// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
__global__ void checkFrustum(int P,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool* present)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 p_view;
	present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
}

// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeys(
	int P,
	const float2* points_xy,
	const float* depths,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	int* radii,
	dim3 grid)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Generate no key/value pair for invisible Gaussians
	if (radii[idx] > 0)
	{
		// Find this Gaussian's offset in buffer for writing keys/values.
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		uint2 rect_min, rect_max;

		getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);

		// For each tile that the bounding rect overlaps, emit a 
		// key/value pair. The key is |  tile ID  |      depth      |,
		// and the value is the ID of the Gaussian. Sorting the values 
		// with this key yields Gaussian IDs in a list, such that they
		// are first sorted by tile and then by depth. 
		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				uint64_t key = y * grid.x + x;
				key <<= 32;
				key |= *((uint32_t*)&depths[idx]);
				gaussian_keys_unsorted[off] = key;
				gaussian_values_unsorted[off] = idx;
				off++;
			}
		}
	}
}

// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
}

// Mark Gaussians as visible/invisible, based on view frustum testing
void CudaRasterizer::Rasterizer::markVisible(
	int P,
	float* means3D,
	float* viewmatrix,
	float* projmatrix,
	bool* present)
{
	checkFrustum <<<(P + 255) / 256, 256 >>> (
		P,
		means3D,
		viewmatrix, projmatrix,
		present);
}

CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P, int num_color_feat_channels, bool need_sh_buffers)
{
	GeometryState geom;
	obtain(chunk, geom.depths, P, 128);
	if (need_sh_buffers) {
		obtain(chunk, geom.clamped, P * num_color_feat_channels, 128);
	} else {
		geom.clamped = nullptr;
	}
	obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.means2D, P, 128);
	obtain(chunk, geom.transMat, P * 9, 128);
	obtain(chunk, geom.normal_opacity, P, 128);
	if (need_sh_buffers) {
		obtain(chunk, geom.rgb, P * num_color_feat_channels, 128);
	} else {
		geom.rgb = nullptr;
	}
	obtain(chunk, geom.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	return geom;
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.accum_alpha, N * 3, 128);
	obtain(chunk, img.n_contrib, N * 2, 128);
	obtain(chunk, img.ranges, N, 128);
	return img;
}

CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

// Forward rendering procedure for differentiable rasterization
// of Gaussians.
int CudaRasterizer::Rasterizer::forward(
	std::function<char* (size_t)> geometryBuffer,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	const int P, int D, int M,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* opacities,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* transMat_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* cam_pos,
	const float tan_fovx, float tan_fovy,
	const bool prefiltered,
	float* out_color,
	float* out_others,
	int num_color_feat_channels,
	int* radii,
	bool debug)
{
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	const bool need_sh = (colors_precomp == nullptr);
	size_t chunk_size = required_geom(P, num_color_feat_channels, need_sh);
	char* chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P, num_color_feat_channels, need_sh);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Dynamically resize image-based auxiliary buffers during training
	size_t img_chunk_size = required<ImageState>(width * height);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	if (num_color_feat_channels != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}

	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
	CHECK_CUDA(FORWARD::preprocess(
		P, D, M,
		means3D,
		(glm::vec2*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		opacities,
		shs,
		geomState.clamped,
		transMat_precomp,
		colors_precomp,
		viewmatrix, projmatrix,
		(glm::vec3*)cam_pos,
		width, height,
		focal_x, focal_y,
		num_color_feat_channels,
		tan_fovx, tan_fovy,
		radii,
		geomState.means2D,
		geomState.depths,
		geomState.transMat,
		geomState.rgb,
		geomState.normal_opacity,
		tile_grid,
		geomState.tiles_touched,
		prefiltered
	), debug)

	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug)

	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	int num_rendered;
	CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	// and corresponding dublicated Gaussian indices to be sorted
	duplicateWithKeys <<<(P + 255) / 256, 256 >>> (
		P,
		geomState.means2D,
		geomState.depths,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		radii,
		tile_grid)
	CHECK_CUDA(, debug)

	int bit = getHigherMsb(tile_grid.x * tile_grid.y);

	// Sort complete list of (duplicated) Gaussian indices by keys
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, 32 + bit), debug)

	CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

	// Identify start and end of per-tile workloads in sorted list
	if (num_rendered > 0)
		identifyTileRanges <<<(num_rendered + 255) / 256, 256 >>> (
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges);
	CHECK_CUDA(, debug)

	// Let each tile blend its range of Gaussians independently in parallel
	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
	const float* transMat_ptr = transMat_precomp != nullptr ? transMat_precomp : geomState.transMat;
	CHECK_CUDA(FORWARD::render(
		tile_grid, block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		focal_x, focal_y,
		num_color_feat_channels,
		geomState.means2D,
		feature_ptr,
		transMat_ptr,
		geomState.depths,
		geomState.normal_opacity,
		imgState.accum_alpha,
		imgState.n_contrib,
		background,
		out_color,
		out_others), debug)

	return num_rendered;
}

// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaRasterizer::Rasterizer::backward(
	const int P, int D, int M, int R,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* transMat_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* campos,
	const float tan_fovx, float tan_fovy,
	const int* radii,
	char* geom_buffer,
	char* binning_buffer,
	char* img_buffer,
	const float* dL_dpix,
	const float* dL_depths,
	float* dL_dmean2D,
	float* dL_dnormal,
	float* dL_dopacity,
	float* dL_dcolor,
	float* dL_dmean3D,
	float* dL_dtransMat,
	float* dL_dsh,
	float* dL_dscale,
	float* dL_drot,
	bool debug,
	int num_color_feat_channels)
{
	const bool need_sh = (colors_precomp == nullptr);
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P, num_color_feat_channels, need_sh);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(img_buffer, width * height);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	// opacity and RGB of Gaussians from per-pixel loss gradients.
	// If we were given precomputed colors and not SHs, use them.
	const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;
	const float* depth_ptr = geomState.depths;
	const float* transMat_ptr = (transMat_precomp != nullptr) ? transMat_precomp : geomState.transMat;
	CHECK_CUDA(BACKWARD::render(
		tile_grid,
		block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		focal_x, focal_y,
		num_color_feat_channels,
		background,
		geomState.means2D,
		geomState.normal_opacity,
		color_ptr,
		transMat_ptr,
		depth_ptr,
		imgState.accum_alpha,
		imgState.n_contrib,
		dL_dpix,
		dL_depths,
		dL_dtransMat,
		(float3*)dL_dmean2D,
		dL_dnormal,
		dL_dopacity,
		dL_dcolor), debug)

	// Take care of the rest of preprocessing. Was the precomputed covariance
	// given to us or a scales/rot pair? If precomputed, pass that. If not,
	// use the one we computed ourselves.
	// const float* transMat_ptr = (transMat_precomp != nullptr) ? transMat_precomp : geomState.transMat;
	CHECK_CUDA(BACKWARD::preprocess(P, D, M,
		(float3*)means3D,
		radii,
		shs,
		geomState.clamped,
		(glm::vec2*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		transMat_ptr,
		viewmatrix,
		projmatrix,
		focal_x, focal_y,
		num_color_feat_channels,
		tan_fovx, tan_fovy,
		(glm::vec3*)campos,
		(float3*)dL_dmean2D, // gradient inputs
		dL_dnormal,		     // gradient inputs
		dL_dtransMat,
		dL_dcolor,
		dL_dsh,
		(glm::vec3*)dL_dmean3D,
		(glm::vec2*)dL_dscale,
		(glm::vec4*)dL_drot), debug)
}

// ===================================================================
// forward_v2: gsplat-style pipeline using individual tensor buffers.
//
// Pipeline:
//   1. preprocessCUDA (unchanged) → fills means2D, depths, transMats, etc.
//   2. fill_isect_ids → scatter (tile_id|depth, idx) into arrays
//   3. CUB RadixSort with DoubleBuffer → sorted isect_ids
//   4. compute_tile_offsets → int32_t offsets per tile
//   5. renderCUDA_v2 → produces pixel colors
//
// All buffers are pre-allocated by the caller as torch::Tensor,
// not by resizeFunctional byte blobs.
// ===================================================================

#include "tile_ops.h"

int CudaRasterizer::Rasterizer::forward_v2(
	const int P, int D, int M,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* opacities,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* transMat_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* cam_pos,
	const float tan_fovx, float tan_fovy,
	const bool prefiltered,
	float* out_color,
	float* out_others,
	int num_color_feat_channels,
	int* radii,
	float2* means2D,
	float* depths,
	float* transMats_out,
	float4* normal_opacity,
	uint32_t* tiles_touched,
	int32_t* tile_offsets,
	int32_t* flatten_ids,
	int64_t* isect_ids,
	int64_t* isect_ids_sorted,
	int32_t* flatten_ids_unsorted,
	float* accum_alpha,
	uint32_t* n_contrib,
	bool debug)
{
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);
	uint32_t n_tiles = tile_grid.x * tile_grid.y;

	if (num_color_feat_channels != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}

	// SH clamped buffer — nullptr if we're using precomputed colors
	bool* clamped = nullptr; // SH path would need to allocate this separately

	// 1. Preprocess: project, compute transMat, AABB, tiles_touched
	CHECK_CUDA(FORWARD::preprocess(
		P, D, M,
		means3D,
		(glm::vec2*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		opacities,
		shs,
		clamped,
		transMat_precomp,
		colors_precomp,
		viewmatrix, projmatrix,
		(glm::vec3*)cam_pos,
		width, height,
		focal_x, focal_y,
		num_color_feat_channels,
		tan_fovx, tan_fovy,
		radii,
		means2D,
		depths,
		transMats_out,
		nullptr, // rgb — null since we always use colors_precomp
		normal_opacity,
		tile_grid,
		tiles_touched,
		prefiltered
	), debug)

	// 2. Inclusive prefix sum of tiles_touched to get cumulative offsets
	//    This is done on the Python side via torch::cumsum() and the result
	//    is passed to fill_isect_ids. But we need the total count now.
	//    The caller computes cumsum externally and passes n_isects.
	//    Here we just need to get num_rendered from the last element.
	int num_rendered;
	CHECK_CUDA(cudaMemcpy(&num_rendered, tiles_touched + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

	// Note: at this point, tiles_touched has been converted to cumsum by the caller
	// and num_rendered == total tile-Gaussian instances.
	// The caller has already allocated isect_ids[num_rendered] etc.

	if (num_rendered == 0)
		return 0;

	// 3. Fill intersection IDs using cumsum offsets
	uint32_t tile_n_bits = getHigherMsb(n_tiles);
	TILE_OPS::fill_isect_ids(
		P, means2D, depths,
		(const int64_t*)tiles_touched,  // reinterpreted cumsum
		radii,
		isect_ids, flatten_ids_unsorted,
		tile_grid, tile_n_bits);
	CHECK_CUDA(, debug)

	// 4. Sort by (tile_id << 32 | depth) using CUB DoubleBuffer
	{
		size_t sort_temp_size = 0;
		cub::DoubleBuffer<int64_t> d_keys(isect_ids, isect_ids_sorted);
		cub::DoubleBuffer<int32_t> d_values(flatten_ids_unsorted, flatten_ids);

		// Query sort workspace size
		CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
			nullptr, sort_temp_size,
			d_keys, d_values,
			num_rendered, 0, 32 + tile_n_bits), debug)

		// Allocate temporary workspace via cudaMalloc (one-shot)
		char* sort_temp = nullptr;
		CHECK_CUDA(cudaMalloc(&sort_temp, sort_temp_size), debug);

		// Execute sort
		CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
			sort_temp, sort_temp_size,
			d_keys, d_values,
			num_rendered, 0, 32 + tile_n_bits), debug)

		// If DoubleBuffer swapped the pointers, we need to copy the results
		// back to the expected output locations
		if (d_keys.Current() != isect_ids_sorted) {
			CHECK_CUDA(cudaMemcpy(isect_ids_sorted, d_keys.Current(),
				num_rendered * sizeof(int64_t), cudaMemcpyDeviceToDevice), debug);
		}
		if (d_values.Current() != flatten_ids) {
			CHECK_CUDA(cudaMemcpy(flatten_ids, d_values.Current(),
				num_rendered * sizeof(int32_t), cudaMemcpyDeviceToDevice), debug);
		}

		CHECK_CUDA(cudaFree(sort_temp), debug);
	}

	// 5. Compute tile offsets from sorted keys
	CHECK_CUDA(cudaMemset(tile_offsets, 0, n_tiles * sizeof(int32_t)), debug);
	TILE_OPS::compute_tile_offsets(
		num_rendered, isect_ids_sorted,
		tile_offsets, n_tiles, tile_n_bits);
	CHECK_CUDA(, debug)

	// 6. Render
	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : nullptr;
	const float* transMat_ptr = transMat_precomp != nullptr ? transMat_precomp : transMats_out;
	CHECK_CUDA(FORWARD::render_v2(
		tile_grid, block,
		tile_offsets,
		flatten_ids,
		num_rendered,
		width, height,
		focal_x, focal_y,
		num_color_feat_channels,
		means2D,
		feature_ptr,
		transMat_ptr,
		depths,
		normal_opacity,
		accum_alpha,
		n_contrib,
		background,
		out_color,
		out_others), debug)

	return num_rendered;
}


// ===================================================================
// backward_v2: gradient computation using saved individual tensors.
// Runs backward render + backward preprocess.
// ===================================================================
void CudaRasterizer::Rasterizer::backward_v2(
	const int P, int D, int M, int R,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* transMat_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* campos,
	const float tan_fovx, float tan_fovy,
	const int* radii,
	const float2* means2D,
	const float* depths,
	const float* transMats,
	const float4* normal_opacity,
	const float* accum_alpha,
	const uint32_t* n_contrib,
	const int32_t* tile_offsets,
	const int32_t* flatten_ids,
	int n_isects,
	const float* dL_dpix,
	const float* dL_depths,
	float* dL_dmean2D,
	float* dL_dnormal,
	float* dL_dopacity,
	float* dL_dcolor,
	float* dL_dmean3D,
	float* dL_dtransMat,
	float* dL_dsh,
	float* dL_dscale,
	float* dL_drot,
	bool debug,
	int num_color_feat_channels)
{
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : nullptr;
	const float* transMat_ptr = (transMat_precomp != nullptr) ? transMat_precomp : transMats;

	// 1. Backward render — produces dL/d(transMat), dL/d(mean2D), dL/d(color), dL/d(opacity)
	CHECK_CUDA(BACKWARD::render_v2(
		tile_grid,
		block,
		tile_offsets,
		flatten_ids,
		n_isects,
		width, height,
		focal_x, focal_y,
		num_color_feat_channels,
		background,
		means2D,
		normal_opacity,
		color_ptr,
		transMat_ptr,
		depths,
		accum_alpha,
		n_contrib,
		dL_dpix,
		dL_depths,
		dL_dtransMat,
		(float3*)dL_dmean2D,
		dL_dnormal,
		dL_dopacity,
		dL_dcolor), debug)

	// 2. Backward preprocess — chains dL/d(transMat) → dL/d(mean3D), dL/d(scale), dL/d(rot)
	//    Also chains dL/d(mean2D) through the AABB→screen projection.
	bool* clamped = nullptr; // no SH clamping when using colors_precomp

	CHECK_CUDA(BACKWARD::preprocess(P, D, M,
		(float3*)means3D,
		radii,
		shs,
		clamped,
		(glm::vec2*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		transMat_ptr,
		viewmatrix,
		projmatrix,
		focal_x, focal_y,
		num_color_feat_channels,
		tan_fovx, tan_fovy,
		(glm::vec3*)campos,
		(float3*)dL_dmean2D,
		dL_dnormal,
		dL_dtransMat,
		dL_dcolor,
		dL_dsh,
		(glm::vec3*)dL_dmean3D,
		(glm::vec2*)dL_dscale,
		(glm::vec4*)dL_drot), debug)
}
