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

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include <functional>

// v2 includes
#include "cuda_rasterizer/forward.h"
#include "cuda_rasterizer/backward.h"
#include "cuda_rasterizer/tile_ops.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#define CHECK_INPUT(x)											\
	AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
	// AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
	auto lambda = [&t](size_t N) {
		t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
	};
	return lambda;
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& colors,
	const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& transMat_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
	const int image_height,
	const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool debug)
{
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
	AT_ERROR("means3D must have dimensions (num_points, 3)");
  }

  
  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;

  CHECK_INPUT(background);
  CHECK_INPUT(means3D);
  CHECK_INPUT(colors);
  CHECK_INPUT(opacity);
  CHECK_INPUT(scales);
  CHECK_INPUT(rotations);
  CHECK_INPUT(transMat_precomp);
  CHECK_INPUT(viewmatrix);
  CHECK_INPUT(projmatrix);
  CHECK_INPUT(sh);
  CHECK_INPUT(campos);

  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  int num_color_feat_channels = (colors.size(0) > 0) ? colors.size(1) : NUM_COLOR_CHANNELS;
  torch::Tensor out_color = torch::full({num_color_feat_channels, H, W}, 0.0, float_opts);
  torch::Tensor out_others = torch::full({3+3+1, H, W}, 0.0, float_opts);
  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
  
  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
  
  int rendered = 0;
  if(P != 0)
  {
	  int M = 0;
	  if(sh.size(0) != 0)
	  {
		M = sh.size(1);
	  }

	  rendered = CudaRasterizer::Rasterizer::forward(
		geomFunc,
		binningFunc,
		imgFunc,
		P, degree, M,
		background.contiguous().data<float>(),
		W, H,
		means3D.contiguous().data<float>(),
		sh.contiguous().data_ptr<float>(),
		colors.contiguous().data<float>(), 
		opacity.contiguous().data<float>(), 
		scales.contiguous().data_ptr<float>(),
		scale_modifier,
		rotations.contiguous().data_ptr<float>(),
		transMat_precomp.contiguous().data<float>(), 
		viewmatrix.contiguous().data<float>(), 
		projmatrix.contiguous().data<float>(),
		campos.contiguous().data<float>(),
		tan_fovx,
		tan_fovy,
		prefiltered,
		out_color.contiguous().data<float>(),
		out_others.contiguous().data<float>(),
		num_color_feat_channels,
		radii.contiguous().data<int>(),
		debug);
  }
  return std::make_tuple(rendered, out_color, out_others, radii, geomBuffer, binningBuffer, imgBuffer);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardCUDA(
	 const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
	const torch::Tensor& colors,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& transMat_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const torch::Tensor& dL_dout_color,
	const torch::Tensor& dL_dout_others,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool debug) 
{

  CHECK_INPUT(background);
  CHECK_INPUT(means3D);
  CHECK_INPUT(radii);
  CHECK_INPUT(colors);
  CHECK_INPUT(scales);
  CHECK_INPUT(rotations);
  CHECK_INPUT(transMat_precomp);
  CHECK_INPUT(viewmatrix);
  CHECK_INPUT(projmatrix);
  CHECK_INPUT(sh);
  CHECK_INPUT(campos);
  CHECK_INPUT(binningBuffer);
  CHECK_INPUT(imageBuffer);
  CHECK_INPUT(geomBuffer);

  const int P = means3D.size(0);
  int num_color_feat_channels = dL_dout_color.size(0);
  const int H = dL_dout_color.size(1);
  const int W = dL_dout_color.size(2);
  
  int M = 0;
  if(sh.size(0) != 0)
  {	
	M = sh.size(1);
  }

  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dcolors = torch::zeros({P, num_color_feat_channels}, means3D.options());
  torch::Tensor dL_dnormal = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dtransMat = torch::zeros({P, 9}, means3D.options());
  torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
  torch::Tensor dL_dscales = torch::zeros({P, 2}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());
  
  if(P != 0)
  {  
	  CudaRasterizer::Rasterizer::backward(P, degree, M, R,
	  background.contiguous().data<float>(),
	  W, H, 
	  means3D.contiguous().data<float>(),
	  sh.contiguous().data<float>(),
	  colors.contiguous().data<float>(),
	  scales.data_ptr<float>(),
	  scale_modifier,
	  rotations.data_ptr<float>(),
	  transMat_precomp.contiguous().data<float>(),
	  viewmatrix.contiguous().data<float>(),
	  projmatrix.contiguous().data<float>(),
	  campos.contiguous().data<float>(),
	  tan_fovx,
	  tan_fovy,
	  radii.contiguous().data<int>(),
	  reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
	  dL_dout_color.contiguous().data<float>(),
	  dL_dout_others.contiguous().data<float>(),
	  dL_dmeans2D.contiguous().data<float>(),
	  dL_dnormal.contiguous().data<float>(),  
	  dL_dopacity.contiguous().data<float>(),
	  dL_dcolors.contiguous().data<float>(),
	  dL_dmeans3D.contiguous().data<float>(),
	  dL_dtransMat.contiguous().data<float>(),
	  dL_dsh.contiguous().data<float>(),
	  dL_dscales.contiguous().data<float>(),
	  dL_drotations.contiguous().data<float>(),
	  debug, num_color_feat_channels);
  }

  return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dtransMat, dL_dsh, dL_dscales, dL_drotations);
}

torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix)
{ 
  const int P = means3D.size(0);
  
  torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));
 
  if(P != 0)
  {
	CudaRasterizer::Rasterizer::markVisible(P,
		means3D.contiguous().data<float>(),
		viewmatrix.contiguous().data<float>(),
		projmatrix.contiguous().data<float>(),
		present.contiguous().data<bool>());
  }
  
  return present;
}

// ===================================================================
// v2 forward: returns individual tensors instead of byte blobs.
// The cumsum is computed inside this function using CUB.
// ===================================================================
std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA_v2(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& colors,
	const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& transMat_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const int image_height,
	const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool debug)
{
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
	AT_ERROR("means3D must have dimensions (num_points, 3)");
  }

  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;

  CHECK_INPUT(background);
  CHECK_INPUT(means3D);
  CHECK_INPUT(colors);
  CHECK_INPUT(opacity);
  CHECK_INPUT(scales);
  CHECK_INPUT(rotations);
  CHECK_INPUT(transMat_precomp);
  CHECK_INPUT(viewmatrix);
  CHECK_INPUT(projmatrix);
  CHECK_INPUT(sh);
  CHECK_INPUT(campos);

  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  int num_color_feat_channels = (colors.size(0) > 0) ? colors.size(1) : NUM_COLOR_CHANNELS;
  torch::Tensor out_color = torch::full({num_color_feat_channels, H, W}, 0.0, float_opts);
  torch::Tensor out_others = torch::full({3+3+1, H, W}, 0.0, float_opts);
  torch::Tensor radii = torch::full({P}, 0, int_opts);

  // Per-Gaussian buffers (individual tensors, not byte blobs)
  torch::Tensor means2D = torch::zeros({P, 2}, float_opts);
  torch::Tensor depths = torch::zeros({P}, float_opts);
  torch::Tensor transMats_buf = torch::zeros({P, 9}, float_opts);
  torch::Tensor normal_opacity = torch::zeros({P, 4}, float_opts);
  torch::Tensor tiles_touched = torch::zeros({P}, int_opts);

  // Per-pixel buffers
  torch::Tensor accum_alpha = torch::zeros({H * W * 3}, float_opts);
  torch::Tensor n_contrib = torch::zeros({H * W * 2}, int_opts.dtype(torch::kInt32));

  int rendered = 0;
  // Tile grid dimensions
  int tile_w = (W + 15) / 16;
  int tile_h = (H + 15) / 16;
  int n_tiles = tile_w * tile_h;
  torch::Tensor tile_offsets = torch::zeros({n_tiles}, int_opts);

  // Placeholder tensors for binning (will be resized after we know num_rendered)
  torch::Tensor flatten_ids;
  torch::Tensor isect_ids_sorted;

  if(P != 0)
  {
	  int M = 0;
	  if(sh.size(0) != 0)
	  {
		M = sh.size(1);
	  }

	  // Step 1: Run preprocess to fill means2D, depths, tiles_touched, etc.
	  // We need a temporary forward_v2 that handles the two-phase approach:
	  //   Phase 1: preprocess → get num_rendered
	  //   Phase 2: allocate binning buffers → fill_isect_ids → sort → offsets → render

	  // Run forward_v2: it will handle the preprocess, cumsum, binning, sort, render
	  // But we need num_rendered first to allocate the binning buffers.
	  // Solution: preprocess fills tiles_touched, we cumsum here, read total, then
	  // call the main pipeline.

	  // First, just run preprocess via forward_v2's internal logic:
	  // We'll pass dummy binning buffers and let forward_v2 handle everything.
	  // Actually, the issue is that forward_v2 expects pre-allocated binning buffers
	  // but we don't know the size yet.

	  // Better approach: run preprocess separately, then cumsum, then allocate, then rest.
	  
	  // For now, we use a two-pass approach:
	  // Pass 1: preprocess to get tiles_touched
	  const float focal_y = H / (2.0f * tan_fovy);
	  const float focal_x = W / (2.0f * tan_fovx);

	  dim3 tile_grid((W + 15) / 16, (H + 15) / 16, 1);

	  bool* clamped_ptr = nullptr;
	  FORWARD::preprocess(
		P, degree, M,
		means3D.contiguous().data<float>(),
		(glm::vec2*)scales.contiguous().data_ptr<float>(),
		scale_modifier,
		(glm::vec4*)rotations.contiguous().data_ptr<float>(),
		opacity.contiguous().data<float>(),
		sh.contiguous().data_ptr<float>(),
		clamped_ptr,
		transMat_precomp.contiguous().data<float>(),
		colors.contiguous().data<float>(),
		viewmatrix.contiguous().data<float>(),
		projmatrix.contiguous().data<float>(),
		(glm::vec3*)campos.contiguous().data<float>(),
		W, H,
		focal_x, focal_y,
		num_color_feat_channels,
		tan_fovx, tan_fovy,
		radii.contiguous().data<int>(),
		(float2*)means2D.contiguous().data<float>(),
		depths.contiguous().data<float>(),
		transMats_buf.contiguous().data<float>(),
		nullptr, // rgb — not needed with colors_precomp
		(float4*)normal_opacity.contiguous().data<float>(),
		tile_grid,
		(uint32_t*)tiles_touched.contiguous().data<int>(),
		prefiltered);

	  // Cumsum on tiles_touched (inclusive) to get offsets
	  // tiles_touched is uint32_t stored as int32 tensor — cumsum in int64 for safety
	  torch::Tensor cum_tiles = tiles_touched.to(torch::kInt64).cumsum(0);
	  
	  // Get total number of tile-Gaussian intersections
	  int64_t num_rendered_64 = cum_tiles[-1].item<int64_t>();
	  rendered = (int)num_rendered_64;

	  if (rendered > 0) {
		  // Allocate binning buffers now that we know the size
		  auto long_opts = means3D.options().dtype(torch::kInt64);
		  torch::Tensor isect_ids_unsorted = torch::empty({rendered}, long_opts);
		  isect_ids_sorted = torch::empty({rendered}, long_opts);
		  torch::Tensor flatten_ids_unsorted = torch::empty({rendered}, int_opts);
		  flatten_ids = torch::empty({rendered}, int_opts);

		  // Fill intersection IDs
		  uint32_t tile_n_bits = 0;
		  { uint32_t n = n_tiles; uint32_t msb = sizeof(n)*4; uint32_t step = msb;
		    while (step > 1) { step /= 2; if (n >> msb) msb += step; else msb -= step; }
		    if (n >> msb) msb++; tile_n_bits = msb; }

		  TILE_OPS::fill_isect_ids(
			P,
			(float2*)means2D.contiguous().data<float>(),
			depths.contiguous().data<float>(),
			cum_tiles.contiguous().data<int64_t>(),
			radii.contiguous().data<int>(),
			isect_ids_unsorted.contiguous().data<int64_t>(),
			flatten_ids_unsorted.contiguous().data<int32_t>(),
			tile_grid, tile_n_bits);

		  // Sort using CUB DoubleBuffer
		  {
			  size_t sort_temp_size = 0;
			  cub::DoubleBuffer<int64_t> d_keys(
				isect_ids_unsorted.data_ptr<int64_t>(),
				isect_ids_sorted.data_ptr<int64_t>());
			  cub::DoubleBuffer<int32_t> d_values(
				flatten_ids_unsorted.data_ptr<int32_t>(),
				flatten_ids.data_ptr<int32_t>());

			  cub::DeviceRadixSort::SortPairs(
				nullptr, sort_temp_size,
				d_keys, d_values,
				rendered, 0, 32 + tile_n_bits);

			  torch::Tensor sort_temp = torch::empty({(int64_t)sort_temp_size},
				means3D.options().dtype(torch::kByte));

			  cub::DeviceRadixSort::SortPairs(
				sort_temp.data_ptr(),
				sort_temp_size,
				d_keys, d_values,
				rendered, 0, 32 + tile_n_bits);

			  // Handle DoubleBuffer swap
			  if (d_keys.Current() != isect_ids_sorted.data_ptr<int64_t>()) {
				cudaMemcpy(isect_ids_sorted.data_ptr<int64_t>(), d_keys.Current(),
					rendered * sizeof(int64_t), cudaMemcpyDeviceToDevice);
			  }
			  if (d_values.Current() != flatten_ids.data_ptr<int32_t>()) {
				cudaMemcpy(flatten_ids.data_ptr<int32_t>(), d_values.Current(),
					rendered * sizeof(int32_t), cudaMemcpyDeviceToDevice);
			  }
		  }

		  // Compute tile offsets
		  cudaMemset(tile_offsets.data_ptr<int32_t>(), 0, n_tiles * sizeof(int32_t));
		  TILE_OPS::compute_tile_offsets(
			rendered,
			isect_ids_sorted.contiguous().data<int64_t>(),
			tile_offsets.data_ptr<int32_t>(),
			n_tiles, tile_n_bits);

		  // Render
		  const float* feature_ptr = (colors.size(0) > 0) ? colors.contiguous().data<float>() : nullptr;
		  const float* transMat_ptr = (transMat_precomp.size(0) > 0) ? transMat_precomp.contiguous().data<float>() : transMats_buf.contiguous().data<float>();

		  dim3 block(BLOCK_X, BLOCK_Y, 1);
		  FORWARD::render_v2(
			tile_grid, block,
			tile_offsets.data_ptr<int32_t>(),
			flatten_ids.data_ptr<int32_t>(),
			rendered,
			W, H,
			focal_x, focal_y,
			num_color_feat_channels,
			(float2*)means2D.contiguous().data<float>(),
			feature_ptr,
			transMat_ptr,
			depths.contiguous().data<float>(),
			(float4*)normal_opacity.contiguous().data<float>(),
			accum_alpha.data_ptr<float>(),
			(uint32_t*)n_contrib.data_ptr<int>(),
			background.contiguous().data<float>(),
			out_color.data_ptr<float>(),
			out_others.data_ptr<float>());
	  } else {
		  // No intersections — create empty tensors
		  flatten_ids = torch::empty({0}, int_opts);
		  isect_ids_sorted = torch::empty({0}, means3D.options().dtype(torch::kInt64));
	  }
  } else {
	  flatten_ids = torch::empty({0}, int_opts);
	  isect_ids_sorted = torch::empty({0}, means3D.options().dtype(torch::kInt64));
  }

  return std::make_tuple(
	rendered, out_color, out_others, radii,
	means2D, depths, transMats_buf, normal_opacity,
	accum_alpha, n_contrib, tile_offsets, flatten_ids);
}


// ===================================================================
// v2 backward: uses individual saved tensors.
// ===================================================================
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansBackwardCUDA_v2(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
	const torch::Tensor& colors,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& transMat_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const torch::Tensor& dL_dout_color,
	const torch::Tensor& dL_dout_others,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	// Saved individual tensors from forward_v2
	const torch::Tensor& means2D,
	const torch::Tensor& depths,
	const torch::Tensor& transMats_buf,
	const torch::Tensor& normal_opacity,
	const torch::Tensor& accum_alpha,
	const torch::Tensor& n_contrib_t,
	const torch::Tensor& tile_offsets,
	const torch::Tensor& flatten_ids,
	const int R, // num_rendered (n_isects)
	const bool debug)
{
  CHECK_INPUT(background);
  CHECK_INPUT(means3D);
  CHECK_INPUT(radii);
  CHECK_INPUT(colors);
  CHECK_INPUT(scales);
  CHECK_INPUT(rotations);
  CHECK_INPUT(transMat_precomp);
  CHECK_INPUT(viewmatrix);
  CHECK_INPUT(projmatrix);
  CHECK_INPUT(sh);
  CHECK_INPUT(campos);

  const int P = means3D.size(0);
  int num_color_feat_channels = dL_dout_color.size(0);
  const int H = dL_dout_color.size(1);
  const int W = dL_dout_color.size(2);

  int M = 0;
  if(sh.size(0) != 0)
  {
	M = sh.size(1);
  }

  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dcolors = torch::zeros({P, num_color_feat_channels}, means3D.options());
  torch::Tensor dL_dnormal = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dtransMat = torch::zeros({P, 9}, means3D.options());
  torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
  torch::Tensor dL_dscales = torch::zeros({P, 2}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());

  if(P != 0)
  {
	  CudaRasterizer::Rasterizer::backward_v2(
		P, degree, M, R,
		background.contiguous().data<float>(),
		W, H,
		means3D.contiguous().data<float>(),
		sh.contiguous().data<float>(),
		colors.contiguous().data<float>(),
		scales.data_ptr<float>(),
		scale_modifier,
		rotations.data_ptr<float>(),
		transMat_precomp.contiguous().data<float>(),
		viewmatrix.contiguous().data<float>(),
		projmatrix.contiguous().data<float>(),
		campos.contiguous().data<float>(),
		tan_fovx, tan_fovy,
		radii.contiguous().data<int>(),
		// Saved individual tensors
		(const float2*)means2D.contiguous().data<float>(),
		depths.contiguous().data<float>(),
		transMats_buf.contiguous().data<float>(),
		(const float4*)normal_opacity.contiguous().data<float>(),
		accum_alpha.contiguous().data<float>(),
		(const uint32_t*)n_contrib_t.contiguous().data<int>(),
		tile_offsets.contiguous().data<int>(),
		flatten_ids.contiguous().data<int>(),
		R,
		// Gradient inputs
		dL_dout_color.contiguous().data<float>(),
		dL_dout_others.contiguous().data<float>(),
		// Gradient outputs
		dL_dmeans2D.contiguous().data<float>(),
		dL_dnormal.contiguous().data<float>(),
		dL_dopacity.contiguous().data<float>(),
		dL_dcolors.contiguous().data<float>(),
		dL_dmeans3D.contiguous().data<float>(),
		dL_dtransMat.contiguous().data<float>(),
		dL_dsh.contiguous().data<float>(),
		dL_dscales.contiguous().data<float>(),
		dL_drotations.contiguous().data<float>(),
		debug, num_color_feat_channels);
  }

  return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D,
                         dL_dtransMat, dL_dsh, dL_dscales, dL_drotations);
}

