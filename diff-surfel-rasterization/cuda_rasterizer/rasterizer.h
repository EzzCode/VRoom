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

#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <functional>
#include <cstdint>
#include <vector_types.h>  // float2, float4

namespace CudaRasterizer
{
	class Rasterizer
	{
	public:

		static void markVisible(
			int P,
			float* means3D,
			float* viewmatrix,
			float* projmatrix,
			bool* present);

		static int forward(
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
			int num_color_feat_channels = 3,
			int* radii = nullptr,
			bool debug = false);

		static void backward(
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
			char* image_buffer,
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
			int num_color_feat_channels = 3);

		// ============= v2 pipeline (gsplat-style, torch::Tensor) ==============
		// Forward: runs preprocess → cumsum → fill_isect_ids → sort → compute_offsets → render
		// Returns individual tensors instead of byte blobs.
		static int forward_v2(
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
			// v2 outputs: per-Gaussian buffers
			float2* means2D,
			float* depths,
			float* transMats,
			float4* normal_opacity,
			uint32_t* tiles_touched,
			// v2 outputs: binning buffers (filled by this function)
			int32_t* tile_offsets,   // [n_tiles]
			int32_t* flatten_ids,    // [n_isects] — filled after sort
			int64_t* isect_ids,      // [n_isects] — scratch for sort
			int64_t* isect_ids_sorted, // [n_isects] — sorted output
			int32_t* flatten_ids_unsorted, // [n_isects] — scratch
			// v2 outputs: per-pixel image buffers
			float* accum_alpha,      // [H*W*3]
			uint32_t* n_contrib,     // [H*W*2]
			bool debug);

		// Backward: uses saved individual tensors for gradient computation.
		static void backward_v2(
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
			// Saved forward tensors (v2)
			const float2* means2D,
			const float* depths,
			const float* transMats,
			const float4* normal_opacity,
			const float* accum_alpha,
			const uint32_t* n_contrib,
			const int32_t* tile_offsets,
			const int32_t* flatten_ids,
			int n_isects,
			// Gradient inputs
			const float* dL_dpix,
			const float* dL_depths,
			// Gradient outputs
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
			int num_color_feat_channels = 3);
	};
};

#endif
