#include "../includes/bwd.cuh"
#include "../includes/utils.cuh"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Rendering backpropagation kernel.
// 1 pixel per thread and 1 tile per block.
template <uint8_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_SIZE)
    render_kernel_bwd(
        // __restrict__ tells the compiler that no two pointers alias the same memory.
        const int img_W, const int img_H,      // Image width and height
        const float *__restrict__ colors_feat, // Concatenation of colors and features per surfel
        const float *__restrict__ background,  // Background values
        // Preprocess buffers
        const float2 *__restrict__ projected_centers, // Mapped pixel locations of each surfel's center
        const float3 *__restrict__ splat2pix_mats,    // Splat to pixel space matrices buffer for each surfel
        const float4 *__restrict__ normal_opacity,    // Normals (camera space) concatenated with opacity for each surfel
        // Binning buffers
        const uint32_t *__restrict__ sorted_surfel_indices, // sorted surfel indices
        // Image buffers
        const uint2 *__restrict__ tile_ranges,               // Per-tile [start, end) index ranges
        const uint32_t *__restrict__ contrib_state,          // Indices of the last and median surfel contributing to a pixel
        const float *__restrict__ transmittance_and_moments, // Transmittance, first and seconds moments of depth (for distortion loss)
        // Input Gradients
        const float *__restrict__ grad_rendered_color_feat, // Pytorch rendered pixel's gradients (image rendering loss)
        const float *__restrict__ grad_rendered_aux,        // Pytorch rendered aux outputs gradients
        // Output Gradients
        float *__restrict__ grad_splat2pix_mats_buff,     // Computed splat to pixel space matrix gradients
        float2 *__restrict__ grad_projected_centers_buff, // Computed projected (pix spcae) surfel center gradients
        float3 *__restrict__ grad_normal_buff,            // Computed surfel normals gradients
        float *__restrict__ grad_opacity_buff,            // Computed surfel opacity gradients
        float *__restrict__ grad_colors_feat_buff         // Computed surfel color gradients
    )
{
    // Define a custom cache size for shared memory buffers.
    // Cache size is controlled by the number of colors+features channels
    constexpr int cache_size = (CHANNELS <= 32) ? 256 : ((CHANNELS <= 64) ? 128 : 64);

    // For larger channel counts, pad the shared memory to prevent bank conflicts.
    constexpr int color_feat_stride = (CHANNELS > 32) ? (cache_size + 1) : cache_size;

    // Find tile (block) ID for the current thread
    auto thread_block_ids = cg::this_thread_block(); // Stores the current thread and block (aka group) IDs

    // Find starting pixel for this tile (block)
    uint2 pixel_min = {thread_block_ids.group_index().x * BLOCK_DIM_X,
                       thread_block_ids.group_index().y * BLOCK_DIM_Y};

    // Find current pixel
    uint2 pixel = {(pixel_min.x + thread_block_ids.thread_index().x),
                   (pixel_min.y + thread_block_ids.thread_index().y)};

    // Validate pixel
    const bool valid_pixel = pixel.x < img_W && pixel.y < img_H;

    // Load sorted surfel ranges for this tile
    const uint2 tile_range = tile_ranges[thread_block_ids.group_index().y * DIV_CEIL(img_W, BLOCK_DIM_X) +
                                         thread_block_ids.group_index().x];

    // Find how many rounds Required to load all surfel batches to finish rendering
    const int rounds = DIV_CEIL(tile_range.y - tile_range.x, cache_size); // We load cache_size every round
    int rem_surfels = tile_range.y - tile_range.x;                        // track remaining surfels to render

    // Allocate the shared (among block's threads) memory buffers for batching VRAM access.
    // Batch size is cache_size so each thread is responsible for fetching 1 data item.
    __shared__ int surfel_idx_batch[cache_size];
    __shared__ float2 projected_centers_batch[cache_size];
    __shared__ float4 normal_opacity_batch[cache_size];
    __shared__ float3 splat2pix_mat_col0_batch[cache_size];
    __shared__ float3 splat2pix_mat_col1_batch[cache_size];
    __shared__ float3 splat2pix_mat_col2_batch[cache_size];
    __shared__ float color_feat_batch[CHANNELS * color_feat_stride];

    // Compute flattened pixel id
    int pixel_idx = img_W * pixel.y + pixel.x;

    /*
    In backward, we loop BACK TO FRONT. So we start by collecting the last
    stored values, reverse computations, and accumulate gradients.
    */

    // Fetch the final transmittance
    const float final_transmittance = transmittance_and_moments[pixel_idx];
    float transmittance = final_transmittance;

    // Track the currently processed surfel.
    // Start at the absolute possible last surfel and
    // verify if it was processed in fwd before using.
    int contributor_idx = rem_surfels; // (1-based)

    // Fetch the actual final contributor ID
    int last_contributor_idx = contrib_state[pixel_idx]; // (1-based)

    // Track accumulated colors+features behind the current
    // processed surfel
    float residual_color_feat_acc[CHANNELS] = {0};
    float prev_color_feat[CHANNELS] = {0};

    // Track previous surfel's alpha
    float prev_alpha = 0;

    // Fetch pytorch-computed pixel gradient and precompute background dot product
    float grad_pixel_color_feat[CHANNELS];
    float bg_dot_pixel_grad = 0.f;

    if (valid_pixel)
    {
        for (int ch = 0; ch < CHANNELS; ch++)
        {
            grad_pixel_color_feat[ch] = grad_rendered_color_feat[pixel_idx +
                                                                 ch * img_H * img_W];
            // Fused Multiply-Add (FMA)
            // ldg routes VRAM access through read-only data (texture) cache
            // which is better for broadcasting to other threads and reducing L1 cache pressure.
            bg_dot_pixel_grad = __fmaf_rn(__ldg(&background[ch]),
                                          grad_pixel_color_feat[ch],
                                          bg_dot_pixel_grad);
        }
    }

    // Prepare for computing auxiliary outputs' gradients
#if RENDER_AUX
    // Define per-pixel gradients and accumulators for aux outputs

    // 1. Depth gradients
    float grad_pixel_depth;
    float prev_depth = 0.f;
    float residual_depth_acc = 0.f;

    // 2. Alpha gradients
    float grad_pixel_alpha_acc;
    float residual_alpha_acc = 0.f;

    // 3. Normal gradients
    float3 grad_pixel_normal;
    float3 prev_normal = {0.f, 0.f, 0.f};
    float3 residual_normal_acc = {0.f};

    // 4. Distortion loss gradients
    float grad_pixel_distortion;
    float final_m1 = 0; // final first moment of depth
    float final_m2 = 0; // final second moment of depth
    float residual_grad_distortion_acc = 0.f;

    // Fetch from VRAM only if the pixel is valid
    if (valid_pixel)
    {
        grad_pixel_depth = grad_rendered_aux[pixel_idx + DEPTH_OFFSET * img_H * img_W];
        grad_pixel_alpha_acc = grad_rendered_aux[pixel_idx + ALPHA_OFFSET * img_H * img_W];
        grad_pixel_normal = {grad_rendered_aux[pixel_idx + (NORMAL_OFFSET + 0) * img_H * img_W],
                             grad_rendered_aux[pixel_idx + (NORMAL_OFFSET + 1) * img_H * img_W],
                             grad_rendered_aux[pixel_idx + (NORMAL_OFFSET + 2) * img_H * img_W]};
        grad_pixel_distortion = grad_rendered_aux[pixel_idx + DISTORTION_OFFSET * img_H * img_W];

        final_m1 = transmittance_and_moments[pixel_idx + img_H * img_W];
        final_m2 = transmittance_and_moments[pixel_idx + 2 * img_H * img_W];
    }
#endif

    // Warp-level optimization for warp-level early exit. #warps = BLOCK_SIZE / 32.
    // Also, track the last contributor ID per warp.
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(thread_block_ids);
    const int warp_last_contributor = cg::reduce(warp, last_contributor_idx, cg::greater<int>());

    // Loop over loaded VRAM batches until the tile (block) is done
    for (int round = 0; round < rounds; round++, rem_surfels -= cache_size)
    {
        // We can't do block level early-exits as we already start from the last surfel
        // and move in reverse to the front surfel.

        // Sync the tile (block) before fetching (Wait for all warps to finish previous round)
        thread_block_ids.sync();

        // Safely fetch data as thread_idx can become larger than cache_size causing corruption.
        if (cache_size == BLOCK_SIZE || thread_block_ids.thread_rank() < cache_size)
        {
            // Tile (block) collaborates on fetching data from VRAM.
            // Fetching is done BACK TO FRONT.
            int progress = round * cache_size + thread_block_ids.thread_rank();
            if (tile_range.x + progress < tile_range.y)
            {
                const int glob_surf_idx = sorted_surfel_indices[tile_range.y - progress - 1];
                surfel_idx_batch[thread_block_ids.thread_rank()] = glob_surf_idx;
                projected_centers_batch[thread_block_ids.thread_rank()] = projected_centers[glob_surf_idx];
                normal_opacity_batch[thread_block_ids.thread_rank()] = normal_opacity[glob_surf_idx];
                // splat2pix matrix is col-major
                splat2pix_mat_col0_batch[thread_block_ids.thread_rank()] = splat2pix_mats[3 * glob_surf_idx + 0];
                splat2pix_mat_col1_batch[thread_block_ids.thread_rank()] = splat2pix_mats[3 * glob_surf_idx + 1];
                splat2pix_mat_col2_batch[thread_block_ids.thread_rank()] = splat2pix_mats[3 * glob_surf_idx + 2];
            }
        }

        // Collaborate on fetching colors+features from VRAM.
        // Requires a separate loop since each thread must fetch #CHANNELS items.
        for (int load_idx = thread_block_ids.thread_rank(); load_idx < CHANNELS * cache_size; load_idx += BLOCK_SIZE)
        {
            int g_idx = load_idx % cache_size;
            int ch_idx = load_idx / cache_size;
            if (tile_range.x + round * cache_size + g_idx < tile_range.y)
            {
                const int surfel_idx = sorted_surfel_indices[tile_range.y - (round * cache_size + g_idx) - 1];
                color_feat_batch[ch_idx * color_feat_stride + g_idx] = colors_feat[surfel_idx * CHANNELS + ch_idx];
            }
        }

        // Sync threads in tile (block) before accessing the fetched data
        thread_block_ids.sync();

        // Loop over fetched batch of surfels
        for (int batch_surf_idx = 0; batch_surf_idx < min(cache_size, rem_surfels); batch_surf_idx++)
        {
            // Track position in surfel's range
            contributor_idx--;

            // Warp-level early exit: if no thread in this warp
            // has remaining contributors, skip this surfel entirely.
            if (contributor_idx >= (uint32_t)warp_last_contributor)
                continue;

            /*
            WARNING: We MUST NOT perform thread-level early-exits
            since we perform warp reduce at the end of the loop which
            requires ALL threads to be present to prevent deadlock.
            However, we can gate parts of the computations if the thread
            is invalid.
            */

            // Verify whether or not to perform computations for this pixel with this
            // specific surfel.
            bool valid_surfel = valid_pixel && (contributor_idx < (uint32_t)last_contributor_idx);

            // Compute ray-surfel intersection again and continuously verify
            // if the surfel can be invalidated.

            // 1. Load surfel center pixel and the splat2pix matrix
            const float2 center_pixel = projected_centers_batch[batch_surf_idx];
            const float3 splat2pix_mat_col0 = splat2pix_mat_col0_batch[batch_surf_idx];
            const float3 splat2pix_mat_col1 = splat2pix_mat_col1_batch[batch_surf_idx];
            const float3 splat2pix_mat_col2 = splat2pix_mat_col2_batch[batch_surf_idx];

            // 1.5 define intermediate variables between validation blocks
            float3 ray_cross, plane_x, plane_y, normal;
            float2 local_uv, pixel_offset;
            float dist_3d_sq, dist_2d_sq, depth, gaussian_dist_sq,
                opacity, power, exp_power, alpha;

            if (valid_surfel)
            {
                // 2. Define two planes representing the vertical and horizontal
                // lines passing through the exact center of this pixel, transformed into
                // the surfel's (u, v) space.
                plane_x = pixel.x * splat2pix_mat_col2 - splat2pix_mat_col0;
                plane_y = pixel.y * splat2pix_mat_col2 - splat2pix_mat_col1;

                // 3. The cross product of these two planes gives the ray passing through the pixel
                ray_cross = cross_product(plane_x, plane_y);

                // Validate progress so far (we skipped edge-on surfels in FWD)
                if (ray_cross.z == 0.f)
                    valid_surfel = false;
            }

            if (valid_surfel)
            {
                // 4. Perspective division to find the exact (u, v) ray-surfel isect
                // on the surfel's local plane.
                local_uv = {ray_cross.x / ray_cross.z, ray_cross.y / ray_cross.z};

                // 5. Calculate squared distance from the surfel's center (0,0 in surfel's space)
                dist_3d_sq = (local_uv.x * local_uv.x) + (local_uv.y * local_uv.y);

                // 6. Calculate 2D (pixel distances) low-pass filter distance to prevent sub-pixel instability.
                // We dilate the pixel in order to prevent aliasing.
                pixel_offset = {center_pixel.x - (float)pixel.x, center_pixel.y - (float)pixel.y};
                dist_2d_sq = FILTER_INV_SQ * (pixel_offset.x * pixel_offset.x + pixel_offset.y * pixel_offset.y);

                // 7. Use the minimum of true 3D distance and 2D filtered distance
                gaussian_dist_sq = min(dist_3d_sq, dist_2d_sq);

                // Compute surfel depth
                depth = local_uv.x * splat2pix_mat_col2.x + local_uv.y * splat2pix_mat_col2.y + splat2pix_mat_col2.z;
                if (depth < NEAR_PLANE)
                    valid_surfel = false; // We skipped surfels that are too close to camera in FWD
            }

            if (valid_surfel)
            {
                // Fetch normals, opacity
                const float4 norm_opa = normal_opacity_batch[batch_surf_idx];
                normal = {norm_opa.x, norm_opa.y, norm_opa.z};
                opacity = norm_opa.w;

                // Compute alpha (effective opacity at the calculated center-ray_isect distance)
                power = -0.5f * gaussian_dist_sq;
                if (power > 0.f)
                    valid_surfel = false; // Opacity can only decrease as center-ray_isect dist increases
            }

            if (valid_surfel)
            {
                exp_power = exp(power);
                alpha = min(.99f, opacity * exp_power);
                if (alpha < 1.f / 255.f)
                    valid_surfel = false; // Effectively transparent surfel
            }

            // Warp-level early exit if all threads in the warp
            // have an invalid surfel.
            if (!warp.any(valid_surfel))
                continue;

            // Define gradient accumulators
            float grad_colors_feat[CHANNELS] = {0.f};
            float3 grad_normal = {0.f, 0.f, 0.f};
            float grad_splat2pix_mats[9] = {0.f};
            float2 grad_projected_centers = {0.f, 0.f};
            float grad_opacity = 0.f;

            // Precomputations
            float inv_one_minus_alpha = __fdividef(1.f, 1.f - alpha);

            // Compute gradients for valid surfels
            if (valid_surfel)
            {
                // Recover transmittance before blending with the current surfel
                transmittance = transmittance * inv_one_minus_alpha;

                // Blending weight for surfel.
                // It is also rendered color/feat loss w.r.t. surfel's color/feat.
                const float blending_weight = alpha * transmittance;

                // To compute alpha gradient, we must do so through all its paths.

                // Define the alpha (accumulated opacity) gradient accumulator
                float grad_alpha = 0.f;

                // Path A: Recursively accumulate gradients from colors+features channels
                for (int ch = 0; ch < CHANNELS; ch++)
                {
                    const float curr_color_feat = color_feat_batch[batch_surf_idx + ch * color_feat_stride];
                    residual_color_feat_acc[ch] = prev_alpha * prev_color_feat[ch] +
                                                  (1.f - prev_alpha) * residual_color_feat_acc[ch];
                    grad_alpha += (curr_color_feat - residual_color_feat_acc[ch]) *
                                  grad_pixel_color_feat[ch];
                    prev_color_feat[ch] = curr_color_feat;

                    // Also compute gradients w.r.t. colors+features
                    grad_colors_feat[ch] = blending_weight * grad_pixel_color_feat[ch];
                }

                // Path B: Accumulate gradients from aux outputs.
#if RENDER_AUX
                // 1. Depth gradients
                residual_depth_acc = prev_alpha * prev_depth + (1.f - prev_alpha) * residual_depth_acc;
                prev_depth = depth;
                grad_alpha += (depth - residual_depth_acc) * grad_pixel_depth;

                // 2. Alpha gradients
                residual_alpha_acc = prev_alpha + (1.f - prev_alpha) * residual_alpha_acc;
                grad_alpha += (1.f - residual_alpha_acc) * grad_pixel_alpha_acc;

                // 3. Normal gradients
                residual_normal_acc = prev_alpha * prev_normal +
                                      (1.f - prev_alpha) * residual_normal_acc;
                prev_normal = normal;
                grad_normal = blending_weight * grad_pixel_normal; // gradients of normal in cam-space
                {
                    const float3 grad_alpha_normal_intermediate = (normal - residual_normal_acc) *
                                                                  grad_pixel_normal;
                    grad_alpha += grad_alpha_normal_intermediate.x +
                                  grad_alpha_normal_intermediate.y +
                                  grad_alpha_normal_intermediate.z;
                }

                // 4. Distortion loss gradients
                {
                    const float normalized_depth = DEPTH_NORM_SCALE *
                                                   (1 - NEAR_PLANE / depth);
                    const float final_alpha_acc = 1.f - final_transmittance;
                    const float grad_distortion_acc = grad_pixel_distortion *
                                                      (final_m2 +
                                                       normalized_depth * normalized_depth * final_alpha_acc -
                                                       2 * normalized_depth * final_m1);
                    grad_alpha += grad_distortion_acc - residual_grad_distortion_acc;
                    residual_grad_distortion_acc = grad_distortion_acc * alpha +
                                                   (1 - alpha) * residual_grad_distortion_acc;
                }
#endif
                // Path C: Accumulate gradients from background.
                grad_alpha *= transmittance;
                grad_alpha += (-final_transmittance * inv_one_minus_alpha) * bg_dot_pixel_grad;

                // Update previous alpha
                prev_alpha = alpha;

                // Compute gradients w.r.t. surfel opacity
                grad_opacity = exp_power * grad_alpha;

                // Compute gradients w.r.t. surfel's geometry.

                // NOTE: We decouple depth gradient from geometry
                // to stabilize training. However, depth losses still
                // flow through opacity gradients.

                const float grad_exp_power = opacity * grad_alpha;

                if (dist_3d_sq <= dist_2d_sq)
                {
                    // Typically, true 3D distance dominates.

                    // Compute gradients through ray-surfel intersection.
                    const float2 grad_local_uv = {
                        grad_exp_power * -exp_power * local_uv.x,
                        grad_exp_power * -exp_power * local_uv.y};

                    // Backpropagate through perspective division
                    const float grad_ray_cross_x = grad_local_uv.x / ray_cross.z;
                    const float grad_ray_cross_y = grad_local_uv.y / ray_cross.z;
                    const float3 grad_ray_cross = {
                        grad_ray_cross_x,
                        grad_ray_cross_y,
                        -(grad_ray_cross_x * local_uv.x + grad_ray_cross_y * local_uv.y)};

                    // Backpropagate through the cross product of the
                    // planes defining the surfel.
                    const float3 grad_plane_x = cross_product(plane_y, grad_ray_cross);
                    const float3 grad_plane_y = cross_product(grad_ray_cross, plane_x);

                    // Backpropagate through splat 2 pixel space matrix

                    // 1. Column 1 (col-major indexing)
                    grad_splat2pix_mats[0] = -grad_plane_x.x;
                    grad_splat2pix_mats[1] = -grad_plane_x.y;
                    grad_splat2pix_mats[2] = -grad_plane_x.z;

                    // 2. Column 2 (col-major indexing)
                    grad_splat2pix_mats[3] = -grad_plane_y.x;
                    grad_splat2pix_mats[4] = -grad_plane_y.y;
                    grad_splat2pix_mats[5] = -grad_plane_y.z;

                    // 3. Column 3 (col-major indexing)
                    grad_splat2pix_mats[6] = pixel.x * grad_plane_x.x + pixel.y * grad_plane_y.x;
                    grad_splat2pix_mats[7] = pixel.x * grad_plane_x.y + pixel.y * grad_plane_y.y;
                    grad_splat2pix_mats[8] = pixel.x * grad_plane_x.z + pixel.y * grad_plane_y.z;
                }
                else
                {
                    // Rare case that the 2D low-pass filter distance dominates (subpixel surfel)

                    // Backpropagate through screen-space pixel offsets (pixel distances)
                    grad_projected_centers.x = grad_exp_power * -exp_power * FILTER_INV_SQ * pixel_offset.x;
                    grad_projected_centers.y = grad_exp_power * -exp_power * FILTER_INV_SQ * pixel_offset.y;

                    // No gradients are computed for the splat2pix matrix
                }
            }

            // Accumulate gradients to global memory.

            // 1. Accumulate gradients for warp through warp reduce.
            // 1.1. Reduce surfel colors+features grad
            for (int ch = 0; ch < CHANNELS; ch++)
                grad_colors_feat[ch] = cg::reduce(warp, grad_colors_feat[ch], cg::plus<float>());
            // 1.2. Reduce splat 2 pixel space matrix grad
            for (int idx = 0; idx < 9; idx++)
                grad_splat2pix_mats[idx] = cg::reduce(warp, grad_splat2pix_mats[idx], cg::plus<float>());
            // 1.3. Reduce normals grad
            grad_normal.x = cg::reduce(warp, grad_normal.x, cg::plus<float>());
            grad_normal.y = cg::reduce(warp, grad_normal.y, cg::plus<float>());
            grad_normal.z = cg::reduce(warp, grad_normal.z, cg::plus<float>());
            // 1.4. Reduce projected centers grad
            grad_projected_centers.x = cg::reduce(warp, grad_projected_centers.x, cg::plus<float>());
            grad_projected_centers.y = cg::reduce(warp, grad_projected_centers.y, cg::plus<float>());
            // 1.5. Reduce surfel opacity grad
            grad_opacity = cg::reduce(warp, grad_opacity, cg::plus<float>());

            // 2. Lane 0 of each warp writes to global memory (reduces locking overhead)
            if (warp.thread_rank() == 0)
            {
                const int glob_surf_idx = surfel_idx_batch[batch_surf_idx];

                // Only accumulate gradients if they are non-zero
                // to save locking overhead.
                // 2.1. Write to global colors+features grad buffer
                for (int ch = 0; ch < CHANNELS; ch++)
                    if (grad_colors_feat[ch] != 0.f)
                        atomicAdd(&(grad_colors_feat_buff[ch + glob_surf_idx * CHANNELS]), grad_colors_feat[ch]);
                // 2.2. Write to global splat 2 pixel space matrix grad buffer
                for (int idx = 0; idx < 9; idx++)
                    if (grad_splat2pix_mats[idx] != 0.f)
                        atomicAdd(&(grad_splat2pix_mats_buff[idx + glob_surf_idx * 9]), grad_splat2pix_mats[idx]);
                // 2.3. Write to global normals grad buffer
                if (grad_normal.x != 0.f)
                    atomicAdd(&(grad_normal_buff[glob_surf_idx].x), grad_normal.x);
                if (grad_normal.y != 0.f)
                    atomicAdd(&(grad_normal_buff[glob_surf_idx].y), grad_normal.y);
                if (grad_normal.z != 0.f)
                    atomicAdd(&(grad_normal_buff[glob_surf_idx].z), grad_normal.z);
                // 2.4. Write to global projected centers grad buffer
                if (grad_projected_centers.x != 0.f)
                    atomicAdd(&(grad_projected_centers_buff[glob_surf_idx].x), grad_projected_centers.x);
                if (grad_projected_centers.y != 0.f)
                    atomicAdd(&(grad_projected_centers_buff[glob_surf_idx].y), grad_projected_centers.y);
                // 2.5. Write to global opacity grads buffer
                if (grad_opacity != 0.f)
                    atomicAdd(&(grad_opacity_buff[glob_surf_idx]), grad_opacity);
            }
        }
    }
}

// Backpropagate gradients through rendering process
void BWD::render(
    const int img_W, const int img_H,  // Image width and height
    const int num_color_feat_channels, // Number of channels in the concat of colors + features
    const float *colors_feat,          // Concatenation of colors and features per surfel
    const float *background,           // Background values
    // Preprocess buffers
    const float2 *projected_centers, // Mapped pixel locations of each surfel's center
    const float3 *splat2pix_mats,    // Splat to pixel space matrices buffer for each surfel
    const float4 *normal_opacity,    // Normals (camera space) concatenated with opacity for each surfel
    // Binning buffers
    const uint32_t *sorted_surfel_indices, // sorted surfel indices
    // Image buffers
    const uint2 *tile_ranges,               // Per-tile [start, end) index ranges
    const uint32_t *contrib_state,          // Indices of the last and median surfel contributing to a pixel
    const float *transmittance_and_moments, // Transmittance, first and seconds moments of depth (for distortion loss)
    // Input Gradients
    const float *grad_rendered_color_feat, // Pytorch rendered pixel's gradients (image rendering loss)
    const float *grad_rendered_aux,        // Pytorch rendered aux outputs gradients
    // Output Gradients
    float *grad_splat2pix_mats_buff,     // Computed splat to pixel space matrix gradients
    float2 *grad_projected_centers_buff, // Computed projected (pix spcae) surfel center gradients
    float3 *grad_normal_buff,            // Computed surfel normals gradients
    float *grad_opacity_buff,            // Computed surfel opacity gradients
    float *grad_colors_feat_buff         // Computed surfel color gradients
)
{
    // Define kernel structure
    dim3 tile_grid(DIV_CEIL(img_W, BLOCK_DIM_X), DIV_CEIL(img_H, BLOCK_DIM_Y), 1);
    dim3 tile(BLOCK_DIM_X, BLOCK_DIM_Y, 1);

#define __RENDER_CALL_(CHANNELS)                          \
    case CHANNELS:                                        \
        render_kernel_bwd<CHANNELS><<<tile_grid, tile>>>( \
            img_W, img_H,                                 \
            colors_feat,                                  \
            background,                                   \
            projected_centers,                            \
            splat2pix_mats,                               \
            normal_opacity,                               \
            sorted_surfel_indices,                        \
            tile_ranges,                                  \
            contrib_state,                                \
            transmittance_and_moments,                    \
            grad_rendered_color_feat,                     \
            grad_rendered_aux,                            \
            grad_splat2pix_mats_buff,                     \
            grad_projected_centers_buff,                  \
            grad_normal_buff,                             \
            grad_opacity_buff,                            \
            grad_colors_feat_buff);                       \
        break;

    switch (num_color_feat_channels)
    {
        __RENDER_CALL_(1)
        __RENDER_CALL_(3)
        __RENDER_CALL_(4)
        __RENDER_CALL_(8)
        __RENDER_CALL_(16)
        __RENDER_CALL_(32)
    default:
        break; // Should never reach here. Python pads / batches to provide only supported sizes
    }

#undef __RENDER_CALL_
}

// Compute Splat to pixel space transformation matrix AND its intermediates.
// This matrix maps the local surfel plane to the pixel coordinate space.
__forceinline__ __device__ void recompute_splat2pix_intermediates(
    const float3 &point_world_space,  // Surfel coordinate in world space
    const glm::vec2 scale_vec,        // Scale vector
    const float glob_scale_mod,       // Scale vector
    const glm::vec4 quat,             // Quaternion
    const float *w2cam_mat,           // World to Cam space matrix
    const float *w2clip_mat,          // World to Clip space matrix
    const int img_W, const int img_H, // Image width and height
    glm::mat3 &splat2pix_mat,         // Computed splat to pixel space matrix
    glm::mat3 &rot_mat,               // Rotation matrix from quaternions
    glm::mat3 &scale_mat,             // Scale matrix from scale vectors
    glm::mat3x4 &world2pix_mat,       // World to pixel space matrix
    float3 &normal                    // Computed surfel normal in camera space
)
{
    // Creat the local frame of the surfel (rotation * scaling)
    rot_mat = transform_quat_to_rotmat(quat);
    scale_mat = create_scale_mat(scale_vec, glob_scale_mod);
    glm::mat3 local_frame_mat = rot_mat * scale_mat;

    // Create the splat to world space matrix. GLM is col-major
    glm::mat3x4 splat2world_mat = glm::mat3x4(
        glm::vec4(local_frame_mat[0], 0.0),
        glm::vec4(local_frame_mat[1], 0.0),
        glm::vec4(point_world_space.x, point_world_space.y, point_world_space.z, 1));

    // This is just the GLM-formatted w2clip_mat.
    glm::mat4 glm_world2clip_mat = glm::mat4(
        glm::vec4(w2clip_mat[0], w2clip_mat[4], w2clip_mat[8], w2clip_mat[12]),
        glm::vec4(w2clip_mat[1], w2clip_mat[5], w2clip_mat[9], w2clip_mat[13]),
        glm::vec4(w2clip_mat[2], w2clip_mat[6], w2clip_mat[10], w2clip_mat[14]),
        glm::vec4(w2clip_mat[3], w2clip_mat[7], w2clip_mat[11], w2clip_mat[15]));

    // Create the clip to pixel [0, W-1] x [0, H-1] space matrix
    glm::mat3x4 clip2pix_mat = glm::mat3x4(
        glm::vec4(img_W / 2.f, 0.f, 0.f, (img_W - 1) / 2.f),
        glm::vec4(0.f, img_H / 2.f, 0.f, (img_H - 1) / 2.f),
        glm::vec4(0.f, 0.f, 0.f, 1.f));

    // Compute splat to pixel space matrix
    world2pix_mat = glm_world2clip_mat * clip2pix_mat;
    splat2pix_mat = glm::transpose(splat2world_mat) * world2pix_mat;

    // Compute surfel normal in cam space
    normal = rotate_vector({local_frame_mat[2].x, local_frame_mat[2].y, local_frame_mat[2].z}, w2cam_mat);
}

// Backpropagate gradients through the AABB computation.
// Computes gradients w.r.t. splat to pixel space matrix.
__forceinline__ __device__ void grad_compute_aabb(
    const glm::mat3 &splat2pix_mat,      // Recomputed splat to pixel space matrix
    const float2 &grad_projected_center, // Gradients of projected centers (pix space)
    glm::mat3 &grad_splat2pix_mat        // Computed gradients of splat 2 pixel space matrices
)
{
    // Pixel cutoff. 3 standard deviations of the gaussian (surfel)
    float cutoff = 3.f;

    // Define the params for projection of surfel to pixel space
    glm::vec3 conic_signature = glm::vec3(cutoff * cutoff, cutoff * cutoff, -1.f);
    float conic_denom = glm::dot(conic_signature, splat2pix_mat[2] * splat2pix_mat[2]);
    glm::vec3 normalized_signature = (1.f / conic_denom) * conic_signature;

    // Compute gradient flowing through centers to splat2pix mat cols
    glm::vec3 grad_splat2pix_mat_col0 = grad_projected_center.x * normalized_signature * splat2pix_mat[2];
    glm::vec3 grad_splat2pix_mat_col1 = grad_projected_center.y * normalized_signature * splat2pix_mat[2];
    glm::vec3 grad_splat2pix_mat_col2 = grad_projected_center.x * normalized_signature * splat2pix_mat[0] +
                                        grad_projected_center.y * normalized_signature * splat2pix_mat[1];

    // Compute gradient flowing through normalized signaturea and conic denom to splat2pix 3rd col
    glm::vec3 grad_normalized_signature = grad_projected_center.x * splat2pix_mat[0] * splat2pix_mat[2] +
                                          grad_projected_center.y * splat2pix_mat[1] * splat2pix_mat[2];
    // (grad_conic_denom) * (grad conic_denom w.r.t. splat2pix_mat_col2)
    grad_splat2pix_mat_col2 += (glm::dot(grad_normalized_signature, normalized_signature) * (-1.f / conic_denom)) *
                               (conic_signature * splat2pix_mat[2] * 2.f);

    // Accumulate into grad_splat2pix_mat
    grad_splat2pix_mat[0] += grad_splat2pix_mat_col0;
    grad_splat2pix_mat[1] += grad_splat2pix_mat_col1;
    grad_splat2pix_mat[2] += grad_splat2pix_mat_col2;
}

// Backpropagate through the preprocessing steps. 1 surfel per thread.
__global__ void preprocess_kernel_bwd(
    const int surfel_count,                        // Surfel / Points count
    const float3 *__restrict__ points_world_space, // All points (surfels) in world space
    const float2 *__restrict__ scale_vecs,         // Scale vectors
    const float glob_scale_mod,                    // Global scale modifier
    const float4 *__restrict__ quats,              // Quaternions
    const float *__restrict__ w2cam_mat,           // World to Cam space matrix
    const float *__restrict__ w2clip_mat,          // World to Clip space matrix
    const int img_W, const int img_H,              // Image width and height
    const uint32_t *__restrict__ asymmetric_radii, // Both surfel radii for tighter bounding boxes
    const float3 *__restrict__ splat2pix_mats,     // Splat to pixel space matrices for each surfel
    const float3 *__restrict__ grad_normal,        // Gradients of 3D normals (cam space)
    float3 *__restrict__ grad_points_world_space,  // Computed gradients of points (world space)
    float2 *__restrict__ grad_scale_vecs,          // Computed gradients of scale vectors
    float4 *__restrict__ grad_quats,               // Computed gradients of quaternions
    float2 *__restrict__ grad_projected_centers,   // Computed gradients of projected centers (pix space)
    float3 *__restrict__ grad_splat2pix_mats       // Computed gradients of splat 2 pixel space matrices
)
{
    auto surfel_idx = cg::this_grid().thread_rank();

    // Sanity check and verify that this surfel wasn't culled in forward.
    if (surfel_idx >= surfel_count || asymmetric_radii[surfel_idx] <= 0)
        return;

    // Load surfel's position in world space, scale vector, quaternion
    float3 point_world_space = points_world_space[surfel_idx];
    float2 scale_vec = scale_vecs[surfel_idx];
    float4 quat = quats[surfel_idx];

    // Compute splat2pix_mat and its intermediates (Compute vs Memory trade-off)
    glm::mat3 splat2pix_mat;
    glm::mat3 rot_mat;
    glm::mat3 scale_mat;
    glm::mat3x4 world2pix_mat;
    float3 normal;

    recompute_splat2pix_intermediates(
        point_world_space,
        {scale_vec.x, scale_vec.y}, glob_scale_mod,
        {quat.x, quat.y, quat.z, quat.w},
        w2cam_mat, w2clip_mat,
        img_W, img_H,
        splat2pix_mat, rot_mat, scale_mat, world2pix_mat,
        normal);

    // Compute gradients through AABB.
    const float3 *g_splat2pix_mat_base = &grad_splat2pix_mats[3 * surfel_idx];
    glm::mat3 grad_splat2pix_mat = glm::mat3(
        g_splat2pix_mat_base[0].x, g_splat2pix_mat_base[0].y, g_splat2pix_mat_base[0].z,
        g_splat2pix_mat_base[1].x, g_splat2pix_mat_base[1].y, g_splat2pix_mat_base[1].z,
        g_splat2pix_mat_base[2].x, g_splat2pix_mat_base[2].y, g_splat2pix_mat_base[2].z);

    float2 &grad_projected_center = grad_projected_centers[surfel_idx];
    if (grad_projected_center.x != 0 || grad_projected_center.y != 0)
    {
        grad_compute_aabb(splat2pix_mat, grad_projected_center, grad_splat2pix_mat);
    }

    // Compute gradients through projection chain

    // 1. gradients from splat2pix matrix to splat2world matrix
    glm::mat3x4 grad_splat2world_mat = world2pix_mat * glm::transpose(grad_splat2pix_mat);

    // 2. gradients from surfel normals (cam space) to 3rd col of local frame matrix
    float3 grad_local_frame_mat_col2 = rotate_vector_vjp(grad_normal[surfel_idx], w2cam_mat);

    // 2.5 Flip gradients of 3rd col of local frame matrix based on FLIP_NORMALS_TO_CAM
#if FLIP_NORMALS_TO_CAM
    float3 point_cam_space = transform_point_world_to_cam(point_world_space, w2cam_mat);
    float cos_theta = -(point_cam_space.x * normal.x +
                        point_cam_space.y * normal.y +
                        point_cam_space.z * normal.z);
    float sign = (cos_theta > .0f) ? 1.f : -1.f;
    grad_local_frame_mat_col2 = sign * grad_local_frame_mat_col2;
#endif

    // 3. gradients to local frame matrix
    glm::mat3 grad_local_frame_mat = glm::mat3(
        glm::vec3(grad_splat2world_mat[0]),
        glm::vec3(grad_splat2world_mat[1]),
        glm::vec3(grad_local_frame_mat_col2.x, grad_local_frame_mat_col2.y, grad_local_frame_mat_col2.z));

    // 4. gradients to surfel world space position, scale vector, quaternions
    grad_points_world_space[surfel_idx] = {grad_splat2world_mat[2].x, grad_splat2world_mat[2].y, grad_splat2world_mat[2].z};

    grad_scale_vecs[surfel_idx] = {glm::dot(grad_local_frame_mat[0], rot_mat[0]),
                                   glm::dot(grad_local_frame_mat[1], rot_mat[1])};

    grad_quats[surfel_idx] = transform_quat_to_rotmat_vjp(
        {quat.x, quat.y, quat.z, quat.w},
        glm::mat3(
            grad_local_frame_mat[0] * glm::vec3(scale_vec.x),
            grad_local_frame_mat[1] * glm::vec3(scale_vec.y),
            grad_local_frame_mat[2]));

    // Update projected centers (pix space) with more stable gradients for densification
    // Use grad_splat2pix_mats before it's updated (pure image rendering loss).
    // splat2pix_mat[2].z is cam space depth.
    grad_projected_center = {
        g_splat2pix_mat_base[0].z * splat2pix_mat[2].z * .5f,
        g_splat2pix_mat_base[1].z * splat2pix_mat[2].z * .5f};
}

// Backpropagate through preprocessing
void BWD::preprocess(
    const int surfel_count,           // Surfel / Points count
    const float3 *points_world_space, // All points (surfels) in world space
    const float2 *scale_vecs,         // Scale vectors
    const float glob_scale_mod,       // Global scale modifier
    const float4 *quats,              // Quaternions
    const float *w2cam_mat,           // World to Cam space matrix
    const float *w2clip_mat,          // World to Clip space matrix
    const int img_W, const int img_H, // Image width and height
    const uint32_t *asymmetric_radii, // Both surfel radii for tighter bounding boxes
    const float3 *splat2pix_mats,     // Splat to pixel space matrices for each surfel
    const float3 *grad_normal,        // Gradients of 3D normals (cam space)
    float3 *grad_points_world_space,  // Computed gradients of points (world space)
    float2 *grad_scale_vecs,          // Computed gradients of scale vectors
    float4 *grad_quats,               // Computed gradients of quaternions
    float2 *grad_projected_centers,   // Computed gradients of projected centers (pix space)
    float3 *grad_splat2pix_mats       // Computed gradients of splat 2 pixel space matrices
)
{
    // Call kernel. 1 surfel per thread.
    preprocess_kernel_bwd<<<DIV_CEIL(surfel_count, BLOCK_SIZE), BLOCK_SIZE>>>(
        surfel_count,
        points_world_space,
        scale_vecs, glob_scale_mod,
        quats, w2cam_mat, w2clip_mat,
        img_W, img_H, asymmetric_radii,
        splat2pix_mats,
        grad_normal, grad_points_world_space,
        grad_scale_vecs, grad_quats, grad_projected_centers,
        grad_splat2pix_mats);
}