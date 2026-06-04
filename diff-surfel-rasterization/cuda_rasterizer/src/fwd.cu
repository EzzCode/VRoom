#include "../includes/fwd.cuh"
#include "../includes/utils.cuh"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Compute Splat to pixel space transformation matrix.
// This matrix maps the local surfel plane to the pixel coordinate space.
__device__ void compute_splat2pix_mat(
    const float3 &point_world_space,  // Surfel coordinate in world space
    const glm::vec2 scale_vec,        // Scale vector (world-space scale)
    const float glob_scale_mod,       // Scale vector global modifier
    const glm::vec4 quat,             // Quaternion (world-space rotation)
    const float *w2cam_mat,           // World to Cam space matrix
    const float *w2clip_mat,          // World to Clip space matrix
    const int img_W, const int img_H, // Image width and height
    glm::mat3 &splat2pix_mat,         // Computed splat to pixel space matrix
    float3 &normal                    // Computed surfel normal in camera space
)
{
    // Creat the local frame of the surfel (rotation * scaling)
    glm::mat3 rot_mat = transform_quat_to_rotmat(quat);
    glm::mat3 scale_mat = create_scale_mat(scale_vec, glob_scale_mod);
    glm::mat3 local_frame_mat = rot_mat * scale_mat;

    // Create the splat to world space matrix. GLM is col-major
    glm::mat3x4 splat2world_mat = glm::mat3x4(
        glm::vec4(local_frame_mat[0], 0.0),
        glm::vec4(local_frame_mat[1], 0.0),
        glm::vec4(point_world_space.x, point_world_space.y, point_world_space.z, 1));

    // This is just GLM-formatted w2clip_mat.
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
    splat2pix_mat = glm::transpose(splat2world_mat) * glm_world2clip_mat * clip2pix_mat;

    // Compute surfel normal in cam space
    normal = rotate_vector({local_frame_mat[2].x, local_frame_mat[2].y, local_frame_mat[2].z}, w2cam_mat);
}

// Compute the Axis-Aligned Bounding Box (AABB) of the projected surfel in pixel space
// and the surfel's pixel-space center.
__device__ bool compute_aabb(
    const glm::mat3 &splat2pix_mat, // Splat to pixel space matrix
    const float cutoff,             // Pixel-space surfel radius cutoff
    float2 &center_pixel,           // Computed Surfel center in pixel-space
    float2 &surfel_radii            // Computed Surfel radii in pixel-space
)
{
    // Define the params for projection of surfel to pixel space
    glm::vec3 conic_signature = glm::vec3(cutoff * cutoff, cutoff * cutoff, -1.f);

    // Compute the pixel-space surfel center
    float conic_denom = glm::dot(conic_signature, splat2pix_mat[2] * splat2pix_mat[2]);
    if (conic_denom == 0) // sanity check
        return false;
    glm::vec3 normalized_signature = (1.f / conic_denom) * conic_signature;
    glm::vec2 center_pixel_glm = glm::vec2(
        glm::dot(normalized_signature, splat2pix_mat[0] * splat2pix_mat[2]),
        glm::dot(normalized_signature, splat2pix_mat[1] * splat2pix_mat[2]));

    // Compute pixel-space surfel radii
    glm::vec2 surfel_radii_glm_0 = center_pixel_glm * center_pixel_glm -
                                   glm::vec2(glm::dot(normalized_signature, splat2pix_mat[0] * splat2pix_mat[0]),
                                             glm::dot(normalized_signature, splat2pix_mat[1] * splat2pix_mat[1]));
    glm::vec2 surfel_radii_glm = glm::sqrt(glm::max(glm::vec2(1e-4f, 1e-4f), surfel_radii_glm_0));

    // Assign outputs
    center_pixel = {center_pixel_glm.x, center_pixel_glm.y};
    surfel_radii = {surfel_radii_glm.x, surfel_radii_glm.y};

    return true;
}

// Preprocessing kernel
__global__ void preprocess_kernel_fwd(
    const int surfel_count,                           // Surfel / Points count
    const float3 *__restrict__ points_world_space,    // All points (surfels) in world space
    const float2 *__restrict__ scale_vecs,            // Scale vectors
    const float glob_scale_mod,                       // Global scale modifier
    const float4 *__restrict__ quats,                 // Quaternions
    const float *__restrict__ opacities,              // surfel opacities
    const float *__restrict__ w2cam_mat,              // World to Cam space matrix
    const float *__restrict__ w2clip_mat,             // World to Clip space matrix
    const int img_W, const int img_H,                 // Image width and height
    const dim3 tile_grid,                             // Grid dimensions for render kernels
    float2 *__restrict__ projected_centers_buff,      // Buffer with mapped pixel locations of each surfel
    uint32_t *__restrict__ asymmetric_radii_buff,     // Both surfel radii for tighter bounding boxes
    float *__restrict__ depths_buff,                  // Computed surfel depths as seen from the image (cam space)
    float3 *__restrict__ splat2pix_mats_buff,         // Splat to pixel space matrices buffer for each surfel
    float4 *__restrict__ normal_opacity_buff,         // Normals (camera space) concatenated with opacity for each surfel
    uint32_t *__restrict__ surfels_tiles_touched_buff // Number of tiles touched by each surfel
)
{
    auto surfel_idx = cg::this_grid().thread_rank();

    // Verify the thread corresponds to a surfel
    if (surfel_idx >= surfel_count)
        return;

    // Frustum cull and cam-space point coords
    float3 point_world_space = points_world_space[surfel_idx];
    float3 point_cam_space;
    if (!verify_in_frustum(point_world_space, w2cam_mat, point_cam_space))
        return;

    // Perform opacity culling, quit if transparent
    if (opacities[surfel_idx] < 0.005f)
        return;

    // Compute splat to pix-space matrix and store it in the buffer for
    // render phase.
    glm::mat3 splat2pix_mat;
    float3 normal;
    compute_splat2pix_mat(point_world_space, {scale_vecs[surfel_idx].x, scale_vecs[surfel_idx].y},
                          glob_scale_mod,
                          {quats[surfel_idx].x, quats[surfel_idx].y, quats[surfel_idx].z, quats[surfel_idx].w},
                          w2cam_mat, w2clip_mat, img_W, img_H, splat2pix_mat, normal);
    splat2pix_mats_buff[surfel_idx * 3 + 0] = {splat2pix_mat[0][0], splat2pix_mat[0][1], splat2pix_mat[0][2]};
    splat2pix_mats_buff[surfel_idx * 3 + 1] = {splat2pix_mat[1][0], splat2pix_mat[1][1], splat2pix_mat[1][2]};
    splat2pix_mats_buff[surfel_idx * 3 + 2] = {splat2pix_mat[2][0], splat2pix_mat[2][1], splat2pix_mat[2][2]};

    // Flip surfel normals to face camera
#if FLIP_NORMALS_TO_CAM
    float cos_theta = -(point_cam_space.x * normal.x +
                        point_cam_space.y * normal.y +
                        point_cam_space.z * normal.z);
    if (cos_theta == 0.f)
        // skip edge-on surfels
        return;
    if (cos_theta < 0.f)
        // flip normal
        normal = {-normal.x, -normal.y, -normal.z};
#endif // IF FLIP_NORMALS_TO_CAM

    // Compute surfel center and pixel-space radii
    float2 center_pixel;
    int2 surfel_radii;
    {
        // Local scope trick for optimizing register usage

        // Define pixel cutoff. 3 standard deviations of the gaussian (surfel)
        // which covers ~ 99.7% of surfel's opacity.
        constexpr float cutoff = 3.f;

        float2 unclamped_surfel_radii;
        if (!compute_aabb(splat2pix_mat, cutoff, center_pixel, unclamped_surfel_radii))
            return;
        // Clamp to prevent OOM from surfels being too close to camera
        surfel_radii.x = min(2048, (int)ceil(max(unclamped_surfel_radii.x, cutoff * FILTER_SIZE)));
        surfel_radii.y = min(2048, (int)ceil(max(unclamped_surfel_radii.y, cutoff * FILTER_SIZE)));
    }

    // Compute tile bounds
    uint2 min_tile_coord, max_tile_coord;
    get_tile_bounds(center_pixel, surfel_radii, min_tile_coord, max_tile_coord, tile_grid, BLOCK_DIM_X, BLOCK_DIM_Y);
    uint32_t surfels_tiles_touched = (max_tile_coord.x - min_tile_coord.x) * (max_tile_coord.y - min_tile_coord.y);
    if (surfels_tiles_touched == 0)
        return;

    // Store output values
    depths_buff[surfel_idx] = point_cam_space.z;
    asymmetric_radii_buff[surfel_idx] = surfel_radii.x | (surfel_radii.y << 16);
    projected_centers_buff[surfel_idx] = center_pixel;
    surfels_tiles_touched_buff[surfel_idx] = surfels_tiles_touched;
    normal_opacity_buff[surfel_idx] = {normal.x, normal.y, normal.z, opacities[surfel_idx]};
}

void FWD::preprocess(
    const int surfel_count,              // Surfel / Points count
    const float3 *points_world_space,    // All points (surfels) in world space
    const float2 *scale_vecs,            // Scale vectors
    const float glob_scale_mod,          // Global scale modifier
    const float4 *quats,                 // Quaternions
    const float *opacities,              // surfel opacities
    const float *w2cam_mat,              // World to Cam space matrix
    const float *w2clip_mat,             // World to Clip space matrix
    const int img_W, const int img_H,    // Image width and height
    const dim3 tile_grid,                // Grid dimensions for render kernels
    float2 *projected_centers_buff,      // Buffer with mapped pixel locations of each surfel
    uint32_t *asymmetric_radii_buff,     // Both surfel radii for tighter bounding boxes
    float *depths_buff,                  // Computed surfel depths as seen from the image (cam space)
    float3 *splat2pix_mats_buff,         // Splat to pixel space matrices buffer for each surfel
    float4 *normal_opacity_buff,         // Normals (camera space) concatenated with opacity for each surfel
    uint32_t *surfels_tiles_touched_buff // Number of tiles touched by each surfel
)
{
    // 1D grid, BLOCK_SIZE threads per block, one thread per surfel
    preprocess_kernel_fwd<<<DIV_CEIL(surfel_count, BLOCK_SIZE), BLOCK_SIZE>>>(
        surfel_count,
        points_world_space,
        scale_vecs, glob_scale_mod,
        quats,
        opacities,
        w2cam_mat,
        w2clip_mat,
        img_W, img_H,
        tile_grid,
        projected_centers_buff,
        asymmetric_radii_buff,
        depths_buff,
        splat2pix_mats_buff,
        normal_opacity_buff,
        surfels_tiles_touched_buff);
}

// Rendering kernel (RGB + features).
// One thread for one pixel with collaborative VRAM loading per tile and each
// tile maps to a block. Each thread takes the surfels (front to back) that
// affect its pixel and renders the pixel's (RGB + features).
template <uint8_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_SIZE)
    render_kernel_fwd(
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
        const uint2 *__restrict__ tile_ranges,              // Per-tile surfel index ranges
        uint32_t *__restrict__ contrib_state_buff,          // Indices of the last and median surfel contributing to a pixel
        float *__restrict__ transmittance_and_moments_buff, // Transmittance, first and seconds moments of depth (for distortion loss)
        // Remaining outputs
        float *__restrict__ rendered_color_feat_buff, // Rendered concat of colors + features per pixel
        float *__restrict__ rendered_aux_buff         // Rendered depth, normal, distortion auxiliary channels per pixel
    )
{
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

    // Invalid / done threads can help with VRAM fetching but dont render pixels
    bool render_done = !valid_pixel;

    // Load sorted surfel ranges for this tile
    const uint2 tile_range = tile_ranges[thread_block_ids.group_index().y * DIV_CEIL(img_W, BLOCK_DIM_X) +
                                         thread_block_ids.group_index().x];

    // Find how many rounds Required to load all surfel batches to finish rendering
    const int rounds = DIV_CEIL(tile_range.y - tile_range.x, BLOCK_SIZE); // We load BLOCK_SIZE every round
    int rem_surfels = tile_range.y - tile_range.x;                        // track remaining surfels to render

    // Allocate the shared (among block's threads) memory buffers for batching VRAM access.
    // Batch size is BLOCK_SIZE so each thread is responsible for fetching 1 data item.
    __shared__ int surfel_idx_batch[BLOCK_SIZE];
    __shared__ float2 projected_centers_batch[BLOCK_SIZE];
    __shared__ float4 normal_opacity_batch[BLOCK_SIZE];
    __shared__ float3 splat2pix_mat_col0_batch[BLOCK_SIZE];
    __shared__ float3 splat2pix_mat_col1_batch[BLOCK_SIZE];
    __shared__ float3 splat2pix_mat_col2_batch[BLOCK_SIZE];

    // Front to back blending. Track the ray state as it passes through surfels.
    float transmittance = 1.f;            // Remaining light that passes
    float color_feat_acc[CHANNELS] = {0}; // Accumulated color/feature values

    // Track how many surfel processed before this pixel becomes opaque
    uint32_t contributor_idx = 0;      // Running counter (1-based)
    uint32_t last_contributor_idx = 0; // Last surfel blended (1-based)

#if RENDER_AUX
    // Auxiliary channel accumulators
    float3 normal_acc = {0};
    float depth_acc = 0;
    // Distortion loss moments
    float m1_acc = 0;
    float m2_acc = 0;
    float distortion_acc = 0; // Accumulated distortion loss
    // Median depth tracking. Track depth and surfel id when transmittance drops under .5
    float median_depth = 0;
    int median_contrib_idx = -1;
#endif

    // Warp-level optimization for warp-level early exit.
    // #warps = BLOCK_SIZE / 32
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(thread_block_ids);

    // Loop over loaded VRAM batches until the tile (block) is done
    for (int round = 0; round < rounds; round++, rem_surfels -= BLOCK_SIZE)
    {
        // Block-level early exit.
        // Also, implicitly syncing block (wait for all warps to finish previous round)
        if (__syncthreads_count(render_done) == BLOCK_SIZE)
            break;

        // Collaborate on fetching data from VRAM
        int progress = round * BLOCK_SIZE + thread_block_ids.thread_rank();
        if (tile_range.x + progress < tile_range.y)
        {
            const int glob_surf_idx = sorted_surfel_indices[tile_range.x + progress];
            surfel_idx_batch[thread_block_ids.thread_rank()] = glob_surf_idx;
            projected_centers_batch[thread_block_ids.thread_rank()] = projected_centers[glob_surf_idx];
            normal_opacity_batch[thread_block_ids.thread_rank()] = normal_opacity[glob_surf_idx];
            // splat2pix matrix is col-major
            splat2pix_mat_col0_batch[thread_block_ids.thread_rank()] = splat2pix_mats[3 * glob_surf_idx + 0];
            splat2pix_mat_col1_batch[thread_block_ids.thread_rank()] = splat2pix_mats[3 * glob_surf_idx + 1];
            splat2pix_mat_col2_batch[thread_block_ids.thread_rank()] = splat2pix_mats[3 * glob_surf_idx + 2];
        }
        // Sync threads in tile (block) before accessing the fetched data
        thread_block_ids.sync();

        // Loop over fetched batch of surfels
        for (int batch_surf_idx = 0; batch_surf_idx < min(BLOCK_SIZE, rem_surfels); batch_surf_idx++)
        {
            // Block-level early exit check every 32 surfels
            if (batch_surf_idx % 32 == 0)
                if (__syncthreads_count(render_done) == BLOCK_SIZE)
                    break;

            // Warp-level early exit
            if (warp.all(render_done))
                continue;

            // Thread-level early exit
            if (render_done)
                continue;

            // Track position in surfel's range
            contributor_idx++;

            // Compute ray-surfel intersection and its distance to surfel's center

            // 1. Load surfel center pixel and the splat2pix matrix
            const float2 center_pixel = projected_centers_batch[batch_surf_idx];
            const float3 splat2pix_mat_col0 = splat2pix_mat_col0_batch[batch_surf_idx];
            const float3 splat2pix_mat_col1 = splat2pix_mat_col1_batch[batch_surf_idx];
            const float3 splat2pix_mat_col2 = splat2pix_mat_col2_batch[batch_surf_idx];

            // 2. Define two st. lines representing the vertical and horizontal
            // lines passing through the exact center of this pixel, transformed into
            // the surfel's (u, v) space.
            const float3 line_x = pixel.x * splat2pix_mat_col2 - splat2pix_mat_col0;
            const float3 line_y = pixel.y * splat2pix_mat_col2 - splat2pix_mat_col1;

            // 3. The cross product of these two st. lines gives the ray passing through the pixel
            const float3 ray_cross = cross_product(line_x, line_y);
            if (ray_cross.z == 0.0f) // Skip edge-on surfels
                continue;

            // 4. Perspective division to find the exact (u, v) ray-surfel isect
            // on the surfel's local plane.
            const float2 local_uv = {ray_cross.x / ray_cross.z, ray_cross.y / ray_cross.z};

            // 5. Calculate squared distance from the surfel's center (0,0 in surfel's space)
            const float dist_3d_sq = (local_uv.x * local_uv.x) + (local_uv.y * local_uv.y);

            // 6. Calculate 2D (pixel distances) low-pass filter distance to prevent sub-pixel instability.
            // We dilate the pixel in order to prevent aliasing.
            const float2 pixel_offset = {center_pixel.x - (float)pixel.x, center_pixel.y - (float)pixel.y};
            const float dist_2d_sq = FILTER_INV_SQ * (pixel_offset.x * pixel_offset.x + pixel_offset.y * pixel_offset.y);

            // 7. Use the minimum of true 3D distance and 2D filtered distance
            const float gaussian_dist_sq = min(dist_3d_sq, dist_2d_sq);

            // Compute surfel depth
            float depth = local_uv.x * splat2pix_mat_col2.x + local_uv.y * splat2pix_mat_col2.y + splat2pix_mat_col2.z;
            if (depth < NEAR_PLANE)
                continue; // skip surfels that are too close to camera

            // Fetch normals, opacity
            float4 norm_opa = normal_opacity_batch[batch_surf_idx];
            float3 normal = {norm_opa.x, norm_opa.y, norm_opa.z};
            float opacity = norm_opa.w;

            // Compute alpha (effective opacity at the calculated center-ray_isect distance)
            float power = -0.5f * gaussian_dist_sq;
            if (power > 0.f)
                continue; // Opacity can only decrease as center-ray_isect dist increases

            float alpha = min(.99f, opacity * exp(power));
            if (alpha < 1.f / 255.f)
                continue; // Effectively transparent surfel

            // Verify opacity of pixel after blending current surfel
            float next_transmittance = transmittance * (1 - alpha);
            if (next_transmittance < .0001f)
            {
                // Pixel has become opaque
                render_done = true;
                continue;
            }

            float blending_weight = alpha * transmittance; // blending weight for surfel

#if RENDER_AUX
            // Compute aux outputs
            // 1. Compute distortion loss and depth moments
            float alpha_acc = 1.f - transmittance;
            float normalized_depth = DEPTH_NORM_SCALE * (1 - NEAR_PLANE / depth);
            // 1.1. instead of computing distortion loss between each pair of surfels in O(N^2),
            // we track first and seconds moments of depth
            distortion_acc += blending_weight * (normalized_depth * normalized_depth * alpha_acc +
                                                 m2_acc - 2 * normalized_depth * m1_acc);
            m1_acc += normalized_depth * blending_weight;
            m2_acc += normalized_depth * normalized_depth * blending_weight;

            // 2. Compute effective depth of pixel.
            depth_acc += depth * blending_weight;

            // 3. Median depth tracking (last surfel where T > 0.5)
            if (transmittance > 0.5f)
            {
                median_depth = depth;
                median_contrib_idx = contributor_idx;
            }

            // 4. Compute rendered normals
            normal_acc.x += normal.x * blending_weight;
            normal_acc.y += normal.y * blending_weight;
            normal_acc.z += normal.z * blending_weight;
#endif
            // Finally, compute rendered colors and features
            for (int ch = 0; ch < CHANNELS; ch++)
                // Fused Multiply-Add (FMA). Could be replaced with -O3 compiler option.
                // ldg routes VRAM access through read-only data (texture) cache which is better for broadcasting to other threads
                color_feat_acc[ch] = __fmaf_rn(__ldg(&colors_feat[surfel_idx_batch[batch_surf_idx] * CHANNELS + ch]), blending_weight, color_feat_acc[ch]);

            // Update transmittance
            transmittance = next_transmittance;

            // Track last contributing surfel
            last_contributor_idx = contributor_idx;
        }
    }

    // Store the computed values in the output buffers for valid threads (pixels)
    if (valid_pixel)
    {
        // Compute flattened pixel id
        int pixel_idx = img_W * pixel.y + pixel.x;

        // blend the background with the pixel and store the computed colors+feat
        for (int ch = 0; ch < CHANNELS; ch++)
            // Fused Multiply-Add (FMA) with read-only cache broadcast
            rendered_color_feat_buff[ch * img_H * img_W + pixel_idx] = __fmaf_rn(transmittance, __ldg(&background[ch]), color_feat_acc[ch]);

        // Store last contributor
        contrib_state_buff[pixel_idx] = last_contributor_idx;

        // Store final transmittance
        transmittance_and_moments_buff[pixel_idx] = transmittance;
#if RENDER_AUX
        // Store the aux outputs
        // 1. first & second moments (saved for backward distortion grad)
        transmittance_and_moments_buff[pixel_idx + img_H * img_W] = m1_acc;
        transmittance_and_moments_buff[pixel_idx + 2 * img_H * img_W] = m2_acc;
        // 2. Median contributor
        contrib_state_buff[pixel_idx + img_H * img_W] = median_contrib_idx;
        // 3. Depths
        rendered_aux_buff[pixel_idx + img_H * img_W * DEPTH_OFFSET] = depth_acc;
        // 4. Accumulated Alphas (accumulated opacity)
        rendered_aux_buff[pixel_idx + img_H * img_W * ALPHA_OFFSET] = 1.f - transmittance;
        // 5. Normals
        rendered_aux_buff[pixel_idx + img_H * img_W * (NORMAL_OFFSET + 0)] = normal_acc.x;
        rendered_aux_buff[pixel_idx + img_H * img_W * (NORMAL_OFFSET + 1)] = normal_acc.y;
        rendered_aux_buff[pixel_idx + img_H * img_W * (NORMAL_OFFSET + 2)] = normal_acc.z;
        // 6. Median depths
        rendered_aux_buff[pixel_idx + img_H * img_W * MEDIAN_DEPTH_OFFSET] = median_depth;
        // 7. Distortions
        rendered_aux_buff[pixel_idx + img_H * img_W * DISTORTION_OFFSET] = distortion_acc;
#endif
    }
}

// Rendering image (RGB + features) using surfels
void FWD::render(
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
    const uint2 *tile_ranges,              // Per-tile [start, end) index ranges
    uint32_t *contrib_state_buff,          // Indices of the last and median surfel contributing to a pixel
    float *transmittance_and_moments_buff, // Transmittance, first and seconds moments of depth (for distortion loss)
    // Remaining outputs
    float *rendered_color_feat_buff, // Rendered concat of colors + features per pixel
    float *rendered_aux_buff         // Rendered depth, normal, distortion auxiliary channels per pixel
)
{
    // Define kernel structure
    dim3 tile_grid(DIV_CEIL(img_W, BLOCK_DIM_X), DIV_CEIL(img_H, BLOCK_DIM_Y), 1);
    dim3 tile(BLOCK_DIM_X, BLOCK_DIM_Y, 1);

#define __RENDER_CALL_(CHANNELS)                          \
    case CHANNELS:                                        \
        render_kernel_fwd<CHANNELS><<<tile_grid, tile>>>( \
            img_W, img_H,                                 \
            colors_feat,                                  \
            background,                                   \
            projected_centers,                            \
            splat2pix_mats,                               \
            normal_opacity,                               \
            sorted_surfel_indices,                        \
            tile_ranges,                                  \
            contrib_state_buff,                           \
            transmittance_and_moments_buff,               \
            rendered_color_feat_buff,                     \
            rendered_aux_buff);                           \
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

// Frustum culling kernel
__global__ void frustum_cull_kernel(
    const int surfel_count,                        // Surfel / Points count
    const float3 *__restrict__ points_world_space, // All points (surfels) in world space
    const float2 *__restrict__ scale_vecs,         // Scale vectors
    const float glob_scale_mod,                    // Global scale modifier
    const float4 *__restrict__ quats,              // Quaternions
    const float *__restrict__ w2cam_mat,           // World to Cam space matrix
    const float *__restrict__ w2clip_mat,          // World to Clip space matrix
    const int img_W, const int img_H,              // Image width and height
    int *__restrict__ radii_buff                   // Pixel-space surfel max radius. Used as the culling metric (zero for culled surfels)
)
{
    auto surfel_idx = cg::this_grid().thread_rank();

    // Verify the thread corresponds to a surfel
    if (surfel_idx >= surfel_count)
        return;

    // Frustum cull and cam-space point coords
    float3 point_world_space = points_world_space[surfel_idx];
    float3 point_cam_space;
    if (!verify_in_frustum(point_world_space, w2cam_mat, point_cam_space))
        return;

    // Compute splat to pix-space matrix
    glm::mat3 splat2pix_mat;
    float3 unused_normal; // Unused so the compiler will optimize it out
    compute_splat2pix_mat(point_world_space, {scale_vecs[surfel_idx].x, scale_vecs[surfel_idx].y},
                          glob_scale_mod,
                          {quats[surfel_idx].x, quats[surfel_idx].y, quats[surfel_idx].z, quats[surfel_idx].w},
                          w2cam_mat, w2clip_mat, img_W, img_H, splat2pix_mat, unused_normal);

    // Compute surfel center and pixel-space radii
    float2 center_pixel;
    int2 surfel_radii;
    {
        // Local scope trick for optimizing register usage

        // Define pixel cutoff. 3 standard deviations of the gaussian (surfel)
        // which covers ~ 99.7% of surfel's opacity.
        constexpr float cutoff = 3.f;

        float2 unclamped_surfel_radii;
        if (!compute_aabb(splat2pix_mat, cutoff, center_pixel, unclamped_surfel_radii))
            return;
        // Clamp to prevent OOM from surfels being too close to camera
        surfel_radii.x = min(2048, (int)ceil(max(unclamped_surfel_radii.x, cutoff * FILTER_SIZE)));
        surfel_radii.y = min(2048, (int)ceil(max(unclamped_surfel_radii.y, cutoff * FILTER_SIZE)));
    }

    // Cull surfels whose projection is outside the image boundaries
    float2 min_pix_coord = {center_pixel.x - surfel_radii.x, center_pixel.y - surfel_radii.y};
    float2 max_pix_coord = {center_pixel.x + surfel_radii.x, center_pixel.y + surfel_radii.y};
    if (max_pix_coord.x < 0.f || min_pix_coord.x >= img_W || max_pix_coord.y < 0.f || min_pix_coord.y >= img_H)
        return;

    // Store max radius value (only visible surfels reach here)
    radii_buff[surfel_idx] = max(surfel_radii.x, surfel_radii.y);
}

// Frustum culling. Cull surfels that aren't in the current camera's view.
void FWD::frustum_cull(
    const int surfel_count,           // Surfel / Points count
    const float3 *points_world_space, // All points (surfels) in world space
    const float2 *scale_vecs,         // Scale vectors
    const float glob_scale_mod,       // Global scale modifier
    const float4 *quats,              // Quaternions
    const float *w2cam_mat,           // World to Cam space matrix
    const float *w2clip_mat,          // World to Clip space matrix
    const int img_W, const int img_H, // Image width and height
    int *radii_buff                   // Pixel-space surfel max radius. Used as the culling metric (zero for culled surfels)
)
{
    // 1D grid, BLOCK_SIZE threads per block, one thread per surfel
    frustum_cull_kernel<<<DIV_CEIL(surfel_count, BLOCK_SIZE), BLOCK_SIZE>>>(
        surfel_count,
        points_world_space,
        scale_vecs, glob_scale_mod,
        quats,
        w2cam_mat, w2clip_mat,
        img_W, img_H,
        radii_buff);
}