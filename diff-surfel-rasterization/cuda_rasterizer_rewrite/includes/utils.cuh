#pragma once
#include "configs.cuh"
#include <cuda.h>

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#define CUDA_SAFE_CALL(kernel_launch, debug)                                           \
    kernel_launch;                                                                     \
    if (debug)                                                                         \
    {                                                                                  \
        auto retStat = cudaDeviceSynchronize();                                        \
        if (retStat != cudaSuccess)                                                    \
        {                                                                              \
            std::cerr << "\n[CUDA KERNEL ERROR] in " << __FILE__;                      \
            std::cerr << "\nLine " << __LINE__ << ": " << cudaGetErrorString(retStat); \
            throw std::runtime_error(cudaGetErrorString(retStat));                     \
        }                                                                              \
    }

// Macro for division ceil trick.
#define DIV_CEIL(A, B) (((A) + (B) - 1) / (B))

// CUDA Vector Math

// Scalar multiplication
__forceinline__ __device__ float3 operator*(float f, const float3 &a)
{
    return {f * a.x, f * a.y, f * a.z};
}
// Cross product of two 3D vectors
__forceinline__ __device__ float3 cross_product(const float3 &a, const float3 &b)
{
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x};
}
// Element-wise subtraction
__forceinline__ __device__ float3 operator-(const float3 &a, const float3 &b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}
// Element-wise addition
__forceinline__ __device__ float3 operator+(const float3 &a, const float3 &b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}
// Element-wise multiplication
__forceinline__ __device__ float3 operator*(const float3 &a, const float3 &b)
{
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}

// Maps NDC (Normalized Device Coordinate) to pixel coordinates in either x or y dim.
// Returns float for extra precision in both rendering and differentiation.
__forceinline__ __device__ float transform_ndc_to_pix(float ndc_val, int image_dim)
{
    return ((ndc_val + 1.0f) * image_dim - 1.0f) * 0.5f;
}

// Bounding box for surfels. Finds which tiles (blocks) the surfel contributes to.
__forceinline__ __device__ void get_tile_bounds(const float2 center_pixel, const int surfel_radius,
                                                uint2 &min_tile_coord,
                                                uint2 &max_tile_coord, const dim3 tile_grid,
                                                const int block_x, const int block_y)
{
    min_tile_coord = {
        (unsigned int)max(0, min((int)tile_grid.x, (int)((center_pixel.x - surfel_radius) / block_x))),
        (unsigned int)max(0, min((int)tile_grid.y, (int)((center_pixel.y - surfel_radius) / block_y)))};

    max_tile_coord = {
        (unsigned int)max(0, min((int)tile_grid.x, (int)DIV_CEIL(center_pixel.x + surfel_radius, block_x))),
        (unsigned int)max(0, min((int)tile_grid.y, (int)DIV_CEIL(center_pixel.y + surfel_radius, block_y)))};
}

// Asymmetric bounding box: separate x/y radii for elongated surfels.
// Finds which tiles (blocks) the surfel contributes to.
__forceinline__ __device__ void get_tile_bounds(const float2 center_pixel, const int2 surfel_radii,
                                                uint2 &min_tile_coord,
                                                uint2 &max_tile_coord, const dim3 tile_grid,
                                                const int block_x, const int block_y)
{
    min_tile_coord = {
        (unsigned int)max(0, min((int)tile_grid.x, (int)((center_pixel.x - surfel_radii.x) / block_x))),
        (unsigned int)max(0, min((int)tile_grid.y, (int)((center_pixel.y - surfel_radii.y) / block_y)))};

    max_tile_coord = {
        (unsigned int)max(0, min((int)tile_grid.x, (int)DIV_CEIL(center_pixel.x + surfel_radii.x, block_x))),
        (unsigned int)max(0, min((int)tile_grid.y, (int)DIV_CEIL(center_pixel.y + surfel_radii.y, block_y)))};
}

// Transform point from world coordinates to camera coordinates.
// w2cam_mat is 4x4 column major as per OpenGL convention and flattened as float*.
// Calculates (w2cam_mat · [point; 1]) and drops last row (w)
__forceinline__ __device__ float3 transform_point_world_to_cam(const float3 point,
                                                               const float *w2cam_mat)
{
    // matrix[i][j] -> col. major flattened_matrix[j * #rows + i].
    return {
        w2cam_mat[0] * point.x + w2cam_mat[4] * point.y + w2cam_mat[8] * point.z + w2cam_mat[12],
        w2cam_mat[1] * point.x + w2cam_mat[5] * point.y + w2cam_mat[9] * point.z + w2cam_mat[13],
        w2cam_mat[2] * point.x + w2cam_mat[6] * point.y + w2cam_mat[10] * point.z + w2cam_mat[14],
    };
}

// Transform point from world coordinates to clip coordinates.
// w2clip_mat is 4x4 column major as per OpenGL convention and flattened as float*.
// Calculates (w2clip_mat · [point; 1]) and keeps last row (w) for perspective division
__forceinline__ __device__ float4 transform_point_world_to_clip(const float3 point,
                                                                const float *w2clip_mat)
{
    // matrix[i][j] -> col. major flattened_matrix[j * #rows + i].
    return {
        w2clip_mat[0] * point.x + w2clip_mat[4] * point.y + w2clip_mat[8] * point.z + w2clip_mat[12],
        w2clip_mat[1] * point.x + w2clip_mat[5] * point.y + w2clip_mat[9] * point.z + w2clip_mat[13],
        w2clip_mat[2] * point.x + w2clip_mat[6] * point.y + w2clip_mat[10] * point.z + w2clip_mat[14],
        w2clip_mat[3] * point.x + w2clip_mat[7] * point.y + w2clip_mat[11] * point.z + w2clip_mat[15],
    };
}

// Rotate vector without translation. The vector is the surfel's normal vector which
// we rotate into camera space to compute the viewing angle.
__forceinline__ __device__ float3 rotate_vector(const float3 vec,
                                                const float *rot_mat)
{
    // matrix[i][j] -> col. major flattened_matrix[j * #rows + i].
    return {
        rot_mat[0] * vec.x + rot_mat[4] * vec.y + rot_mat[8] * vec.z,
        rot_mat[1] * vec.x + rot_mat[5] * vec.y + rot_mat[9] * vec.z,
        rot_mat[2] * vec.x + rot_mat[6] * vec.y + rot_mat[10] * vec.z,
    };
}

// Rotate vector with the transpose of the rotation matrix for gradient backprop.
// Chain rule requires the transpose of the rotation matrix (which is the Jacobian
// of the rotation matrix) so this is a vector-Jacobian product (VJP).
__forceinline__ __device__ float3 rotate_vector_vjp(const float3 vec, const float *rot_mat)
{
    // To index the transpose of the matrix we use row-major indexing while skipping the
    // last column.
    // matrix[i][j] -> row. major flattened_matrix[i * #cols + j].
    return {
        rot_mat[0] * vec.x + rot_mat[1] * vec.y + rot_mat[2] * vec.z,
        rot_mat[4] * vec.x + rot_mat[5] * vec.y + rot_mat[6] * vec.z,
        rot_mat[8] * vec.x + rot_mat[9] * vec.y + rot_mat[10] * vec.z,
    };
}

// Computes the gradient of normalized 3D vector.
// Forward:  norm = vec / ||vec||. Backward: dLoss_dvec = J^T * dLoss_dnorm
__forceinline__ __device__ float3 grad_norm_vec_vjp(const float3 vec, const float3 grad_norm)
{
    float norm_sq = vec.x * vec.x + vec.y * vec.y + vec.z * vec.z;

    // Safety guard: prevent division by zero
    if (norm_sq < 1e-12f)
        return {0.0f, 0.0f, 0.0f};

    float inv_norm_cube = 1.0f / sqrt(norm_sq * norm_sq * norm_sq); // OPTIMIZATION LATER: USE rsqrtf

    // VJP
    return {
        ((norm_sq - vec.x * vec.x) * grad_norm.x - vec.x * vec.y * grad_norm.y - vec.x * vec.z * grad_norm.z) * inv_norm_cube,
        (-vec.y * vec.x * grad_norm.x + (norm_sq - vec.y * vec.y) * grad_norm.y - vec.y * vec.z * grad_norm.z) * inv_norm_cube,
        (-vec.z * vec.x * grad_norm.x - vec.z * vec.y * grad_norm.y + (norm_sq - vec.z * vec.z) * grad_norm.z) * inv_norm_cube,
    };
}

// Computes the gradient of normalized 4D vector.
// Forward:  norm = vec / ||vec||. Backward: dLoss_dvec = J^T * grad_norm
__forceinline__ __device__ float4 grad_norm_vec_vjp(const float4 vec, const float4 grad_norm)
{
    float norm_sq = vec.x * vec.x + vec.y * vec.y + vec.z * vec.z + vec.w * vec.w;

    // Safety guard: prevent division by zero
    if (norm_sq < 1e-12f)
        return {0.0f, 0.0f, 0.0f, 0.0f};

    float inv_norm_cube = 1.0f / sqrt(norm_sq * norm_sq * norm_sq); // OPTIMIZATION LATER: USE rsqrtf

    float4 vec_grad_norm = {vec.x * grad_norm.x, vec.y * grad_norm.y,
                            vec.z * grad_norm.z, vec.w * grad_norm.w};
    float vec_dot_grad_norm = vec_grad_norm.x + vec_grad_norm.y +
                              vec_grad_norm.z + vec_grad_norm.w;

    // VJP
    return {
        ((norm_sq - vec.x * vec.x) * grad_norm.x - vec.x * (vec_dot_grad_norm - vec_grad_norm.x)) * inv_norm_cube,
        ((norm_sq - vec.y * vec.y) * grad_norm.y - vec.y * (vec_dot_grad_norm - vec_grad_norm.y)) * inv_norm_cube,
        ((norm_sq - vec.z * vec.z) * grad_norm.z - vec.z * (vec_dot_grad_norm - vec_grad_norm.z)) * inv_norm_cube,
        ((norm_sq - vec.w * vec.w) * grad_norm.w - vec.w * (vec_dot_grad_norm - vec_grad_norm.w)) * inv_norm_cube,
    };
}

// Computes the gradient of normalized 3D vector z-component only.
// Forward:  norm = vec / ||vec||. Backward: dLoss_dvec = J^T * grad_norm
__forceinline__ __device__ float grad_norm_vecz_vjp(const float3 vec, const float3 grad_norm)
{
    float norm_sq = vec.x * vec.x + vec.y * vec.y + vec.z * vec.z;

    // Safety guard: prevent division by zero
    if (norm_sq < 1e-12f)
        return 0.0f;

    float inv_norm_cube = 1.0f / sqrt(norm_sq * norm_sq * norm_sq); // OPTIMIZATION LATER: USE rsqrtf

    // VJP
    return (-vec.z * vec.x * grad_norm.x - vec.z * vec.y * grad_norm.y +
            (norm_sq - vec.z * vec.z) * grad_norm.z) *
           inv_norm_cube;
}

// Transform quaternion into a rotation matrix
__forceinline__ __device__ glm::mat3 transform_quat_to_rotmat(const glm::vec4 quat)
{
    float inv_norm = rsqrtf(quat.x * quat.x + quat.y * quat.y + quat.z * quat.z + quat.w * quat.w);

    // GLM's convention is to store .w in .x, .x in .y, and so on.
    // We fix it before the calculations as well as apply
    // normalization to quat vector
    float w = quat.x * inv_norm;
    float x = quat.y * inv_norm;
    float y = quat.z * inv_norm;
    float z = quat.w * inv_norm;

    // GLM matrices are col-major
    return {
        1.f - 2.f * (y * y + z * z), // m[0][0]
        2.f * (x * y + w * z),       // m[1][0]
        2.f * (x * z - w * y),       // m[2][0]
        2.f * (x * y - w * z),       // m[0][1]
        1.f - 2.f * (x * x + z * z), // m[1][1]
        2.f * (y * z + w * x),       // m[2][1]
        2.f * (x * z + w * y),       // m[0][2]
        2.f * (y * z - w * x),       // m[1][2]
        1.f - 2.f * (x * x + y * y), // m[2][2]
    };
}

// Gradient backprop for transform quaternion into a rotation matrix
__forceinline__ __device__ float4 transform_quat_to_rotmat_vjp(const glm::vec4 quat,
                                                               const glm::mat3 grad_rotmat)
{
    float inv_norm = rsqrtf(quat.x * quat.x + quat.y * quat.y + quat.z * quat.z + quat.w * quat.w);

    // GLM's convention is to store .w in .x, .x in .y, and so on.
    // We fix it before the calculations as well as apply
    // normalization to quat vector
    float w = quat.x * inv_norm;
    float x = quat.y * inv_norm;
    float y = quat.z * inv_norm;
    float z = quat.w * inv_norm;

    float4 quat_vjp;

    // GLM matrix grad_rotmat is col-major
    // We will use the same GLM convention of shifting fields.
    // .w is stored in .x
    quat_vjp.x = 2.f *
                 (x * (grad_rotmat[1][2] - grad_rotmat[2][1]) +
                  y * (grad_rotmat[2][0] - grad_rotmat[0][2]) +
                  z * (grad_rotmat[0][1] - grad_rotmat[1][0]));

    // .x is stored in .y
    quat_vjp.y = 2.f *
                 (-2.f * x * (grad_rotmat[1][1] + grad_rotmat[2][2]) +
                  y * (grad_rotmat[0][1] + grad_rotmat[1][0]) +
                  z * (grad_rotmat[0][2] + grad_rotmat[2][0]) +
                  w * (grad_rotmat[1][2] - grad_rotmat[2][1]));

    // .y is stored in .z
    quat_vjp.z = 2.f *
                 (x * (grad_rotmat[0][1] + grad_rotmat[1][0]) -
                  2.f * y * (grad_rotmat[0][0] + grad_rotmat[2][2]) +
                  z * (grad_rotmat[1][2] + grad_rotmat[2][1]) +
                  w * (grad_rotmat[2][0] - grad_rotmat[0][2]));

    // .z is stored in .w
    quat_vjp.w = 2.f *
                 (x * (grad_rotmat[0][2] + grad_rotmat[2][0]) +
                  y * (grad_rotmat[1][2] + grad_rotmat[2][1]) -
                  2.f * z * (grad_rotmat[0][0] + grad_rotmat[1][1]) +
                  w * (grad_rotmat[0][1] - grad_rotmat[1][0]));

    return quat_vjp;
}

// Create scale matrix from scale vector. The scale vector represents
// local radius of the surfel ellipse in both x (u) and y (v) directions.
// We use the global scale modifier to modify the scales of surfels without altering
// the learned scale parameters
__forceinline__ __device__ glm::mat3 create_scale_mat(const glm::vec2 scale_vec,
                                                      const float glob_scale_mod)
{
    glm::mat3 scale_mat = glm::mat3(1.f);
    scale_mat[0][0] = glob_scale_mod * scale_vec.x;
    scale_mat[1][1] = glob_scale_mod * scale_vec.y;

    return scale_mat;
}

// Verify if the surfel is in the camera frustum, returns true if in frustum.
// Additionally, store the point's cam space coord.
__forceinline__ __device__ bool verify_in_frustum(const float3 points_world_space,
                                                  const float *w2cam_mat, float3 &point_cam_space)
{
    point_cam_space = transform_point_world_to_cam(points_world_space, w2cam_mat);

    if (point_cam_space.z <= NEAR_PLANE)
        return false;

    return true;
}