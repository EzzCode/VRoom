import cv2
import numpy as np


COLOR_BGR2GRAY = 6
COLOR_BGR2HSV = 40
TERM_CRITERIA_EPS = 2
TERM_CRITERIA_COUNT = 1
RANSAC = 8
HISTCMP_BHATTACHARYYA = 3
NORM_MINMAX = 32
FONT_HERSHEY_SIMPLEX = 0

imread = cv2.imread
imwrite = cv2.imwrite
putText = cv2.putText
bitwise_not = cv2.bitwise_not

class KalmanFilter:
    def __init__(self, dynam_params, measure_params, control_params=0):
        self.dynam_params = dynam_params
        self.measure_params = measure_params
        self.control_params = control_params
        
        self.transitionMatrix = np.eye(dynam_params, dtype=np.float32)
        self.measurementMatrix = np.zeros((measure_params, dynam_params), dtype=np.float32)
        self.processNoiseCov = np.eye(dynam_params, dtype=np.float32)
        self.measurementNoiseCov = np.eye(measure_params, dtype=np.float32)
        self.errorCovPost = np.eye(dynam_params, dtype=np.float32)
        
        self.statePost = np.zeros((dynam_params, 1), dtype=np.float32)
        self.statePre = np.zeros((dynam_params, 1), dtype=np.float32)
        self.errorCovPre = np.eye(dynam_params, dtype=np.float32)

    def predict(self):
        self.statePre = self.transitionMatrix @ self.statePost
        self.errorCovPre = (self.transitionMatrix @ self.errorCovPost @ self.transitionMatrix.T) + self.processNoiseCov
        self.statePost = self.statePre.copy()
        self.errorCovPost = self.errorCovPre.copy()
        return self.statePre

    def correct(self, measurement):
        residual = measurement - (self.measurementMatrix @ self.statePre)
        s_val = (self.measurementMatrix @ self.errorCovPre @ self.measurementMatrix.T) + self.measurementNoiseCov
        k_val = self.errorCovPre @ self.measurementMatrix.T @ np.linalg.inv(s_val)
        self.statePost = self.statePre + (k_val @ residual)
        identity = np.eye(self.dynam_params, dtype=np.float32)
        self.errorCovPost = (identity - (k_val @ self.measurementMatrix)) @ self.errorCovPre
        return self.statePost

def cvtColor(image, code):
    "convert color space"
    if image is None:
        raise ValueError("Input image is None")

    if code == COLOR_BGR2GRAY:
        blue = image[:, :, 0].astype(np.float32)
        green = image[:, :, 1].astype(np.float32)
        red = image[:, :, 2].astype(np.float32)
        return (0.299 * red + 0.587 * green + 0.114 * blue).astype(image.dtype)

    if code == COLOR_BGR2HSV:
        blue = image[:, :, 0].astype(np.float32) / 255.0
        green = image[:, :, 1].astype(np.float32) / 255.0
        red = image[:, :, 2].astype(np.float32) / 255.0

        value = np.maximum(np.maximum(red, green), blue)
        min_value = np.minimum(np.minimum(red, green), blue)
        delta = value - min_value

        saturation = np.zeros_like(value)
        valid_v = value > 0.0
        saturation[valid_v] = delta[valid_v] / value[valid_v]
        
        hue = np.zeros_like(value)
        non_zero_delta = delta > 0.0
        
        red_max = (value == red) & non_zero_delta
        hue[red_max] = 60.0 * ((green[red_max] - blue[red_max]) / delta[red_max])
        
        green_max = (value == green) & non_zero_delta
        hue[green_max] = 120.0 + 60.0 * ((blue[green_max] - red[green_max]) / delta[green_max])
        
        blue_max = (value == blue) & non_zero_delta
        hue[blue_max] = 240.0 + 60.0 * ((red[blue_max] - green[blue_max]) / delta[blue_max])
        
        hue[hue < 0.0] += 360.0

        hue_output = (hue / 2.0).astype(np.uint8)
        saturation_output = (saturation * 255.0).astype(np.uint8)
        value_output = (value * 255.0).astype(np.uint8)
        
        return np.stack([hue_output, saturation_output, value_output], axis=-1)

    raise ValueError(f"Unsupported color conversion: {code}")


def dilate(image, kernel, iterations=1):
    if image is None or kernel is None:
        raise ValueError("Image or kernel cannot be None")
        
    kernel_height, kernel_width = kernel.shape
    pad_y, pad_x = kernel_height // 2, kernel_width // 2
    current_image = image.copy()
    
    for _ in range(iterations):
        horiz_dilated = np.zeros_like(current_image)
        for x in range(-pad_x, pad_x + 1):
            if kernel[pad_y, x + pad_x] == 0:
                continue
            shifted = np.zeros_like(current_image)
            src_x_start = max(0, -x)
            src_x_end = current_image.shape[1] - max(0, x)
            dst_x_start = max(0, x)
            dst_x_end = current_image.shape[1] - max(0, -x)
            
            shifted[:, dst_x_start:dst_x_end] = current_image[:, src_x_start:src_x_end]
            horiz_dilated = np.maximum(horiz_dilated, shifted)
            
        vert_dilated = np.zeros_like(horiz_dilated)
        for y in range(-pad_y, pad_y + 1):
            if kernel[y + pad_y, pad_x] == 0:
                continue
            shifted = np.zeros_like(horiz_dilated)
            src_y_start = max(0, -y)
            src_y_end = horiz_dilated.shape[0] - max(0, y)
            dst_y_start = max(0, y)
            dst_y_end = horiz_dilated.shape[0] - max(0, -y)
            
            shifted[dst_y_start:dst_y_end, :] = horiz_dilated[src_y_start:src_y_end, :]
            vert_dilated = np.maximum(vert_dilated, shifted)
            
        current_image = vert_dilated
        
    return current_image


def normalize(source, destination, alpha=0.0, beta=1.0, norm_type=NORM_MINMAX):
    """Normalize input array in-place."""
    if norm_type != NORM_MINMAX:
        raise ValueError("Only NORM_MINMAX is supported")
        
    source_min, source_max = source.min(), source.max()
    diff = source_max - source_min
    if diff > 0.0:
        destination[:] = (source - source_min) / diff * (beta - alpha) + alpha
    else:
        destination[:] = np.full_like(source, alpha)
    return destination


def goodFeaturesToTrack(image, mask, maxCorners, qualityLevel, minDistance, blockSize):
    """Shi-Tomasi corner detector."""
    if image is None:
        raise ValueError("Image input is None")
    if len(image.shape) == 3:
        raise ValueError("goodFeaturesToTrack expects grayscale image")
        
    def _convolve2d(image_array, kernel):
        h_im, w_im = image_array.shape
        h_k, w_k = kernel.shape
        pad_y, pad_x = h_k // 2, w_k // 2
        padded = np.pad(image_array, ((pad_y, pad_y), (pad_x, pad_x)), mode="edge")
        conv_res = np.zeros_like(image_array, dtype=np.float32)
        for y in range(h_k):
            for x in range(w_k):
                w_val = kernel[y, x]
                if w_val != 0.0:
                    conv_res += w_val * padded[y:y+h_im, x:x+w_im]
        return conv_res

    def _box_filter_2d(image_array, size):
        pad = size // 2
        padded_h = np.pad(image_array, ((0, 0), (pad + 1, pad)), mode="edge")
        cumsum_h = np.cumsum(padded_h, axis=1)
        horiz_sum = cumsum_h[:, size:] - cumsum_h[:, :-size]
        padded_v = np.pad(horiz_sum, ((pad + 1, pad), (0, 0)), mode="edge")
        cumsum_v = np.cumsum(padded_v, axis=0)
        return cumsum_v[size:, :] - cumsum_v[:-size, :]

    gray_image = image.astype(np.float32)
    kernel_x = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=np.float32)
    kernel_y = np.array([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=np.float32)
    
    gradient_x = _convolve2d(gray_image, kernel_x)
    gradient_y = _convolve2d(gray_image, kernel_y)
    
    tensor_xx = _box_filter_2d(gradient_x * gradient_x, blockSize)
    tensor_xy = _box_filter_2d(gradient_x * gradient_y, blockSize)
    tensor_yy = _box_filter_2d(gradient_y * gradient_y, blockSize)
    
    trace = tensor_xx + tensor_yy
    diff_squared = (tensor_xx - tensor_yy) ** 2
    b_sq_4 = 4.0 * (tensor_xy ** 2)
    sqrt_term = np.sqrt(np.maximum(0.0, diff_squared + b_sq_4))
    min_eigenvalue = 0.5 * (trace - sqrt_term)
    
    if mask is not None:
        min_eigenvalue[mask == 0] = 0.0
        
    is_local_max = np.ones_like(min_eigenvalue, dtype=bool)
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            shifted = np.pad(min_eigenvalue, ((max(0, -dy), max(0, dy)), (max(0, -dx), max(0, dx))), mode="constant", constant_values=-1.0)
            cropped = shifted[max(0, dy):max(0, dy)+min_eigenvalue.shape[0], max(0, dx):max(0, dx)+min_eigenvalue.shape[1]]
            is_local_max &= (min_eigenvalue >= cropped)
            
    max_score = min_eigenvalue.max()
    threshold = qualityLevel * max_score
    candidates = is_local_max & (min_eigenvalue > threshold) & (min_eigenvalue > 0.0)
    
    y_coords, x_coords = np.where(candidates)
    scores = min_eigenvalue[candidates]
    sorted_indices = np.argsort(scores)[::-1]
    sorted_y_coords = y_coords[sorted_indices]
    sorted_x_coords = x_coords[sorted_indices]
    
    chosen_corners = []
    min_distance_squared = minDistance ** 2
    for cy, cx in zip(sorted_y_coords, sorted_x_coords):
        too_close = False
        for py, px in chosen_corners:
            if (cy - py)**2 + (cx - px)**2 < min_distance_squared:
                too_close = True
                break
        if not too_close:
            chosen_corners.append((cy, cx))
            if len(chosen_corners) >= maxCorners:
                break
                
    if not chosen_corners:
        return None
        
    return np.array([[[float(x), float(y)]] for y, x in chosen_corners], dtype=np.float32)
    # 2D window of image 
def _get_subpixel_window_multi(img, cys, cxs, grid_dx, grid_dy):
    height, width = img.shape
    xv = grid_dx[np.newaxis, :, :] + cxs[:, np.newaxis, np.newaxis]
    yv = grid_dy[np.newaxis, :, :] + cys[:, np.newaxis, np.newaxis]
    
    x0 = np.floor(xv).astype(np.int32)
    x1 = x0 + 1
    y0 = np.floor(yv).astype(np.int32)
    y1 = y0 + 1
    
    x0_clipped = np.clip(x0, 0, width - 1)
    x1_clipped = np.clip(x1, 0, width - 1)
    y0_clipped = np.clip(y0, 0, height - 1)
    y1_clipped = np.clip(y1, 0, height - 1)
    
    weight_left_top = (x1 - xv) * (y1 - yv)
    weight_right_top = (xv - x0) * (y1 - yv)
    weight_left_bottom = (x1 - xv) * (yv - y0)
    weight_right_bottom = (xv - x0) * (yv - y0)
    
    result = weight_left_top * img[y0_clipped, x0_clipped] + weight_right_top * img[y0_clipped, x1_clipped] + weight_left_bottom * img[y1_clipped, x0_clipped] + weight_right_bottom * img[y1_clipped, x1_clipped]
    return result
def calcOpticalFlowPyrLK(
    prev_gray,
    curr_gray,
    p0,
    next_pts=None,
    winSize=(21, 21),
    maxLevel=3,
    criteria=None
):
    if prev_gray is None or curr_gray is None or p0 is None:
        raise ValueError("frames or keypoints cannot be None")

    max_iters = 30
    epsilon = 0.01
    if criteria is not None:
        max_iters = criteria[1]
        epsilon = criteria[2]
        
    w_w, w_h = winSize
    grid_dx, grid_dy = np.meshgrid(np.arange(w_w) - w_w // 2, np.arange(w_h) - w_h // 2)
    


    # build image pyramids
    pyramid_prev = [prev_gray.astype(np.float32)]
    pyramid_current = [curr_gray.astype(np.float32)]
    for _ in range(maxLevel):
        for pyr, src_pyr in [(pyramid_prev, pyramid_prev), (pyramid_current, pyramid_current)]:
            img = src_pyr[-1]
            height, width = img.shape
            height_new, width_new = height // 2, width // 2
            downsampled = 0.25 * (
                img[0:2*height_new:2, 0:2*width_new:2] +
                img[1:2*height_new:2, 0:2*width_new:2] +
                img[0:2*height_new:2, 1:2*width_new:2] +
                img[1:2*height_new:2, 1:2*width_new:2]
            )
            pyr.append(downsampled)
        
    pyramid_x = []
    pyramid_y = []
    for img_level in pyramid_prev:
        gx = np.zeros_like(img_level)
        gy = np.zeros_like(img_level)
        gx[:, 1:-1] = 0.5 * (img_level[:, 2:] - img_level[:, :-2])
        gy[1:-1, :] = 0.5 * (img_level[2:, :] - img_level[:-2, :])
        pyramid_x.append(gx)
        pyramid_y.append(gy)
        
    num_points = p0.shape[0]
    prev_xs = p0[:, 0, 0].copy()
    prev_ys = p0[:, 0, 1].copy()
    
    displacements_x = np.zeros(num_points, dtype=np.float32)
    displacements_y = np.zeros(num_points, dtype=np.float32)
    status = np.ones(num_points, dtype=np.uint8)
    
    for lvl in range(maxLevel, -1, -1):
        scale = 2.0 ** lvl
        img_i = pyramid_prev[lvl]
        img_j = pyramid_current[lvl]
        grad_x = pyramid_x[lvl]
        grad_y = pyramid_y[lvl]
        
        active = np.where(status == 1)[0]
        if len(active) == 0:
            break
            
        active_xs = prev_xs[active] / scale
        active_ys = prev_ys[active] / scale
        
        template_window = _get_subpixel_window_multi(img_i, active_ys, active_xs, grid_dx, grid_dy)
        template_grad_x = _get_subpixel_window_multi(grad_x, active_ys, active_xs, grid_dx, grid_dy)
        template_grad_y = _get_subpixel_window_multi(grad_y, active_ys, active_xs, grid_dx, grid_dy)
        
        hessian_00 = np.sum(template_grad_x * template_grad_x, axis=(1, 2))
        hessian_01 = np.sum(template_grad_x * template_grad_y, axis=(1, 2))
        hessian_11 = np.sum(template_grad_y * template_grad_y, axis=(1, 2))
        
        det = hessian_00 * hessian_11 - hessian_01 * hessian_01
        invalid = det < 1e-6
        status[active[invalid]] = 0
        
        valid_active = np.where(det >= 1e-6)[0]
        if len(valid_active) == 0:
            continue
            
        active = active[valid_active]
        active_xs = active_xs[valid_active]
        active_ys = active_ys[valid_active]
        template_window = template_window[valid_active]
        template_grad_x = template_grad_x[valid_active]
        template_grad_y = template_grad_y[valid_active]
        det = det[valid_active]
        hessian_00 = hessian_00[valid_active]
        hessian_01 = hessian_01[valid_active]
        hessian_11 = hessian_11[valid_active]
        
        inv_hessian_00 = hessian_11 / det
        inv_hessian_01 = -hessian_01 / det
        inv_hessian_11 = hessian_00 / det
        
        lvl_displacements_x = displacements_x[active] / scale
        lvl_displacements_y = displacements_y[active] / scale
        
        for iter_idx in range(max_iters):
            search_window = _get_subpixel_window_multi(img_j, active_ys + lvl_displacements_y, active_xs + lvl_displacements_x, grid_dx, grid_dy)
            error_difference = template_window - search_window
            b0_val = np.sum(error_difference * template_grad_x, axis=(1, 2))
            b1_val = np.sum(error_difference * template_grad_y, axis=(1, 2))
            
            update_x = inv_hessian_00 * b0_val + inv_hessian_01 * b1_val
            update_y = inv_hessian_01 * b0_val + inv_hessian_11 * b1_val
            
            lvl_displacements_x += update_x
            lvl_displacements_y += update_y
            
            if np.max(update_x ** 2 + update_y ** 2) < epsilon ** 2:
                break
                
        displacements_x[active] = lvl_displacements_x * scale
        displacements_y[active] = lvl_displacements_y * scale
        
    final_xs = prev_xs + displacements_x
    final_ys = prev_ys + displacements_y
    
    orig_height, orig_width = prev_gray.shape
    next_points = np.zeros_like(p0)
    output_status = np.zeros((num_points, 1), dtype=np.uint8)
    output_errors = np.zeros((num_points, 1), dtype=np.float32)
    
    for i in range(num_points):
        if status[i] == 1 and 0 <= final_xs[i] < orig_width and 0 <= final_ys[i] < orig_height:
            next_points[i, 0, 0] = final_xs[i]
            next_points[i, 0, 1] = final_ys[i]
            output_status[i, 0] = 1
            output_errors[i, 0] = 0.0
            
    return next_points, output_status, output_errors

def estimateAffinePartial2D(
    from_pts,
    to_pts,
    method=RANSAC,
    ransacReprojThreshold=3.0,
    max_iters=200
):
    points_from = from_pts.reshape(-1, 2)
    points_to = to_pts.reshape(-1, 2)
    num_points = points_from.shape[0]
    
    if num_points < 2:
        return None, np.zeros(num_points, dtype=np.uint8)
        
    best_inliers = np.zeros(num_points, dtype=np.uint8)
    best_num_inliers = -1
    best_model = None
    rng = np.random.RandomState(42)
    
    for _ in range(max_iters):
        indices = rng.choice(num_points, 2, replace=False)
        pt_from1, pt_from2 = points_from[indices[0]], points_from[indices[1]]
        pt_to1, pt_to2 = points_to[indices[0]], points_to[indices[1]]
        
        matrix_x = np.array([
            [pt_from1[0], -pt_from1[1], 1.0, 0.0],
            [pt_from1[1],  pt_from1[0], 0.0, 1.0],
            [pt_from2[0], -pt_from2[1], 1.0, 0.0],
            [pt_from2[1],  pt_from2[0], 0.0, 1.0]
        ], dtype=np.float32)
        
        if np.abs(np.linalg.det(matrix_x)) < 1e-5:
            continue
            
        vector_y = np.array([pt_to1[0], pt_to1[1], pt_to2[0], pt_to2[1]], dtype=np.float32)
        transform_params = np.linalg.solve(matrix_x, vector_y)
        scale_cos, scale_sin, translation_x, translation_y = transform_params
        
        projected_x = scale_cos * points_from[:, 0] - scale_sin * points_from[:, 1] + translation_x
        projected_y = scale_sin * points_from[:, 0] + scale_cos * points_from[:, 1] + translation_y
        
        distance_squared = (points_to[:, 0] - projected_x)**2 + (points_to[:, 1] - projected_y)**2
        inliers = (distance_squared < ransacReprojThreshold**2).astype(np.uint8)
        num_inliers = np.sum(inliers)
        
        if num_inliers > best_num_inliers:
            best_num_inliers = num_inliers
            best_inliers = inliers
            best_model = transform_params
            
    if best_model is None or best_num_inliers < 2:
        return None, np.zeros(num_points, dtype=np.uint8)
        
    inlier_indices = np.where(best_inliers == 1)[0]
    matrix_a_builder = []
    vector_b_builder = []
    for idx in inlier_indices:
        p, q = points_from[idx], points_to[idx]
        matrix_a_builder.append([p[0], -p[1], 1.0, 0.0])
        matrix_a_builder.append([p[1],  p[0], 0.0, 1.0])
        vector_b_builder.extend([q[0], q[1]])
        
    matrix_a = np.array(matrix_a_builder, dtype=np.float32)
    vector_b = np.array(vector_b_builder, dtype=np.float32)
    refined_params, _, _, _ = np.linalg.lstsq(matrix_a, vector_b, rcond=None)
    scale_cos, scale_sin, translation_x, translation_y = refined_params
    
    affine_matrix = np.array([
        [scale_cos, -scale_sin, translation_x],
        [scale_sin,  scale_cos, translation_y]
    ], dtype=np.float32)
    
    return affine_matrix, best_inliers.reshape(-1, 1)

def calcHist(
    images,
    channels,
    mask,
    histSize,
    ranges
):
    img = images[0]
    channel_1 = img[:, :, channels[0]]
    channel_2 = img[:, :, channels[1]]
    
    valid_pixels = mask > 0
    values_1 = channel_1[valid_pixels]
    values_2 = channel_2[valid_pixels]
    
    if len(values_1) == 0:
        return np.zeros((histSize[0], histSize[1]), dtype=np.float32)
        
    range_1_min, range_1_max = ranges[0], ranges[1]
    range_2_min, range_2_max = ranges[2], ranges[3]
    
    histogram_2d, _, _ = np.histogram2d(
        values_1, values_2,
        bins=histSize,
        range=[[range_1_min, range_1_max], [range_2_min, range_2_max]]
    )
    return histogram_2d.astype(np.float32)

def compareHist(hist_1, hist_2, method):
    if method != HISTCMP_BHATTACHARYYA:
        raise ValueError("Only HISTCMP_BHATTACHARYYA is supported")
    sum_1, sum_2 = np.sum(hist_1), np.sum(hist_2)
    if sum_1 == 0.0 or sum_2 == 0.0:
        return 1.0
        
    coefficient = np.sum(np.sqrt(hist_1 * hist_2)) / np.sqrt(sum_1 * sum_2)
    distance_value = 1.0 - coefficient
    if distance_value < 0.0:
        distance_value = 0.0
        
    return float(np.sqrt(distance_value))
