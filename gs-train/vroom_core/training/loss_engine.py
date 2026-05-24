import torch
import torch.nn.functional as F

class LossEngine:
    def __init__(self, semantic_manager):
        self.semantic_manager = semantic_manager

    def calc_l1_loss(self, render, real_image):
        return torch.mean(torch.abs(render - real_image))

    def calc_ssim_loss(self, render, real_image):
        window_size = 11
        padding = window_size // 2

        # because avg_pool2d requires 4D inputs (Num of batches, color Channels, Height, Width)
        render_4d = render.unsqueeze(0) if render.dim() == 3 else render
        real_image_4d = real_image.unsqueeze(0) if real_image.dim() == 3 else real_image

        # Represents the average luminance (mu)
        mean_filter_render = F.avg_pool2d(render_4d, kernel_size=window_size, stride=1, padding=padding) # mu x
        mean_filter_real = F.avg_pool2d(real_image_4d, kernel_size=window_size, stride=1, padding=padding) # mu y

        mean_filter_render_squared = mean_filter_render ** 2 # (mu x)^2
        mean_filter_real_squared = mean_filter_real ** 2 # (mu y)^2

        # product of means, not product of squared means
        mean_product = mean_filter_render * mean_filter_real # (mu x) * (mu y)

        # variance calculation: E(x^2) - E(x)^2
        real_img_squared_avg = F.avg_pool2d(real_image_4d ** 2, kernel_size=window_size, stride=1, padding=padding) # E(x^2)
        render_squared_avg = F.avg_pool2d(render_4d ** 2, kernel_size=window_size, stride=1, padding=padding) # E(y^2)
        real_and_render_avg = F.avg_pool2d(render_4d * real_image_4d, kernel_size=window_size, stride=1, padding=padding) # E(xy)

        variance_real_image = real_img_squared_avg - mean_filter_real_squared
        variance_render_image = render_squared_avg - mean_filter_render_squared

        # covariance = E(xy) - mu_x * mu_y
        covariance = real_and_render_avg - mean_product

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        # numerator uses 2 * mu_x * mu_y, not 2 * (mu_x^2 * mu_y^2)
        numerator = (2 * mean_product + C1) * (2 * covariance + C2)
        denominator = (mean_filter_render_squared + mean_filter_real_squared + C1) * (variance_real_image + variance_render_image + C2)

        return 1 - (numerator / denominator).mean()

    def calc_semantic_loss(self, mask, model_guess):
        mask_label_idx = self.semantic_manager.build_lookup_table(mask)
        model_guess = model_guess.unsqueeze(0)
        mask_label_idx = mask_label_idx.unsqueeze(0)
        return F.cross_entropy(model_guess, mask_label_idx, ignore_index=0, reduction="mean")

    def calc_volumetric_loss(self, scales, volume_lambda):
        volumes = torch.prod(scales, dim=1) # multiply the x, y and z scales to get the volumes of each gaussian
        return torch.sum(volumes) * volume_lambda # sum them up and multiply by the lambda

    def compute_total_losses(self, render_pkg, viewpoint_cam, anchor_cloud, opt, iteration):
        render = render_pkg["render"]
        real_image = viewpoint_cam.original_image.to(render.device)

        mask = getattr(viewpoint_cam, "object_mask", None)
        if mask is None:
            mask = torch.zeros(render.shape[1:], dtype=torch.long, device=render.device)
        else:
            mask = mask.to(render.device)

        model_guess = render_pkg.get("render_semantics")

        scales = render_pkg["scaling"]
        lambda_dssim = getattr(opt, "lambda_dssim", 0.2)
        lambda_object_loss = getattr(opt, "lambda_object_loss", 0.1)
        volume_lambda = getattr(opt, "lambda_dreg", 0.01)

        l1_loss = self.calc_l1_loss(render, real_image)
        ssim_loss = self.calc_ssim_loss(render, real_image)
        # Standard 3DGS blend: (1 - λ) * L1 + λ * SSIM
        rgb_loss = (1.0 - lambda_dssim) * l1_loss + lambda_dssim * ssim_loss

        has_mask = getattr(viewpoint_cam, "object_mask", None) is not None
        if has_mask and model_guess is not None:
            semantic_loss = self.calc_semantic_loss(mask, model_guess) * lambda_object_loss
        else:
            semantic_loss = torch.zeros(1, device=render.device).squeeze()
        volumetric_loss = self.calc_volumetric_loss(scales, volume_lambda)

        return {
            "rgb_loss": rgb_loss,
            "semantic_loss": semantic_loss,
            "volumetric_loss": volumetric_loss,
        }
