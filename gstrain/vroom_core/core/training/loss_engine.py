import torch
import torch.nn.functional as F


class LossEngine:
    def __init__(self, semantic_manager):
        self.semantic_manager = semantic_manager

    def calc_volumetric_loss(self, scales: torch.Tensor, volume_lambda):
        """Calculates volumetric loss from scaling factors."""
        volumes = torch.prod(
            scales, dim=1
        )  # multiply the x, y and z scales to get the volumes of each gaussian
        return torch.mean(volumes) * volume_lambda

    def calc_l1_loss(self, render, real_image):
        return torch.mean(torch.abs(render - real_image))

    def calc_ssim_loss(
        self,
        render,
        real_image,
        window_size,
        padding,
        luminance_stabilizer,
        contrast_stabilizer,
    ):
        # because avg_pool2d requires 4D inputs (Num of batches, color Channels, Height, Width)
        render_4d = render.unsqueeze(0) if render.dim() == 3 else render
        real_image_4d = real_image.unsqueeze(0) if real_image.dim() == 3 else real_image

        # Represents the average luminance (mu)
        mean_filter_render = F.avg_pool2d(
            render_4d, kernel_size=window_size, stride=1, padding=padding
        )  # mu x
        mean_filter_real = F.avg_pool2d(
            real_image_4d, kernel_size=window_size, stride=1, padding=padding
        )  # mu y

        mean_filter_render_squared = mean_filter_render**2  # (mu x)^2
        mean_filter_real_squared = mean_filter_real**2  # (mu y)^2

        # product of means, not product of squared means
        mean_product = mean_filter_render * mean_filter_real  # (mu x) * (mu y)

        # variance calculation: E(x^2) - E(x)^2
        real_img_squared_avg = F.avg_pool2d(
            real_image_4d**2, kernel_size=window_size, stride=1, padding=padding
        )  # E(x^2)
        render_squared_avg = F.avg_pool2d(
            render_4d**2, kernel_size=window_size, stride=1, padding=padding
        )  # E(y^2)
        real_and_render_avg = F.avg_pool2d(
            render_4d * real_image_4d,
            kernel_size=window_size,
            stride=1,
            padding=padding,
        )  # E(xy)

        variance_real_image = real_img_squared_avg - mean_filter_real_squared
        variance_render_image = render_squared_avg - mean_filter_render_squared

        # covariance = E(xy) - mu_x * mu_y
        covariance = real_and_render_avg - mean_product

        # numerator uses 2 * mu_x * mu_y, not 2 * (mu_x^2 * mu_y^2)
        numerator = (2 * mean_product + luminance_stabilizer) * (
            2 * covariance + contrast_stabilizer
        )
        denominator = (
            mean_filter_render_squared + mean_filter_real_squared + luminance_stabilizer
        ) * (variance_real_image + variance_render_image + contrast_stabilizer)

        return 1 - (numerator / denominator).mean()

    def calc_semantic_loss(self, mask, model_guess):
        mask_label_idx = self.semantic_manager.build_lookup_table(mask)
        model_guess = model_guess.unsqueeze(0)
        mask_label_idx = mask_label_idx.unsqueeze(0)
        return F.cross_entropy(
            model_guess, mask_label_idx, ignore_index=0, reduction="mean"
        )

    def compute_total_losses(
        self, render_pkg, viewpoint_cam, anchor_cloud, optimizer_configs, iteration
    ):
        render = render_pkg["render"]
        real_image = viewpoint_cam.original_image.to(render.device)

        mask = getattr(viewpoint_cam, "object_mask", None)
        if mask is None:
            mask = torch.zeros(render.shape[1:], dtype=torch.long, device=render.device)
        else:
            mask = mask.to(render.device)

        model_guess = render_pkg.get("render_semantics")

        scales = render_pkg["scaling"]
        ssim_weight = getattr(optimizer_configs, "ssim_weight")
        semantic_loss_weight = getattr(optimizer_configs, "semantic_loss_weight")
        volume_reg_weight = getattr(optimizer_configs, "volume_reg_weight")

        ssim_window_size = getattr(optimizer_configs, "ssim_window_size")
        ssim_padding = getattr(optimizer_configs, "ssim_padding")
        ssim_luminance_stabilizer = getattr(
            optimizer_configs, "ssim_luminance_stabilizer"
        )
        ssim_contrast_stabilizer = getattr(
            optimizer_configs, "ssim_contrast_stabilizer"
        )

        l1_loss = self.calc_l1_loss(render, real_image)
        ssim_loss = self.calc_ssim_loss(
            render,
            real_image,
            window_size=ssim_window_size,
            padding=ssim_padding,
            luminance_stabilizer=ssim_luminance_stabilizer,
            contrast_stabilizer=ssim_contrast_stabilizer,
        )

        rgb_loss = (1.0 - ssim_weight) * l1_loss + ssim_weight * ssim_loss

        has_mask = getattr(viewpoint_cam, "object_mask", None) is not None
        if has_mask and model_guess is not None:
            semantic_loss = (
                self.calc_semantic_loss(mask, model_guess) * semantic_loss_weight
            )
        else:
            semantic_loss = torch.zeros(1, device=render.device).squeeze()
        volumetric_loss = self.calc_volumetric_loss(scales, volume_reg_weight)

        return {
            "rgb_loss": rgb_loss,
            "semantic_loss": semantic_loss,
            "volumetric_loss": volumetric_loss,
        }
