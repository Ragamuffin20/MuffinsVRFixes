import torch

class VR180StereoTools:
    """
    Utility node for VR180 side-by-side workflows:
    - Extract one half (mono) from SBS
    - Copy one half across both halves (SBS->SBS, no mirror)
    - Rebuild SBS from mono half (mono->SBS)
    - Optional even-width cropping control
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),

                # What do we want to do?
                "mode": ([
                    "sbs_extract_half",          # SBS -> mono (one half)
                    "sbs_copy_half_to_stereo",   # SBS -> SBS (copy selected half to both halves)
                    "mono_to_stereo_copy",       # mono -> SBS (duplicate mono into both halves)
                    "even_crop_only"             # just enforce even width (no other change)
                ], {"default": "sbs_copy_half_to_stereo"}),

                # Which half to use when mode uses a source half
                "source_half": (["left", "right"], {"default": "left"}),

                # Output ordering for stereo outputs
                "output_layout": (["cross_eyed", "parallel"], {"default": "cross_eyed"}),

                # Handle odd widths
                "even_width_handling": ([
                    "auto_crop_if_odd",  # if W is odd, drop last column
                    "skip"               # do nothing even if odd (may break perfect halving)
                ], {"default": "auto_crop_if_odd"}),

                # Optional seam feather for stereo outputs (center blend)
                "seam_feather": ("INT", {"default": 0, "min": 0, "max": 256, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "apply"
    CATEGORY = "video/vr"

    def _ensure_even_width(self, images, even_width_handling):
        if even_width_handling == "skip":
            return images

        # auto_crop_if_odd
        B, H, W, C = images.shape
        if W % 2 != 0:
            images = images[:, :, :W-1, :]
        return images

    def _seam_feather(self, out, seam_feather, half_width):
        if not seam_feather or seam_feather <= 0:
            return out

        f = min(seam_feather, half_width)
        seam_l_start = half_width - f
        seam_l_end = half_width
        seam_r_start = half_width
        seam_r_end = half_width + f

        ramp = torch.linspace(0.0, 1.0, steps=f, device=out.device, dtype=out.dtype).view(1, 1, f, 1)
        left_band = out[:, :, seam_l_start:seam_l_end, :]
        right_band = out[:, :, seam_r_start:seam_r_end, :]

        blended = left_band * (1.0 - ramp) + right_band * ramp
        out[:, :, seam_l_start:seam_l_end, :] = blended
        out[:, :, seam_r_start:seam_r_end, :] = blended
        return out

    def apply(self, images, mode="sbs_copy_half_to_stereo", source_half="left",
              output_layout="cross_eyed", even_width_handling="auto_crop_if_odd", seam_feather=0):

        if not isinstance(images, torch.Tensor):
            raise ValueError("images must be a torch Tensor")
        if images.dim() != 4:
            raise ValueError(f"Expected images with shape [B,H,W,C], got {tuple(images.shape)}")

        images = self._ensure_even_width(images, even_width_handling)
        B, H, W, C = images.shape

        # Mode: only enforce even width and exit
        if mode == "even_crop_only":
            return (torch.clamp(images, 0.0, 1.0),)

        # MODE A: SBS -> mono (extract half)
        if mode == "sbs_extract_half":
            if W % 2 != 0:
                raise ValueError("Width is odd and even_width_handling='skip'. Can't split evenly.")
            half = W // 2
            left = images[:, :, :half, :]
            right = images[:, :, half:, :]
            out = left if source_half == "left" else right
            return (torch.clamp(out, 0.0, 1.0),)

        # MODE B: SBS -> SBS (copy selected half into both halves)
        if mode == "sbs_copy_half_to_stereo":
            if W % 2 != 0:
                raise ValueError("Width is odd and even_width_handling='skip'. Can't split evenly.")
            half = W // 2
            left = images[:, :, :half, :]
            right = images[:, :, half:, :]

            src = left if source_half == "left" else right
            other = src.clone()

            # Build stereo output
            if output_layout == "parallel":
                # [L|R]
                if source_half == "left":
                    out = torch.cat([src, other], dim=2)
                else:
                    out = torch.cat([other, src], dim=2)
            else:
                # cross-eyed typically [R|L]
                # We output matched pair; treat src as "R" and other as "L" => [src|other]
                out = torch.cat([src, other], dim=2)

            out = self._seam_feather(out, seam_feather, half)
            return (torch.clamp(out, 0.0, 1.0),)

        # MODE C: mono -> SBS (duplicate mono into both halves)
        if mode == "mono_to_stereo_copy":
            # Here images is a mono half (per-eye frame). We rebuild SBS.
            mono = images
            half = mono.shape[2]
            other = mono.clone()

            if output_layout == "parallel":
                # [L|R]
                out = torch.cat([mono, other], dim=2)
            else:
                # cross-eyed [R|L] (matched pair anyway)
                out = torch.cat([mono, other], dim=2)

            out = self._seam_feather(out, seam_feather, half)
            return (torch.clamp(out, 0.0, 1.0),)

        raise ValueError(f"Unknown mode: {mode}")


NODE_CLASS_MAPPINGS = {
    "VR180StereoTools": VR180StereoTools
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VR180StereoTools": "VR180 Stereo Tools (Extract / Rebuild / Copy)"
}
