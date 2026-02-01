import torch

# ComfyUI IMAGE format: torch float tensor in [0,1], shape [B,H,W,C] where C=3
# This node offsets like GIMP Offset. Supports wrap-around or fill modes.

class MuffinsOffsetNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),

                "units": (["pixels", "percent"], {"default": "pixels"}),

                "x_offset": ("FLOAT", {"default": 0.0, "min": -100000.0, "max": 100000.0, "step": 1.0}),
                "y_offset": ("FLOAT", {"default": 0.0, "min": -100000.0, "max": 100000.0, "step": 1.0}),

                # Convenience for panoramas: set X=W/2 (and Y=0 unless you want it)
                "auto_half_width": ("BOOLEAN", {"default": False}),
                "auto_half_height": ("BOOLEAN", {"default": False}),

                "edge_mode": (["wrap", "fill_color", "transparent_black"], {"default": "wrap"}),

                # Fill color only used if edge_mode == fill_color
                "fill_r": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "fill_g": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "fill_b": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "offset"
    CATEGORY = "image/transform"

    def offset(
        self,
        image,
        units,
        x_offset,
        y_offset,
        auto_half_width,
        auto_half_height,
        edge_mode,
        fill_r,
        fill_g,
        fill_b,
    ):
        if not isinstance(image, torch.Tensor):
            raise TypeError("Expected IMAGE as a torch.Tensor")

        # image: [B,H,W,C]
        b, h, w, c = image.shape
        if c != 3:
            raise ValueError(f"Expected 3-channel IMAGE, got C={c}")

        # Determine offsets in pixels
        if auto_half_width:
            x_px = w // 2
        else:
            if units == "percent":
                x_px = int(round((x_offset / 100.0) * w))
            else:
                x_px = int(round(x_offset))

        if auto_half_height:
            y_px = h // 2
        else:
            if units == "percent":
                y_px = int(round((y_offset / 100.0) * h))
            else:
                y_px = int(round(y_offset))

        # Normalize to [-w..w], [-h..h] range for safety
        if w != 0:
            x_px = x_px % w
            # Make it symmetric like typical offset tools
            if x_px > w // 2:
                x_px -= w
        if h != 0:
            y_px = y_px % h
            if y_px > h // 2:
                y_px -= h

        if x_px == 0 and y_px == 0:
            return (image,)

        if edge_mode == "wrap":
            # torch.roll wraps around both axes
            out = torch.roll(image, shifts=(y_px, x_px), dims=(1, 2))
            return (out,)

        # Non-wrapping modes: create a filled canvas and paste shifted image into it
        device = image.device
        dtype = image.dtype

        if edge_mode == "fill_color":
            fill = torch.tensor([fill_r, fill_g, fill_b], device=device, dtype=dtype).view(1, 1, 1, 3)
            out = fill.expand(b, h, w, 3).clone()
        else:
            # transparent_black (no real alpha in ComfyUI IMAGE), so we fill with black
            out = torch.zeros((b, h, w, 3), device=device, dtype=dtype)

        # Compute source and destination rectangles
        # Positive x_px means shift RIGHT, so source starts at 0 and dest starts at x_px
        # Negative x_px means shift LEFT, so source starts at -x_px and dest starts at 0
        if x_px >= 0:
            src_x0, dst_x0 = 0, x_px
            width_copy = w - x_px
        else:
            src_x0, dst_x0 = -x_px, 0
            width_copy = w + x_px  # x_px is negative

        if y_px >= 0:
            src_y0, dst_y0 = 0, y_px
            height_copy = h - y_px
        else:
            src_y0, dst_y0 = -y_px, 0
            height_copy = h + y_px

        if width_copy <= 0 or height_copy <= 0:
            return (out,)

        out[:, dst_y0:dst_y0 + height_copy, dst_x0:dst_x0 + width_copy, :] = \
            image[:, src_y0:src_y0 + height_copy, src_x0:src_x0 + width_copy, :]

        return (out,)


NODE_CLASS_MAPPINGS = {
    "MuffinsOffsetNode": MuffinsOffsetNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MuffinsOffsetNode": "MuffinsOffsetNode"
}

