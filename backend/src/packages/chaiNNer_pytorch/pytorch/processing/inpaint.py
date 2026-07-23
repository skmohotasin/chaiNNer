from __future__ import annotations

import gc

import numpy as np
import torch
from spandrel import MaskedImageModelDescriptor

import navi
from api import NodeContext
from nodes.impl.image_utils import as_target_channels
from nodes.impl.pytorch.utils import (
    move_model_to_device,
    np2tensor,
    safe_cuda_cache_empty,
    tensor2np,
)
from nodes.properties.inputs import ImageInput
from nodes.properties.inputs.pytorch_inputs import InpaintModelInput
from nodes.properties.outputs import ImageOutput
from nodes.utils.utils import get_h_w_c

from ...settings import PyTorchSettings, get_settings
from .. import processing_group


def inpaint(
    img: np.ndarray,
    mask: np.ndarray,
    model: MaskedImageModelDescriptor,
    options: PyTorchSettings,
):
    with torch.no_grad():
        # TODO: use bfloat16 if RTX
        use_fp16 = options.use_fp16 and model.supports_half
        dtype = torch.float16 if use_fp16 else torch.float32
        device = options.device

        # Spandrel descriptors expose .to; also move MAT's loose Tensor attrs for XPU.
        model = model.to(device, dtype)
        move_model_to_device(model.model, device, dtype)

        img_tensor = np2tensor(img, change_range=True)
        mask_tensor = np2tensor(mask, change_range=True)

        d_img = None
        d_mask = None
        try:
            d_img = img_tensor.to(device, dtype)

            d_mask = mask_tensor.to(device, dtype)
            d_mask = (d_mask > 0.5) * 1
            d_mask = d_mask.to(dtype)

            result = model(d_img, d_mask)
            result = tensor2np(
                result.detach().cpu(),
                change_range=False,
                imtype=np.float32,
            )

            return result
        finally:
            # Collect garbage (clear VRAM)
            if d_img is not None:
                try:
                    d_img.detach().cpu()
                except Exception:
                    pass
                del d_img
            if d_mask is not None:
                try:
                    d_mask.detach().cpu()
                except Exception:
                    pass
                del d_mask

            gc.collect()
            safe_cuda_cache_empty()


@processing_group.register(
    schema_id="chainner:pytorch:inpaint",
    name="Inpaint",
    description=[
        "Inpaint an image with given mask, using a PyTorch inpainting model.",
        "Accepts RGB or RGBA images (alpha is preserved if present).",
        "Masks must typically be made outside of chaiNNer.",
        "Supported models include LaMa and MAT",
    ],
    icon="PyTorch",
    inputs=[
        ImageInput(channels=[3, 4]),
        ImageInput("Mask", channels=1).with_docs(
            "An inpainting mask is a grayscale image where white represents what to inpaint and black represents what to keep.",
            "This must typically be made outside of chaiNNer.",
            hint=True,
        ),
        InpaintModelInput(),
    ],
    outputs=[
        ImageOutput(
            image_type=navi.Image(
                width="Input0.width & Input1.width",
                height="Input0.height & Input1.height",
                channels="Input0.channels",
            ),
        ).with_never_reason("The given image and mask must have the same resolution.")
    ],
    node_context=True,
)
def inpaint_node(
    context: NodeContext,
    img: np.ndarray,
    mask: np.ndarray,
    model: MaskedImageModelDescriptor,
) -> np.ndarray:
    assert img.shape[:2] == mask.shape[:2], (
        "Input image and mask must have the same resolution"
    )

    exec_options = get_settings(context)
    context.add_cleanup(
        safe_cuda_cache_empty,
        after="node" if exec_options.force_cache_wipe else "chain",
    )

    channels = get_h_w_c(img)[2]
    alpha = img[:, :, 3] if channels == 4 else None
    rgb = as_target_channels(img, 3, narrowing=True)

    result = inpaint(rgb, mask, model, exec_options)
    if alpha is not None:
        return np.dstack((result, alpha))
    return result

