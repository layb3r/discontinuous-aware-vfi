from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .image_encoder import TinyViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder
from .segmentation import GOSeg
from .transformer import TwoWayTransformer


def build_goseg(
    checkpoint: Optional[str] = None,
    yolo_weights_path: Optional[str] = None,
    tracker_config_path: Optional[str] = None,
    yolo_config: Optional[Dict[str, Any]] = None,
    use_detector: bool = True,
) -> GOSeg:
    """Build a GOSeg model with MobileSAM-style components."""
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size

    this_dir = Path(__file__).resolve().parent
    default_yolo_weights = this_dir.parent / "weights" / "detector.pt"
    default_tracker_cfg = this_dir / "botsort.yaml"

    model = GOSeg(
        image_encoder=TinyViT(
            img_size=1024,
            in_chans=3,
            num_classes=1000,
            embed_dims=[64, 128, 160, 320],
            depths=[2, 2, 6, 2],
            num_heads=[2, 4, 5, 10],
            window_sizes=[7, 7, 14, 7],
            mlp_ratio=4.0,
            drop_rate=0.0,
            drop_path_rate=0.0,
            use_checkpoint=False,
            mbconv_expand_ratio=4.0,
            local_conv_size=3,
            layer_lr_decay=0.8,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
        yolo_weights_path=(yolo_weights_path or str(default_yolo_weights)),
        tracker_config_path=(tracker_config_path or str(default_tracker_cfg)),
        yolo_config=yolo_config,
        use_detector=use_detector,
    )

    if checkpoint is not None:
        state_dict = torch.load(checkpoint, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict, strict=False)

    return model