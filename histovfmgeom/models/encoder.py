import timm
import torch
import torch.nn as nn


def build_encoder(encoder_id: str) -> tuple[nn.Module, dict]:
    if encoder_id == "uni2h":
        amp_dtype = torch.bfloat16
        timm_kwargs = {
            "model_name": "hf-hub:MahmoodLab/UNI2-h",
            "pretrained": True,
            "img_size": 224,
            "patch_size": 14,
            "depth": 24,
            "num_heads": 24,
            "init_values": 1e-5,
            "embed_dim": 1536,
            "mlp_ratio": 2.66667 * 2,
            "num_classes": 0,
            "no_embed_class": True,
            "mlp_layer": timm.layers.SwiGLUPacked,
            "act_layer": torch.nn.SiLU,
            "reg_tokens": 8,
            "dynamic_img_size": True,
        }
        encoder = timm.create_model(**timm_kwargs)

        embed_dim = 1536
        patch_size = 14
        pixel_mean = encoder.default_cfg["mean"]
        pixel_std = encoder.default_cfg["std"]
        n_blocks = len(encoder.blocks)
    elif encoder_id == "h-optimus-1":
        amp_dtype = torch.float16
        encoder = timm.create_model(
            "hf-hub:bioptimus/H-optimus-1",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=True,
        )
        embed_dim = 1536
        patch_size = 14
        pixel_mean = [0.707223, 0.578729, 0.703617]
        pixel_std = [0.211883, 0.230117, 0.177517]
        n_blocks = len(encoder.blocks)
    elif encoder_id == "h0-mini":
        amp_dtype = torch.float16
        encoder = timm.create_model(
            "hf-hub:bioptimus/H0-mini",
            pretrained=True,
            mlp_layer=timm.layers.SwiGLUPacked,
            act_layer=torch.nn.SiLU,
            dynamic_img_size=True,  # keep this so your hooks work on 448
        )
        embed_dim = getattr(encoder, "embed_dim", 768)
        patch_size = 14
        pixel_mean = encoder.default_cfg["mean"]
        pixel_std = encoder.default_cfg["std"]
        n_blocks = len(encoder.blocks)
    elif encoder_id == "vit-small":
        encoder = timm.create_model(
            "vit_small_patch16_224.augreg_in21k", pretrained=True, num_classes=0
        )
        amp_dtype = torch.float16
        embed_dim = getattr(encoder, "embed_dim")
        patch_size = 16
        pixel_mean = encoder.default_cfg["mean"]
        pixel_std = encoder.default_cfg["std"]
        n_blocks = len(encoder.blocks)

    else:
        raise ValueError(f"unknown encoder_id {encoder_id}")

    return encoder, {
        "amp_dtype": amp_dtype,
        "embed_dim": embed_dim,
        "patch_size": patch_size,
        "pixel_mean": pixel_mean,
        "pixel_std": pixel_std,
        "n_blocks": n_blocks,
    }
