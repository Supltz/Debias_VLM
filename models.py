from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from PIL import Image
import open_clip
from transformers import (
    BlipForImageTextRetrieval,
    BlipProcessor,
    Blip2ForImageTextRetrieval,
    Blip2Processor,
    CLIPModel,
    CLIPProcessor,
)


MODEL_CONFIGS: Dict[str, Dict[str, str]] = {
    "clip_vit_l14": {"family": "clip", "hf_id": "openai/clip-vit-large-patch14"},
    "openclip_vit_h14_sd21": {"family": "open_clip", "model_name": "ViT-H-14", "pretrained": "laion2b_s32b_b79k"},
    "clip_rn50": {"family": "open_clip", "model_name": "RN50", "pretrained": "openai"},
    "blip": {"family": "blip_retrieval", "hf_id": "Salesforce/blip-itm-base-coco"},
    # BLIP-2 is heavy; this default may require substantial VRAM/RAM.
    "blip2": {"family": "blip2_retrieval", "hf_id": "Salesforce/blip2-itm-vit-g-coco"},
}


@dataclass
class LoadedModel:
    name: str
    family: str
    model_id: str
    model: torch.nn.Module
    processor: object
    tokenizer: object | None
    device: torch.device
    amp_enabled: bool

    def _as_feature_tensor(self, features: object) -> torch.Tensor:
        if isinstance(features, torch.Tensor):
            if features.ndim == 3:
                return features.mean(dim=1)
            if features.ndim > 3:
                return features.flatten(start_dim=1)
            return features
        for attr in ("text_embeds", "image_embeds", "pooler_output", "last_hidden_state"):
            if hasattr(features, attr):
                val = getattr(features, attr)
                if isinstance(val, torch.Tensor):
                    # If only token-wise embeddings are available, use [CLS]-like first token.
                    if attr == "last_hidden_state" and val.ndim == 3:
                        return val[:, 0, :]
                    return val
        raise TypeError(f"Unsupported feature output type: {type(features)}")

    @torch.inference_mode()
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        if self.family == "clip":
            inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp_enabled):
                features = self.model.get_text_features(**inputs)
        elif self.family == "blip_retrieval":
            inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp_enabled):
                txt_out = self.model.text_encoder(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    return_dict=True,
                )
                features = self.model.text_proj(txt_out.last_hidden_state[:, 0, :])
        elif self.family == "blip2_retrieval":
            inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            img_size = int(self.model.config.vision_config.image_size)
            pixel_values = torch.zeros((len(texts), 3, img_size, img_size), device=self.device, dtype=torch.float32)
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp_enabled):
                out = self.model(
                    pixel_values=pixel_values,
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    use_image_text_matching_head=False,
                    return_dict=True,
                )
                features = out.text_embeds
        elif self.family == "open_clip":
            if self.tokenizer is None:
                raise RuntimeError("OpenCLIP tokenizer is missing")
            tokens = self.tokenizer(texts).to(self.device)
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp_enabled):
                features = self.model.encode_text(tokens)
        else:
            raise ValueError(f"Unsupported model family: {self.family}")

        features = self._as_feature_tensor(features)
        return torch.nn.functional.normalize(features, dim=-1)

    @torch.inference_mode()
    def encode_images(self, images: List[Image.Image]) -> torch.Tensor:
        if self.family == "clip":
            inputs = self.processor(images=images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp_enabled):
                features = self.model.get_image_features(**inputs)
        elif self.family == "blip_retrieval":
            inputs = self.processor(images=images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp_enabled):
                vision_out = self.model.vision_model(pixel_values=inputs["pixel_values"], return_dict=True)
                features = self.model.vision_proj(vision_out.last_hidden_state[:, 0, :])
        elif self.family == "blip2_retrieval":
            inputs = self.processor(images=images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            text_inputs = self.processor(
                text=[""] * len(images),
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp_enabled):
                out = self.model(
                    pixel_values=inputs["pixel_values"],
                    input_ids=text_inputs["input_ids"],
                    attention_mask=text_inputs["attention_mask"],
                    use_image_text_matching_head=False,
                    return_dict=True,
                )
                features = out.image_embeds.mean(dim=1)
        elif self.family == "open_clip":
            tensors = [self.processor(img) for img in images]
            pixel_values = torch.stack(tensors, dim=0).to(self.device)
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp_enabled):
                features = self.model.encode_image(pixel_values)
        else:
            raise ValueError(f"Unsupported model family: {self.family}")

        features = self._as_feature_tensor(features)
        return torch.nn.functional.normalize(features, dim=-1)


def _default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def available_models() -> Dict[str, Dict[str, str]]:
    return MODEL_CONFIGS


def _load_processor(processor_cls, model_id: str):
    # Explicit use_fast choice avoids transformers advisory warning about future default changes.
    try:
        return processor_cls.from_pretrained(model_id, use_fast=False)
    except TypeError:
        return processor_cls.from_pretrained(model_id)


def _load_hf_model_with_dtype(model_cls, model_id: str, model_dtype: torch.dtype):
    # Prefer dtype (new API), fallback to torch_dtype for older versions.
    try:
        return model_cls.from_pretrained(model_id, dtype=model_dtype)
    except TypeError:
        return model_cls.from_pretrained(model_id, torch_dtype=model_dtype)


def load_model(model_name: str, device: str | None = None) -> LoadedModel:
    if model_name not in MODEL_CONFIGS:
        valid = ", ".join(sorted(MODEL_CONFIGS.keys()))
        raise ValueError(f"Unknown model_name '{model_name}'. Valid options: {valid}")

    cfg = MODEL_CONFIGS[model_name]
    family = cfg["family"]
    model_id = cfg.get("hf_id", f"{cfg.get('model_name')}:{cfg.get('pretrained')}")
    target_device = torch.device(device) if device else _default_device()
    amp_enabled = target_device.type == "cuda"
    model_dtype = torch.float16 if amp_enabled else torch.float32

    if family == "clip":
        processor = _load_processor(CLIPProcessor, model_id)
        model = _load_hf_model_with_dtype(CLIPModel, model_id, model_dtype)
        tokenizer = None
    elif family == "blip_retrieval":
        processor = _load_processor(BlipProcessor, model_id)
        model = _load_hf_model_with_dtype(BlipForImageTextRetrieval, model_id, model_dtype)
        tokenizer = None
    elif family == "blip2_retrieval":
        processor = _load_processor(Blip2Processor, model_id)
        model = _load_hf_model_with_dtype(Blip2ForImageTextRetrieval, model_id, model_dtype)
        tokenizer = None
    elif family == "open_clip":
        model_name = cfg["model_name"]
        pretrained = cfg["pretrained"]
        # Match OpenAI pretrained tags that expect QuickGELU to avoid runtime warning/noise.
        force_quick_gelu = pretrained.lower() == "openai"
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            force_quick_gelu=force_quick_gelu,
        )
        processor = preprocess
        tokenizer = open_clip.get_tokenizer(model_name)
    else:
        raise ValueError(f"Unsupported model family: {family}")

    model.eval().to(target_device)

    return LoadedModel(
        name=model_name,
        family=family,
        model_id=model_id,
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        device=target_device,
        amp_enabled=amp_enabled,
    )
