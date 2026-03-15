import math
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from safetensors.torch import load_file as safe_load
from transformers import (
    WEIGHTS_NAME,
    ModernBertConfig,
    ModernBertModel,
    PreTrainedModel,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
LCR_ROOT = REPO_ROOT / "Legal Case Retrieval"
if str(LCR_ROOT) not in sys.path:
    sys.path.insert(0, str(LCR_ROOT))


class ContrastiveConfig(ModernBertConfig):
    model_type = "modernbert"

    def __init__(
        self,
        temperature_init: float = 0.55555,
        temperature_min: float = 1e-3,
        temperature_max: float = 2.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.temperature_init = float(temperature_init)
        self.temperature_min = float(temperature_min)
        self.temperature_max = float(temperature_max)


class ModernBERTContrastive(PreTrainedModel):
    config_class = ContrastiveConfig
    base_model_prefix = "encoder."

    @property
    def temperature(self) -> torch.Tensor:
        temp = torch.exp(self.log_temperature)
        return temp.clamp(self.temperature_min, self.temperature_max)

    def __init__(self, config: ContrastiveConfig):
        super().__init__(config)
        self.encoder: Optional[ModernBertModel] = None

        hidden_dim = config.hidden_size
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.temperature_min = getattr(config, "temperature_min", 1e-3)
        self.temperature_max = getattr(config, "temperature_max", 2.0)
        init_temp = getattr(config, "temperature_init", 0.55555)
        self.log_temperature = nn.Parameter(
            torch.tensor(math.log(float(init_temp)), dtype=torch.float32)
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args,
        encoder_model_name_or_path: Optional[str] = None,
        encoder_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        hf_loading_args = [
            "device_map",
            "torch_dtype",
            "attn_implementation",
            "trust_remote_code",
        ]
        if encoder_kwargs is None:
            encoder_kwargs = {}
        for key in hf_loading_args:
            if key in kwargs:
                encoder_kwargs[key] = kwargs.pop(key)

        config = kwargs.pop("config", None)
        config_kwargs = dict(kwargs)
        last_error = None
        if config is None:
            for cfg_path in (pretrained_model_name_or_path, encoder_model_name_or_path):
                if cfg_path is None:
                    continue
                try:
                    config = cls.config_class.from_pretrained(cfg_path, **config_kwargs)
                    break
                except Exception as err:  # pragma: no cover
                    last_error = err
                    continue
            if config is None:
                raise last_error if last_error is not None else ValueError(
                    "Failed to load ModernBERT config."
                )

        model = cls(config)

        safetensors_path = os.path.join(pretrained_model_name_or_path, "model.safetensors")
        bin_path = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
        if os.path.exists(safetensors_path):
            state_dict = safe_load(safetensors_path, device="cpu")
        elif os.path.exists(bin_path):
            state_dict = torch.load(bin_path, map_location="cpu")
        else:
            raise FileNotFoundError(
                f"No model.safetensors or {WEIGHTS_NAME} under {pretrained_model_name_or_path}"
            )

        prefix = cls.base_model_prefix
        encoder_state_dict = {
            key[len(prefix) :]: value
            for key, value in state_dict.items()
            if key.startswith(prefix)
        }

        model.encoder, loading_info = ModernBertModel.from_pretrained(
            pretrained_model_name_or_path=None,
            config=config,
            state_dict=encoder_state_dict,
            output_loading_info=True,
            **encoder_kwargs,
        )
        try:
            model.encoder.config.use_cache = False
            model.encoder.enable_input_require_grads()
            model.encoder.gradient_checkpointing_enable()
        except Exception:
            pass

        print("encoder missing keys:")
        print(loading_info.get("missing_keys", []))
        print("encoder unexpected keys:")
        print(loading_info.get("unexpected_keys", []))
        print("encoder error messages:")
        print(loading_info.get("error_msgs", []))

        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=True)
        if missing_keys:
            print("Missing keys:")
            for key in missing_keys:
                print("  ", key)
        if unexpected_keys:
            print("Unexpected keys:")
            for key in unexpected_keys:
                print("  ", key)

        print(f"[from_pretrained] log_temperature = {model.log_temperature.item():.6f}")
        print(f"[from_pretrained] temperature     = {model.temperature.item():.6f}")
        return model

    def encode(self, input_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        output = self.encoder(**input_batch)
        cls_hidden = output.last_hidden_state[:, 0, :]
        proj_vec = self.projector(cls_hidden)
        proj_vec = torch.nn.functional.normalize(proj_vec, p=2, dim=-1)
        return proj_vec

    def encode_in_chunks(
        self, negatives_input: Dict[str, torch.Tensor], chunk_size: int
    ) -> torch.Tensor:
        ids = negatives_input["input_ids"]
        attn = negatives_input["attention_mask"]
        all_vecs = []
        for start in range(0, ids.size(0), chunk_size):
            end = start + chunk_size
            out = self.encoder(input_ids=ids[start:end], attention_mask=attn[start:end])
            cls = out.last_hidden_state[:, 0, :]
            proj = self.projector(cls)
            all_vecs.append(torch.nn.functional.normalize(proj, p=2, dim=-1))
        return torch.cat(all_vecs, dim=0)

    def forward(
        self,
        anchor_input: Dict[str, torch.Tensor],
        positive_input: Dict[str, torch.Tensor],
        negative_input: Dict[str, torch.Tensor],
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        anchor_vec = self.encode(anchor_input)
        positive_vec = self.encode(positive_input)
        negative_vec = self.encode(negative_input)

        batch_size = anchor_vec.size(0)
        neg_count = negative_vec.size(0) // batch_size
        negative_vec = negative_vec.view(batch_size, neg_count, -1)

        pos_sim = torch.cosine_similarity(anchor_vec, positive_vec, dim=-1).unsqueeze(1)
        neg_sim = torch.cosine_similarity(anchor_vec.unsqueeze(1), negative_vec, dim=-1)
        logits = torch.cat([pos_sim, neg_sim], dim=1) / self.temperature

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return {"loss": loss, "logits": logits}
