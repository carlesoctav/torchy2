#!/usr/bin/env python3
"""SST training implementation."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import torch
from transformers import TrainerCallback



TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass
class ProtectedWeight:
    name: str
    parameter: torch.nn.Parameter
    reference: torch.Tensor
    left: torch.Tensor
    right: torch.Tensor


def _is_target(name: str, parameter: torch.nn.Parameter) -> bool:
    return parameter.ndim == 2 and any(
        name.endswith(f".{module}.weight") for module in TARGET_MODULES
    )


def _svd_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return torch.device("cuda", local_rank)


def build_protected_weights(model, rank_percent: float) -> list[ProtectedWeight]:
    if not 0 < rank_percent < 100:
        raise ValueError("rank_percent must be in (0, 100)")
    device = _svd_device()
    protected = []
    for name, parameter in model.named_parameters():
        if not _is_target(name, parameter):
            continue
        weight = parameter.detach().to(device=device, dtype=torch.float32)
        minimum = min(weight.shape)
        rank = min(max(1, math.floor(minimum * rank_percent / 100.0)), minimum - 1)
        left, _, vh = torch.linalg.svd(weight, full_matrices=False)
        protected.append(
            ProtectedWeight(
                name=name,
                parameter=parameter,
                reference=weight.cpu(),
                left=left[:, :rank].contiguous().cpu(),
                right=vh[:rank].T.contiguous().cpu(),
            )
        )
        del weight, left, vh
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print(f"[sst] prepared {len(protected)} protected matrices", flush=True)
    if not protected:
        raise RuntimeError("SST found no target weights")
    return protected


class SSTProjectionCallback(TrainerCallback):
    """Constrain the cumulative AdamW update after every optimizer step."""

    def __init__(self, protected: list[ProtectedWeight]):
        self.protected = protected

    @staticmethod
    def _project(entry: ProtectedWeight) -> None:
        parameter = entry.parameter
        master = None
        try:
            from deepspeed.utils import safe_get_full_fp32_param

            master = safe_get_full_fp32_param(parameter)
        except (ImportError, RuntimeError, AttributeError):
            master = None

        current = master if master is not None else parameter.data.float()
        device = current.device
        reference = entry.reference.to(device=device, dtype=torch.float32, non_blocking=True)
        left = entry.left.to(device=device, dtype=torch.float32, non_blocking=True)
        right = entry.right.to(device=device, dtype=torch.float32, non_blocking=True)
        delta = current.float() - reference
        delta = delta - left @ (left.T @ delta)
        delta = delta - (delta @ right) @ right.T
        projected = reference + delta

        if master is not None:
            try:
                from deepspeed.utils import safe_set_full_fp32_param

                safe_set_full_fp32_param(parameter, projected)
            except (ImportError, RuntimeError, AttributeError):
                parameter.data.copy_(projected.to(parameter.dtype))
        parameter.data.copy_(projected.to(parameter.dtype))

    def on_optimizer_step(self, args, state, control, **kwargs):
        with torch.no_grad():
            for entry in self.protected:
                self._project(entry)

