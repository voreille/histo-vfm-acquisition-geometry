from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


@torch.no_grad()
def apply_eraser(
    eraser: Any,
    values: np.ndarray,
    *,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
) -> np.ndarray:
    eraser = eraser.to(device=device, dtype=dtype)
    out: list[np.ndarray] = []
    for i in range(0, len(values), batch_size):
        x = torch.as_tensor(values[i : i + batch_size], device=device, dtype=dtype)
        out.append(eraser(x).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(out, axis=0) if out else np.empty_like(values, dtype=np.float32)


@torch.no_grad()
def apply_delta_transform(
    eraser: Any,
    deltas: np.ndarray,
    *,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
) -> np.ndarray:
    eraser = eraser.to(device=device, dtype=dtype)
    out: list[np.ndarray] = []
    for i in range(0, len(deltas), batch_size):
        x = torch.as_tensor(deltas[i : i + batch_size], device=device, dtype=dtype)
        out.append(eraser.transform_delta(x).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(out, axis=0) if out else np.empty_like(deltas, dtype=np.float32)


def save_eraser_npz(path: Path, eraser: Any, *, metadata: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {"metadata_json": np.asarray(json.dumps(dict(metadata)))}
    for name in ("P", "proj_left", "proj_right", "bias", "eigenvalues"):
        v = getattr(eraser, name, None)
        if v is not None:
            arrays[name] = v.detach().cpu().numpy().astype(np.float32)
    np.savez_compressed(path, **arrays)


def save_chained_eraser_npz(
    path: Path,
    erasers: Sequence[Any],
    *,
    component_paths: Sequence[Path],
    metadata: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "type": "chained_linear_eraser",
        "n_components": len(erasers),
        "component_paths": [str(p) for p in component_paths],
        **dict(metadata),
    }
    arrays: dict[str, Any] = {"metadata_json": np.asarray(json.dumps(payload))}
    for i, eraser in enumerate(erasers):
        for name in ("P", "proj_left", "proj_right", "bias", "eigenvalues"):
            v = getattr(eraser, name, None)
            if v is not None:
                arrays[f"component_{i}_{name}"] = v.detach().cpu().numpy().astype(np.float32)
    np.savez_compressed(path, **arrays)
