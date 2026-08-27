"""Stage config expansion, source spec parsing, eraser fitting, and name generation."""
from __future__ import annotations

from itertools import product
from typing import Any, Mapping, Sequence

from histovfmgeom.concept_erasure.multi_paired_delta_erasers import (
    DeltaSourceSpec,
    PairedDeltaFitter,
)


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def parse_ranks(value: Any) -> list[int | None]:
    if value is None:
        return [None]
    if isinstance(value, int):
        return [int(value)]
    if isinstance(value, str):
        out: list[int | None] = []
        for item in value.split(","):
            item = item.strip()
            if item.lower() in {"none", "null", "full", "untruncated"}:
                out.append(None)
            elif item:
                out.append(int(item))
        return out
    return [None if r is None else int(r) for r in value]


def safe_name(value: object) -> str:
    return (
        str(value)
        .replace("/", "-").replace("\\", "-")
        .replace(" ", "_").replace(".", "p").replace(":", "-")
    )


def expand_stage_config(stage: Mapping[str, Any]) -> list[dict[str, Any]]:
    cfg = dict(stage)
    method = str(cfg["method"])
    expanded: list[dict[str, Any]] = []

    if method == "paired_delta_pca":
        ranks = [r for r in parse_ranks(cfg.get("ranks", cfg.get("rank", [1, 8, 16, 32, 64]))) if r is not None]
        for rank, whitening, shrink_A in product(ranks, as_list(cfg.get("whitening", True)), as_list(cfg.get("shrink_A", True))):
            expanded.append({**cfg, "method": method, "rank": int(rank), "whitening": bool(whitening), "shrink_A": bool(shrink_A)})

    elif method == "soft_delta_projection":
        ranks = parse_ranks(cfg.get("ranks", cfg.get("rank", [None, 64])))
        lams = [float(v) for v in as_list(cfg.get("lambdas", cfg.get("lambda", [1000.0])))]
        for rank, lam, shrink_A in product(ranks, lams, as_list(cfg.get("shrink_A", True))):
            expanded.append({**cfg, "method": method, "rank": rank, "lam": float(lam), "shrink_A": bool(shrink_A)})

    elif method == "hard_delta_projection":
        ranks = parse_ranks(cfg.get("ranks", cfg.get("rank", [None, 64])))
        for rank, shrink_A in product(ranks, as_list(cfg.get("shrink_A", True))):
            expanded.append({**cfg, "method": method, "rank": rank, "shrink_A": bool(shrink_A)})

    else:
        raise ValueError(f"Unsupported eraser method: {method!r}")

    if not expanded:
        raise ValueError(f"Stage {cfg.get('name')!r} produced no configurations.")
    return expanded


def expand_stage_grid(stages: Sequence[Mapping[str, Any]]) -> list[list[dict[str, Any]]]:
    return [expand_stage_config(stage) for stage in stages]


def stage_source_specs(stage_cfg: Mapping[str, Any]) -> list[DeltaSourceSpec]:
    if "components" in stage_cfg:
        components = stage_cfg["components"]
    elif "source" in stage_cfg:
        components = [{"source": stage_cfg["source"], "weight": stage_cfg.get("weight", 1.0)}]
    else:
        raise ValueError(f"Stage {stage_cfg.get('name')!r} must define 'source' or 'components'.")

    default_moment = str(stage_cfg.get("delta_moment", "second_moment"))
    default_shrinkage = bool(stage_cfg.get("shrink_B", False))
    default_normalization = str(stage_cfg.get("moment_normalization", "trace"))

    return [
        DeltaSourceSpec(
            name=str(c["source"]),
            weight=float(c.get("weight", 1.0)),
            moment=c.get("moment") or default_moment,
            shrinkage=bool(c["shrinkage"]) if c.get("shrinkage") is not None else default_shrinkage,
            normalization=c.get("normalization") or default_normalization,
        )
        for c in components
    ]


def fit_eraser(
    *,
    fitter: PairedDeltaFitter,
    stage_cfg: Mapping[str, Any],
    source_specs: Sequence[DeltaSourceSpec],
) -> Any:
    method = str(stage_cfg["method"])
    common = {
        "affine": bool(stage_cfg.get("affine", True)),
        "delta_sources": source_specs,
        "normalize_source_weights": bool(stage_cfg.get("normalize_source_weights", True)),
        "shrink_A": bool(stage_cfg.get("shrink_A", True)),
        "ridge": float(stage_cfg.get("ridge", 1e-4)),
        "svd_tol": float(stage_cfg.get("svd_tol", 1e-7)),
    }
    if method == "paired_delta_pca":
        return fitter.make_pca_eraser(rank=int(stage_cfg["rank"]), whitening=bool(stage_cfg.get("whitening", True)), **common)
    if method == "soft_delta_projection":
        return fitter.make_soft_eraser(lam=float(stage_cfg["lam"]), rank=stage_cfg.get("rank"),
                                       joint_normalization=str(stage_cfg.get("joint_normalization", "none")), **common)
    if method == "hard_delta_projection":
        return fitter.make_hard_eraser(rank=stage_cfg.get("rank"),
                                       joint_normalization=str(stage_cfg.get("joint_normalization", "none")), **common)
    raise ValueError(f"Unsupported eraser method: {method!r}")


def stage_name(stage_cfg: Mapping[str, Any], fold_idx: int) -> str:
    method = str(stage_cfg["method"])
    rank = stage_cfg.get("rank")
    parts = [safe_name(stage_cfg["name"]), method, f"fold{fold_idx}", "full" if rank is None else f"rank{rank}"]
    if method == "paired_delta_pca":
        parts.append(f"white{int(bool(stage_cfg.get('whitening', True)))}")
    elif method == "soft_delta_projection":
        parts.append(f"lambda{float(stage_cfg['lam']):g}")
    elif method == "hard_delta_projection":
        parts.append(f"jointnorm{safe_name(str(stage_cfg.get('joint_normalization', 'none')))}")
    parts += [f"shrinkA{int(bool(stage_cfg.get('shrink_A', True)))}", f"ridge{float(stage_cfg.get('ridge', 1e-4)):g}"]
    return safe_name("_".join(parts))


def chain_name(stage_cfgs: Sequence[Mapping[str, Any]], fold_idx: int, combo_idx: int) -> str:
    parts = [f"fold{fold_idx}", f"combo{combo_idx}"]
    for stage in stage_cfgs:
        method = str(stage["method"])
        rank = stage.get("rank")
        label = "full" if rank is None else f"r{rank}"
        if method == "paired_delta_pca":
            extra = f"w{int(bool(stage.get('whitening', True)))}"
        elif method == "soft_delta_projection":
            extra = f"l{float(stage['lam']):g}"
        elif method == "hard_delta_projection":
            extra = "hard"
        else:
            raise ValueError(f"Unsupported eraser method: {method!r}")
        parts.append(f"{safe_name(stage['name'])}-{label}-{extra}")
    return safe_name("__".join(parts))
