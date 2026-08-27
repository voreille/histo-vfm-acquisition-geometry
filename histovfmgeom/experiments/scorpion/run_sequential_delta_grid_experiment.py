from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold

from histovfmgeom.concept_erasure.multi_paired_delta_erasers import (
    DeltaSourceSpec,
    PairedDeltaFitter,
)
from histovfmgeom.deltas.domain_deltas import build_domain_deltas
from histovfmgeom.evaluation.erasure_metrics import (
    covariance_trace_np,
    delta_residual_metrics,
    feature_variance_metrics,
    joint_moment_diagnostics,
    probe_excess_ratio,
)
from histovfmgeom.evaluation.probe import evaluate_probe_train_test
from histovfmgeom.projections.linear import delta_change_summary, feature_change_summary

from ._eraser_io import (
    apply_delta_transform,
    apply_eraser,
    save_chained_eraser_npz,
    save_eraser_npz,
)
from ._stage_config import (
    chain_name as make_chain_name,
)
from ._stage_config import (
    expand_stage_grid,
    fit_eraser,
    safe_name,
    stage_source_specs,
)
from ._stage_config import (
    stage_name as make_stage_name,
)
from ._stain_probe import StainProbeData, build_stain_probe

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeltaSourceData:
    name: str
    kind: str
    config: dict[str, Any]
    train: np.ndarray
    test: np.ndarray


def _to_tensor(
    values: np.ndarray, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    return torch.as_tensor(values, device=device, dtype=dtype)


def _atomic_write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, path)


def _atomic_write_json(data: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _stain_rows(
    table_metadata: pd.DataFrame,
    original_indices: np.ndarray,
    source_row_index_col: str = "source_row_index",
) -> np.ndarray:
    mask = (
        table_metadata[source_row_index_col]
        .astype(np.int64)
        .isin(np.asarray(original_indices, dtype=np.int64))
    )
    return np.flatnonzero(mask.to_numpy()).astype(np.int64)


def _build_delta_sources(
    *,
    features: np.ndarray,
    metadata: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    scanner_col: str,
    scanner_configurations: Sequence[Mapping[str, Any]],
    stain_features: np.ndarray | None,
    stain_metadata: pd.DataFrame | None,
    stain_configurations: Sequence[Mapping[str, Any]],
    stain_source_row_index_col: str,
    seed: int,
    fold_idx: int,
) -> dict[str, DeltaSourceData]:
    sources: dict[str, DeltaSourceData] = {}

    for raw_cfg in scanner_configurations:
        cfg = dict(raw_cfg)
        name = f"scanner.{cfg['name']}"
        kw = dict(
            features=features,
            metadata=metadata,
            domain_col=str(cfg.get("domain_col", scanner_col)),
            group_col=str(cfg["group_col"]),
            delta_mode=cfg["delta_mode"],
            pair_col=cfg.get("pair_col"),
            sign_mode=cfg.get("sign_mode", "one"),
        )
        sources[name] = DeltaSourceData(
            name=name,
            kind="scanner",
            config=cfg,
            train=build_domain_deltas(
                **kw,
                row_indices=train_idx,
                max_deltas=cfg.get("max_deltas_per_fold"),
                seed=seed + fold_idx,
            ),
            test=build_domain_deltas(
                **kw,
                row_indices=test_idx,
                max_deltas=cfg.get("max_test_deltas"),
                seed=seed + 10_000 + fold_idx,
            ),
        )

    if stain_configurations:
        train_stain = _stain_rows(stain_metadata, train_idx, stain_source_row_index_col)
        test_stain = _stain_rows(stain_metadata, test_idx, stain_source_row_index_col)
        for raw_cfg in stain_configurations:
            cfg = dict(raw_cfg)
            name = f"stain.{cfg['name']}"
            kw = dict(
                features=stain_features,
                metadata=stain_metadata,
                domain_col=str(cfg.get("domain_col", "target_id")),
                group_col=str(cfg["group_col"]),
                delta_mode=cfg["delta_mode"],
                pair_col=cfg.get("pair_col"),
                sign_mode=cfg.get("sign_mode", "one"),
            )
            sources[name] = DeltaSourceData(
                name=name,
                kind="stain",
                config=cfg,
                train=build_domain_deltas(
                    **kw,
                    row_indices=train_stain,
                    max_deltas=cfg.get("max_deltas_per_fold"),
                    seed=seed + fold_idx,
                ),
                test=build_domain_deltas(
                    **kw,
                    row_indices=test_stain,
                    max_deltas=cfg.get("max_test_deltas"),
                    seed=seed + 10_000 + fold_idx,
                ),
            )

    return sources


def run_sequential_delta_grid_experiment(
    *,
    features: np.ndarray,
    metadata: pd.DataFrame,
    output_dir: str | Path,
    scanner_col: str,
    cv_group_col: str,
    scanner_delta_configurations: Sequence[Mapping[str, Any]],
    stain_delta_configurations: Sequence[Mapping[str, Any]],
    sequential_stages: Sequence[Mapping[str, Any]],
    stain_features: np.ndarray | None = None,
    stain_metadata: pd.DataFrame | None = None,
    stain_source_row_index_col: str = "source_row_index",
    n_splits: int = 5,
    seed: int = 0,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float32,
    apply_batch_size: int = 8192,
    probe_type: str = "logistic",
    stain_probe_enabled: bool = True,
    stain_probe_label_col: str = "target_id",
    stain_probe_max_examples_per_split: int | None = None,
    run_only_one_fold: bool = False,
    diagnostics_config: Mapping[str, Any] | None = None,
    save_erasers: bool = True,
    evaluate_intermediate_stages: bool = True,
    checkpoint_every: int = 1,
    reuse_soft_families: bool = True,
) -> dict[str, Any]:
    dcfg = dict(diagnostics_config or {})
    source_moment_diag = bool(dcfg.get("source_moments", True))
    spectral_diag = bool(dcfg.get("spectral", False))
    spectral_top_k = int(dcfg.get("spectral_top_k", 32))

    stage_options = expand_stage_grid(sequential_stages)
    n_combos = 1
    for opts in stage_options:
        n_combos *= len(opts)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    eraser_dir = output_dir / "fold_erasers"
    if save_erasers:
        eraser_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA unavailable; falling back to CPU.")
        device = torch.device("cpu")

    scanner_values = metadata[scanner_col].astype(str).to_numpy()
    cv_groups = metadata[cv_group_col].astype(str).to_numpy()
    n_splits = min(int(n_splits), len(np.unique(cv_groups)))

    cv = GroupKFold(n_splits=n_splits)
    chain_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []
    moment_rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {
        "experiment_type": "sequential_delta_grid_table",
        "n_samples": int(len(features)),
        "embedding_dim": int(features.shape[1]),
        "n_splits": n_splits,
        "n_stage_combinations": int(n_combos),
        "folds": [],
    }

    def _flush() -> None:
        if chain_rows:
            _atomic_write_csv(chain_rows, output_dir / "chain_scores.csv")
        if stage_rows:
            _atomic_write_csv(stage_rows, output_dir / "stage_scores.csv")
        if delta_rows:
            _atomic_write_csv(delta_rows, output_dir / "delta_scores.csv")
        if moment_rows:
            _atomic_write_csv(moment_rows, output_dir / "moment_diagnostics.csv")

    def _cfg_key(stage_cfg: Mapping[str, Any], specs: Sequence[DeltaSourceSpec]) -> str:
        return json.dumps(
            {
                "method": str(stage_cfg["method"]),
                "rank": stage_cfg.get("rank"),
                "affine": bool(stage_cfg.get("affine", True)),
                "normalize_source_weights": bool(
                    stage_cfg.get("normalize_source_weights", True)
                ),
                "shrink_A": bool(stage_cfg.get("shrink_A", True)),
                "ridge": float(stage_cfg.get("ridge", 1e-4)),
                "svd_tol": float(stage_cfg.get("svd_tol", 1e-7)),
                "joint_normalization": str(
                    stage_cfg.get("joint_normalization", "none")
                ),
                "source_specs": [asdict(s) for s in specs],
            },
            sort_keys=True,
        )

    for fold_idx, (train_idx, test_idx) in enumerate(
        cv.split(features, scanner_values, groups=cv_groups)
    ):
        if run_only_one_fold and fold_idx > 0:
            break

        logger.info("Starting fold %d/%d", fold_idx + 1, n_splits)
        x_train_raw = features[train_idx].astype(np.float32, copy=False)
        x_test_raw = features[test_idx].astype(np.float32, copy=False)
        ref_trace_A = covariance_trace_np(x_train_raw)
        scanner_train = scanner_values[train_idx]
        scanner_test = scanner_values[test_idx]

        raw_scanner_probe = evaluate_probe_train_test(
            x_train_raw, x_test_raw, scanner_train, scanner_test, probe_type=probe_type
        )

        stain_probe_data: StainProbeData | None = None
        raw_stain_probe = None
        if (
            stain_probe_enabled
            and stain_features is not None
            and stain_metadata is not None
        ):
            train_stain = _stain_rows(
                stain_metadata, train_idx, stain_source_row_index_col
            )
            test_stain = _stain_rows(
                stain_metadata, test_idx, stain_source_row_index_col
            )
            stain_probe_data = build_stain_probe(
                stain_features=stain_features,
                stain_metadata=stain_metadata,
                train_rows=train_stain,
                test_rows=test_stain,
                source_row_index_col=stain_source_row_index_col,
                label_col=stain_probe_label_col,
                max_examples_per_split=stain_probe_max_examples_per_split,
                seed=seed + fold_idx,
            )
            if stain_probe_data is not None:
                raw_stain_probe = evaluate_probe_train_test(
                    stain_probe_data.x_train,
                    stain_probe_data.x_test,
                    stain_probe_data.y_train,
                    stain_probe_data.y_test,
                    probe_type=probe_type,
                )

        sources = _build_delta_sources(
            features=features,
            metadata=metadata,
            train_idx=train_idx,
            test_idx=test_idx,
            scanner_col=scanner_col,
            scanner_configurations=scanner_delta_configurations,
            stain_features=stain_features,
            stain_metadata=stain_metadata,
            stain_configurations=stain_delta_configurations,
            stain_source_row_index_col=stain_source_row_index_col,
            seed=seed,
            fold_idx=fold_idx,
        )

        fold_diag: dict[str, Any] = {
            "fold": fold_idx,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "raw_scanner_balanced_accuracy": raw_scanner_probe.balanced_accuracy,
            "stage_combinations": [],
        }
        combo_counter = 0

        def _evaluate_leaf(
            *,
            stage_cfgs: list[dict[str, Any]],
            x_train: np.ndarray,
            x_test: np.ndarray,
            src_test: dict[str, np.ndarray],
            stain_x_train: np.ndarray | None,
            stain_x_test: np.ndarray | None,
            erasers: list[Any],
            comp_paths: list[Path],
            acc_stage_rows: list[dict],
            acc_moment_rows: list[dict],
            acc_stage_diags: list[dict],
            last_scanner_probe: Any | None,
            last_stain_probe: Any | None,
        ) -> None:
            nonlocal combo_counter
            combo_idx = combo_counter
            combo_counter += 1
            logger.info("fold=%d combo=%d/%d", fold_idx, combo_idx + 1, n_combos)

            scanner_probe = last_scanner_probe or evaluate_probe_train_test(
                x_train, x_test, scanner_train, scanner_test, probe_type=probe_type
            )
            stain_probe = last_stain_probe
            if (
                stain_probe is None
                and raw_stain_probe is not None
                and stain_x_train is not None
            ):
                assert stain_probe_data is not None
                stain_probe = evaluate_probe_train_test(
                    stain_x_train,
                    stain_x_test,
                    stain_probe_data.y_train,
                    stain_probe_data.y_test,
                    probe_type=probe_type,
                )

            chain_path: Path | None = None
            if save_erasers:
                chain_path = eraser_dir / (
                    make_chain_name(stage_cfgs, fold_idx, combo_idx) + ".npz"
                )
                save_chained_eraser_npz(
                    chain_path,
                    erasers,
                    component_paths=comp_paths,
                    metadata={
                        "fold": fold_idx,
                        "combo": combo_idx,
                        "stage_configs": [dict(s) for s in stage_cfgs],
                    },
                )

            feat_change = feature_change_summary(raw=x_test_raw, projected=x_test)
            feat_variance = feature_variance_metrics(
                raw=x_test_raw,
                projected=x_test,
                reference_trace_A=ref_trace_A,
                projected_reference=x_train,
            )

            chain_rows.append(
                {
                    "fold": fold_idx,
                    "combo": combo_idx,
                    "stage_names": json.dumps([s["name"] for s in stage_cfgs]),
                    "stage_methods": json.dumps([s["method"] for s in stage_cfgs]),
                    "stage_ranks": json.dumps(
                        [
                            "full" if s.get("rank") is None else int(s["rank"])
                            for s in stage_cfgs
                        ]
                    ),
                    "stage_lambdas": json.dumps(
                        [
                            None if s.get("lam") is None else float(s["lam"])
                            for s in stage_cfgs
                        ]
                    ),
                    "stage_configs": json.dumps([dict(s) for s in stage_cfgs]),
                    "raw_score": raw_scanner_probe.balanced_accuracy,
                    "projected_score": scanner_probe.balanced_accuracy,
                    "raw_accuracy": raw_scanner_probe.accuracy,
                    "projected_accuracy": scanner_probe.accuracy,
                    "chance_balanced_accuracy": raw_scanner_probe.chance_balanced_accuracy,
                    "raw_stain_target_balanced_accuracy": np.nan
                    if raw_stain_probe is None
                    else raw_stain_probe.balanced_accuracy,
                    "projected_stain_target_balanced_accuracy": np.nan
                    if stain_probe is None
                    else stain_probe.balanced_accuracy,
                    "scanner_probe_excess_ratio": probe_excess_ratio(
                        raw_balanced_accuracy=raw_scanner_probe.balanced_accuracy,
                        projected_balanced_accuracy=scanner_probe.balanced_accuracy,
                        chance_balanced_accuracy=raw_scanner_probe.chance_balanced_accuracy,
                    ),
                    "stain_probe_excess_ratio": np.nan
                    if (raw_stain_probe is None or stain_probe is None)
                    else probe_excess_ratio(
                        raw_balanced_accuracy=raw_stain_probe.balanced_accuracy,
                        projected_balanced_accuracy=stain_probe.balanced_accuracy,
                        chance_balanced_accuracy=raw_stain_probe.chance_balanced_accuracy,
                    ),
                    **{
                        k: feat_change[k]
                        for k in (
                            "mean_l2_change",
                            "median_l2_change",
                            "mean_raw_norm",
                            "mean_relative_change",
                        )
                    },
                    **feat_variance,
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "chained_eraser_path": ""
                    if chain_path is None
                    else str(chain_path),
                }
            )

            for row in acc_stage_rows:
                stage_rows.append({**row, "combo": combo_idx})
            for row in acc_moment_rows:
                moment_rows.append({**row, "combo": combo_idx})

            proj_trace_A = covariance_trace_np(x_train)
            for src_name, src_data in sources.items():
                change = delta_change_summary(
                    raw_delta=src_data.test, projected_delta=src_test[src_name]
                )
                residual = delta_residual_metrics(
                    raw_delta=src_data.test,
                    projected_delta=src_test[src_name],
                    reference_trace_A=ref_trace_A,
                    projected_trace_A=proj_trace_A,
                )
                delta_rows.append(
                    {
                        "fold": fold_idx,
                        "combo": combo_idx,
                        "stage_names": json.dumps([s["name"] for s in stage_cfgs]),
                        "stage_configs": json.dumps([dict(s) for s in stage_cfgs]),
                        "evaluation_source": src_name,
                        "evaluation_source_kind": src_data.kind,
                        "n_delta_test": int(len(src_data.test)),
                        "chained_eraser_path": ""
                        if chain_path is None
                        else str(chain_path),
                        **change,
                        **residual,
                    }
                )

            fold_diag["stage_combinations"].append(
                {
                    "combo": combo_idx,
                    "stage_configs": [dict(s) for s in stage_cfgs],
                    "stage_diagnostics": acc_stage_diags,
                    "component_eraser_paths": [str(p) for p in comp_paths],
                    "chained_eraser_path": ""
                    if chain_path is None
                    else str(chain_path),
                    "final_scanner_balanced_accuracy": scanner_probe.balanced_accuracy,
                    "final_stain_target_balanced_accuracy": np.nan
                    if stain_probe is None
                    else stain_probe.balanced_accuracy,
                }
            )

            if checkpoint_every and combo_counter % checkpoint_every == 0:
                _flush()

        def _walk(
            *,
            stage_idx: int,
            x_train: np.ndarray,
            x_test: np.ndarray,
            src_train: dict[str, np.ndarray],
            src_test: dict[str, np.ndarray],
            stain_x_train: np.ndarray | None,
            stain_x_test: np.ndarray | None,
            cfg_prefix: list[dict],
            erasers: list[Any],
            comp_paths: list[Path],
            acc_stage_rows: list[dict],
            acc_moment_rows: list[dict],
            acc_stage_diags: list[dict],
            last_scanner_probe: Any | None,
            last_stain_probe: Any | None,
        ) -> None:
            if stage_idx == len(stage_options):
                _evaluate_leaf(
                    stage_cfgs=cfg_prefix,
                    x_train=x_train,
                    x_test=x_test,
                    src_test=src_test,
                    stain_x_train=stain_x_train,
                    stain_x_test=stain_x_test,
                    erasers=erasers,
                    comp_paths=comp_paths,
                    acc_stage_rows=acc_stage_rows,
                    acc_moment_rows=acc_moment_rows,
                    acc_stage_diags=acc_stage_diags,
                    last_scanner_probe=last_scanner_probe,
                    last_stain_probe=last_stain_probe,
                )
                return

            options = stage_options[stage_idx]
            option_specs = [stage_source_specs(cfg) for cfg in options]
            required = sorted({spec.name for specs in option_specs for spec in specs})

            fitter = PairedDeltaFitter(
                x_dim=features.shape[1], device=device, dtype=dtype
            )
            fitter.update_x(_to_tensor(x_train, device=device, dtype=dtype))
            for src_name in required:
                fitter.update_delta_source(
                    src_name,
                    _to_tensor(src_train[src_name], device=device, dtype=dtype),
                )

            soft_cache: dict[str, Any] = {}
            diag_cache: dict[str, tuple] = {}

            for stage_cfg, specs in zip(options, option_specs):
                key = _cfg_key(stage_cfg, specs)
                if key not in diag_cache:
                    jn = str(stage_cfg.get("joint_normalization", "none"))
                    src_diag = (
                        fitter.source_diagnostics(specs) if source_moment_diag else {}
                    )
                    mom_diag = (
                        joint_moment_diagnostics(
                            fitter=fitter,
                            source_specs=specs,
                            normalize_source_weights=bool(
                                stage_cfg.get("normalize_source_weights", True)
                            ),
                            joint_normalization=jn,
                            shrink_A=bool(stage_cfg.get("shrink_A", True)),
                            ridge=float(stage_cfg.get("ridge", 1e-4)),
                            svd_tol=float(stage_cfg.get("svd_tol", 1e-7)),
                            include_spectrum=spectral_diag,
                            top_k=spectral_top_k,
                        )
                        if source_moment_diag
                        else {"joint_normalization": jn}
                    )
                    diag_cache[key] = (src_diag, mom_diag)
                src_diag, mom_diag = diag_cache[key]

                use_soft_family = (
                    reuse_soft_families
                    and stage_cfg["method"] == "soft_delta_projection"
                    and stage_cfg.get("rank") is None
                )
                if use_soft_family:
                    if key not in soft_cache:
                        soft_cache[key] = fitter.prepare_soft_eraser_family(
                            affine=bool(stage_cfg.get("affine", True)),
                            delta_sources=specs,
                            normalize_source_weights=bool(
                                stage_cfg.get("normalize_source_weights", True)
                            ),
                            shrink_A=bool(stage_cfg.get("shrink_A", True)),
                            ridge=float(stage_cfg.get("ridge", 1e-4)),
                            svd_tol=float(stage_cfg.get("svd_tol", 1e-7)),
                            joint_normalization=str(
                                stage_cfg.get("joint_normalization", "none")
                            ),
                        )
                    eraser = soft_cache[key].make_eraser(float(stage_cfg["lam"]))
                else:
                    eraser = fit_eraser(
                        fitter=fitter, stage_cfg=stage_cfg, source_specs=specs
                    )

                comp_path: Path | None = None
                next_comp_paths = list(comp_paths)
                if save_erasers:
                    comp_path = eraser_dir / (
                        make_stage_name(stage_cfg, fold_idx)
                        + f"__prefix-{safe_name('__'.join(c['name'] for c in [*cfg_prefix, stage_cfg]))}.npz"
                    )
                    save_eraser_npz(
                        comp_path,
                        eraser,
                        metadata={
                            "fold": fold_idx,
                            "stage_index": stage_idx,
                            "stage_config": dict(stage_cfg),
                            "source_specs": [asdict(s) for s in specs],
                            "source_diagnostics": src_diag,
                            "moment_diagnostics": mom_diag,
                        },
                    )
                    next_comp_paths.append(comp_path)

                kw = dict(device=device, dtype=dtype, batch_size=apply_batch_size)
                x_train_next = apply_eraser(eraser, x_train, **kw)
                x_test_next = apply_eraser(eraser, x_test, **kw)
                src_train_next = {
                    n: apply_delta_transform(eraser, v, **kw)
                    for n, v in src_train.items()
                }
                src_test_next = {
                    n: apply_delta_transform(eraser, v, **kw)
                    for n, v in src_test.items()
                }

                stain_x_train_next = (
                    apply_eraser(eraser, stain_x_train, **kw)
                    if stain_x_train is not None
                    else None
                )
                stain_x_test_next = (
                    apply_eraser(eraser, stain_x_test, **kw)
                    if stain_x_test is not None
                    else None
                )

                stage_scanner_probe = stage_stain_probe = None
                next_stage_rows = list(acc_stage_rows)
                if evaluate_intermediate_stages:
                    stage_scanner_probe = evaluate_probe_train_test(
                        x_train_next,
                        x_test_next,
                        scanner_train,
                        scanner_test,
                        probe_type=probe_type,
                    )
                    if raw_stain_probe is not None and stain_x_train_next is not None:
                        assert stain_probe_data is not None
                        stage_stain_probe = evaluate_probe_train_test(
                            stain_x_train_next,
                            stain_x_test_next,
                            stain_probe_data.y_train,
                            stain_probe_data.y_test,
                            probe_type=probe_type,
                        )

                    s_feat_change = feature_change_summary(
                        raw=x_test_raw, projected=x_test_next
                    )
                    s_feat_variance = feature_variance_metrics(
                        raw=x_test_raw,
                        projected=x_test_next,
                        reference_trace_A=ref_trace_A,
                        projected_reference=x_train_next,
                    )
                    next_stage_rows.append(
                        {
                            "fold": fold_idx,
                            "stage_index": stage_idx,
                            "stage_name": str(stage_cfg["name"]),
                            "method": str(stage_cfg["method"]),
                            "rank": -1
                            if stage_cfg.get("rank") is None
                            else int(stage_cfg["rank"]),
                            "lambda": np.nan
                            if stage_cfg.get("lam") is None
                            else float(stage_cfg["lam"]),
                            "shrink_A": bool(stage_cfg.get("shrink_A", True)),
                            "source_names": json.dumps([s.name for s in specs]),
                            "scanner_balanced_accuracy": stage_scanner_probe.balanced_accuracy,
                            "scanner_accuracy": stage_scanner_probe.accuracy,
                            "stain_target_balanced_accuracy": np.nan
                            if stage_stain_probe is None
                            else stage_stain_probe.balanced_accuracy,
                            **{
                                k: s_feat_change[k]
                                for k in (
                                    "mean_l2_change",
                                    "median_l2_change",
                                    "mean_raw_norm",
                                    "mean_relative_change",
                                )
                            },
                            **s_feat_variance,
                            "component_eraser_path": ""
                            if comp_path is None
                            else str(comp_path),
                        }
                    )

                next_moment_rows = list(acc_moment_rows)
                if source_moment_diag:
                    next_moment_rows.append(
                        {
                            "fold": fold_idx,
                            "stage_index": stage_idx,
                            "stage_name": str(stage_cfg["name"]),
                            "method": str(stage_cfg["method"]),
                            "source_names": json.dumps([s.name for s in specs]),
                            **mom_diag,
                        }
                    )

                _walk(
                    stage_idx=stage_idx + 1,
                    x_train=x_train_next,
                    x_test=x_test_next,
                    src_train=src_train_next,
                    src_test=src_test_next,
                    stain_x_train=stain_x_train_next,
                    stain_x_test=stain_x_test_next,
                    cfg_prefix=[*cfg_prefix, dict(stage_cfg)],
                    erasers=[*erasers, eraser],
                    comp_paths=next_comp_paths,
                    acc_stage_rows=next_stage_rows,
                    acc_moment_rows=next_moment_rows,
                    acc_stage_diags=[
                        *acc_stage_diags,
                        {
                            "stage_index": stage_idx,
                            "stage_name": str(stage_cfg["name"]),
                            "stage_config": dict(stage_cfg),
                            "source_specs": [asdict(s) for s in specs],
                            "source_diagnostics": src_diag,
                            "moment_diagnostics": mom_diag,
                            "component_eraser_path": ""
                            if comp_path is None
                            else str(comp_path),
                        },
                    ],
                    last_scanner_probe=stage_scanner_probe,
                    last_stain_probe=stage_stain_probe,
                )

        _walk(
            stage_idx=0,
            x_train=x_train_raw,
            x_test=x_test_raw,
            src_train={n: s.train for n, s in sources.items()},
            src_test={n: s.test for n, s in sources.items()},
            stain_x_train=stain_probe_data.x_train if stain_probe_data else None,
            stain_x_test=stain_probe_data.x_test if stain_probe_data else None,
            cfg_prefix=[],
            erasers=[],
            comp_paths=[],
            acc_stage_rows=[],
            acc_moment_rows=[],
            acc_stage_diags=[],
            last_scanner_probe=None,
            last_stain_probe=None,
        )

        diagnostics["folds"].append(fold_diag)
        _flush()
        _atomic_write_json(diagnostics, output_dir / "diagnostics.json")

    chain_scores = pd.DataFrame(chain_rows)
    delta_scores = pd.DataFrame(delta_rows)

    group_cols = [
        "stage_names",
        "stage_methods",
        "stage_ranks",
        "stage_lambdas",
        "stage_configs",
    ]
    (
        chain_scores.groupby(group_cols, dropna=False)
        .agg(
            raw_score_mean=("raw_score", "mean"),
            raw_score_std=("raw_score", "std"),
            projected_score_mean=("projected_score", "mean"),
            projected_score_std=("projected_score", "std"),
            projected_stain_target_balanced_accuracy_mean=(
                "projected_stain_target_balanced_accuracy",
                "mean",
            ),
            scanner_probe_excess_ratio_mean=("scanner_probe_excess_ratio", "mean"),
            stain_probe_excess_ratio_mean=("stain_probe_excess_ratio", "mean"),
            n_folds=("fold", "nunique"),
        )
        .reset_index()
        .to_csv(output_dir / "summary_by_chain.csv", index=False)
    )

    if not delta_scores.empty:
        (
            delta_scores.groupby(
                [
                    "stage_names",
                    "stage_configs",
                    "evaluation_source",
                    "evaluation_source_kind",
                ],
                dropna=False,
            )
            .agg(
                remaining_delta_energy_ratio_mean=(
                    "remaining_delta_energy_ratio",
                    "mean",
                ),
                remaining_delta_energy_ratio_std=(
                    "remaining_delta_energy_ratio",
                    "std",
                ),
                delta_residual_energy_ratio_mean=(
                    "delta_residual_energy_ratio",
                    "mean",
                ),
                n_folds=("fold", "nunique"),
            )
            .reset_index()
            .to_csv(output_dir / "summary_by_delta_source.csv", index=False)
        )

    _atomic_write_json(diagnostics, output_dir / "diagnostics.json")
    return diagnostics
