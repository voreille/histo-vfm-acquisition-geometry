"""Paired-delta moment accumulation and eraser fitting.

Three eraser types:
  LinearProjectionEraser  – removes top nuisance directions (PCA or hard).
  SoftDeltaProjectionEraser – soft attenuation P = A(A + lam B)^-1.
  PreparedSoftDeltaFamily   – pre-decomposed family for sweeping lambda.

PairedDeltaFitter accumulates X statistics and named delta sources,
then builds any of the erasers above.
"""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from typing import Any, Literal, Mapping, Sequence

import torch
from torch import Tensor

try:
    from .shrinkage import optimal_linear_shrinkage
except ImportError:
    optimal_linear_shrinkage = None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeltaSourceSpec:
    """How one nuisance-delta source contributes to the joint moment matrix."""

    name: str
    weight: float = 1.0
    moment: Literal["covariance", "second_moment"] = "second_moment"
    shrinkage: bool = False
    normalization: Literal["none", "trace", "frobenius"] = "none"

    @classmethod
    def from_value(
        cls, value: "DeltaSourceSpec | Mapping[str, Any]"
    ) -> "DeltaSourceSpec":
        if isinstance(value, cls):
            return value
        return cls(
            name=str(value["name"]),
            weight=float(value.get("weight", 1.0)),
            moment=value.get("moment", "second_moment"),
            shrinkage=bool(value.get("shrinkage", False)),
            normalization=value.get("normalization", "none"),
        )


# ---------------------------------------------------------------------------
# Erasers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LinearProjectionEraser:
    """Removes top nuisance directions: x -> mu + (I - VW^H)(x - mu).

    Used for both PCA-based and hard covariance-aware erasure; the only
    difference is how V (proj_left) and W (proj_right) are constructed.
    """

    proj_left: Tensor
    proj_right: Tensor
    bias: Tensor | None
    eigenvalues: Tensor

    @property
    def rank(self) -> int:
        return int(self.proj_left.shape[1])

    def apply_linear(self, x: Tensor) -> Tensor:
        work = x.to(device=self.proj_left.device, dtype=self.proj_left.dtype)
        out = work - (work @ self.proj_right.mH) @ self.proj_left.mH
        return out.to(device=x.device, dtype=x.dtype)

    def transform_delta(self, delta: Tensor) -> Tensor:
        return self.apply_linear(delta)

    def __call__(self, x: Tensor) -> Tensor:
        work = x.to(device=self.proj_left.device, dtype=self.proj_left.dtype)
        centered = work - self.bias if self.bias is not None else work
        out = self.apply_linear(centered)
        if self.bias is not None:
            out = out + self.bias
        return out.to(device=x.device, dtype=x.dtype)

    def transform(self, x: Tensor) -> Tensor:
        return self(x)

    def state_dict(self) -> dict[str, Any]:
        return {
            "proj_left": self.proj_left,
            "proj_right": self.proj_right,
            "bias": self.bias,
            "eigenvalues": self.eigenvalues,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "LinearProjectionEraser":
        return cls(
            proj_left=state["proj_left"],
            proj_right=state["proj_right"],
            bias=state.get("bias"),
            eigenvalues=state.get("eigenvalues", torch.empty(0)),
        )

    def save(self, path: str | PathLike[str]) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load(
        cls, path: str | PathLike[str], map_location=None
    ) -> "LinearProjectionEraser":
        return cls.from_state_dict(
            torch.load(path, map_location=map_location, weights_only=True)
        )

    def to(self, device=None, dtype=None) -> "LinearProjectionEraser":
        def _m(t: Tensor | None) -> Tensor | None:
            return t.to(device=device, dtype=dtype) if t is not None else None

        return LinearProjectionEraser(
            _m(self.proj_left), _m(self.proj_right), _m(self.bias), _m(self.eigenvalues)
        )


# Aliases for backward-compatible state-dict loading.
PairedDeltaPcaEraser = LinearProjectionEraser
HardDeltaProjectionEraser = LinearProjectionEraser


@dataclass(frozen=True)
class SoftDeltaProjectionEraser:
    """Soft attenuation: x -> mu + P(x - mu), P = A(A + lam B)^{-1}."""

    P: Tensor
    bias: Tensor | None
    lam: float

    def apply_linear(self, x: Tensor) -> Tensor:
        work = x.to(device=self.P.device, dtype=self.P.dtype)
        return (work @ self.P.mH).to(device=x.device, dtype=x.dtype)

    def transform_delta(self, delta: Tensor) -> Tensor:
        return self.apply_linear(delta)

    def __call__(self, x: Tensor) -> Tensor:
        work = x.to(device=self.P.device, dtype=self.P.dtype)
        centered = work - self.bias if self.bias is not None else work
        out = self.apply_linear(centered)
        if self.bias is not None:
            out = out + self.bias
        return out.to(device=x.device, dtype=x.dtype)

    def transform(self, x: Tensor) -> Tensor:
        return self(x)

    def state_dict(self) -> dict[str, Any]:
        return {"P": self.P, "bias": self.bias, "lam": self.lam}

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "SoftDeltaProjectionEraser":
        P = state.get("P")
        if P is None:  # reconstruct P from old low-rank format
            pl, pr = state.get("proj_left"), state.get("proj_right")
            if pl is not None and pr is not None:
                eye = torch.eye(pl.shape[0], device=pl.device, dtype=pl.dtype)
                P = eye - pl @ pr
        return cls(P=P, bias=state.get("bias"), lam=float(state["lam"]))

    def save(self, path: str | PathLike[str]) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load(
        cls, path: str | PathLike[str], map_location=None
    ) -> "SoftDeltaProjectionEraser":
        return cls.from_state_dict(
            torch.load(path, map_location=map_location, weights_only=True)
        )

    def to(self, device=None, dtype=None) -> "SoftDeltaProjectionEraser":
        def _m(t: Tensor | None) -> Tensor | None:
            return t.to(device=device, dtype=dtype) if t is not None else None

        return SoftDeltaProjectionEraser(_m(self.P), _m(self.bias), self.lam)


@dataclass(frozen=True)
class PreparedSoftDeltaFamily:
    """Precomputed generalized eigendecomposition for sweeping lambda cheaply.

    P(lam) = I - basis_left * diag(beta(lam)) * basis_right,
    where beta_i = lam * mu_i / (1 + lam * mu_i).
    """

    basis_left: Tensor
    basis_right: Tensor
    generalized_eigenvalues: Tensor
    bias: Tensor | None

    @torch.no_grad()
    def make_eraser(self, lam: float) -> SoftDeltaProjectionEraser:
        mu = self.generalized_eigenvalues
        if lam == 0:
            beta = torch.zeros_like(mu)
        else:
            scaled = float(lam) * mu
            beta = (scaled / (1.0 + scaled)).clamp(0.0, 1.0)
        eye = torch.eye(
            self.basis_left.shape[0],
            device=self.basis_left.device,
            dtype=self.basis_left.dtype,
        )
        P = eye - (self.basis_left * beta.unsqueeze(0)) @ self.basis_right
        return SoftDeltaProjectionEraser(P=P, bias=self.bias, lam=float(lam))


# ---------------------------------------------------------------------------
# Running moment accumulator
# ---------------------------------------------------------------------------


@dataclass
class _Stats:
    mean: Tensor
    m2: Tensor  # unscaled sum of outer products (Welford)
    count: Tensor

    @classmethod
    def zeros(cls, dim: int, device, dtype) -> "_Stats":
        return cls(
            mean=torch.zeros(dim, device=device, dtype=dtype),
            m2=torch.zeros(dim, dim, device=device, dtype=dtype),
            count=torch.tensor(0, device=device, dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# Fitter
# ---------------------------------------------------------------------------


class PairedDeltaFitter:
    """Accumulates X and delta statistics, then fits erasers.

    Usage::

        fitter = PairedDeltaFitter(x_dim=d, device="cuda", dtype=torch.float32)
        fitter.update_x(x_train)
        fitter.update_delta_source("scanner", scanner_deltas)
        fitter.update_delta_source("stain", stain_deltas)
        eraser = fitter.make_pca_eraser(rank=4, ...)
    """

    DEFAULT_SOURCE_NAME = "default"

    def __init__(self, x_dim: int, *, device=None, dtype=None) -> None:
        self.x_dim = int(x_dim)
        self._x = _Stats.zeros(self.x_dim, device, dtype)
        self._deltas: dict[str, _Stats] = {}

    @property
    def mean_x(self) -> Tensor:
        return self._x.mean

    @property
    def delta_source_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._deltas))

    def delta_count(self, name: str) -> int:
        return int(self._deltas[name].count.item())

    @torch.no_grad()
    def update_x(self, x: Tensor) -> "PairedDeltaFitter":
        self._update(
            x.reshape(-1, self.x_dim).to(
                device=self._x.mean.device, dtype=self._x.mean.dtype
            ),
            self._x,
        )
        return self

    @torch.no_grad()
    def update_delta_source(self, name: str, delta: Tensor) -> "PairedDeltaFitter":
        if name not in self._deltas:
            self._deltas[name] = _Stats.zeros(
                self.x_dim, self._x.mean.device, self._x.mean.dtype
            )
        stats = self._deltas[name]
        self._update(
            delta.reshape(-1, self.x_dim).to(
                device=stats.mean.device, dtype=stats.mean.dtype
            ),
            stats,
        )
        return self

    @torch.no_grad()
    def update(
        self,
        x: Tensor | None = None,
        delta: Tensor | None = None,
        *,
        source_name: str = DEFAULT_SOURCE_NAME,
    ) -> "PairedDeltaFitter":
        if x is not None:
            self.update_x(x)
        if delta is not None:
            self.update_delta_source(source_name, delta)
        return self

    @staticmethod
    @torch.no_grad()
    def _update(batch: Tensor, stats: _Stats) -> None:
        n = batch.shape[0]
        if n == 0:
            return
        old_n = stats.count.clone()
        new_n = old_n + n
        batch_mean = batch.mean(0)
        shift = batch_mean - stats.mean
        centered = batch - batch_mean
        stats.m2.add_(
            centered.mH @ centered
            + torch.outer(shift, shift.conj())
            * (old_n.to(batch.dtype) * n / new_n.to(batch.dtype))
        )
        stats.mean.add_(shift * (n / new_n.to(batch.dtype)))
        stats.count.copy_(new_n)

    @staticmethod
    def _sym(m: Tensor) -> Tensor:
        return (m + m.mH) / 2

    @staticmethod
    def _eigh(m: Tensor) -> tuple[Tensor, Tensor]:
        vals, vecs = torch.linalg.eigh((m + m.mH) / 2)
        return vals.clamp_min(0), vecs

    def _covariance(self, stats: _Stats, *, shrinkage: bool) -> Tensor:
        m2 = self._sym(stats.m2)
        if shrinkage and optimal_linear_shrinkage is not None:
            cov = optimal_linear_shrinkage(
                m2 / stats.count.to(m2.dtype), stats.count, inplace=False
            )
        else:
            cov = m2 / (stats.count - 1).to(m2.dtype)
        return self._sym(cov)

    def covariance_x(self, *, shrinkage: bool = True) -> Tensor:
        return self._covariance(self._x, shrinkage=shrinkage)

    def _delta_matrix(self, name: str, *, moment: str, shrinkage: bool) -> Tensor:
        stats = self._deltas[name]
        if moment == "covariance":
            return self._covariance(stats, shrinkage=shrinkage)
        # second_moment = population covariance + mean outer product
        m2 = self._sym(stats.m2)
        if shrinkage and optimal_linear_shrinkage is not None:
            cov = optimal_linear_shrinkage(
                m2 / stats.count.to(m2.dtype), stats.count, inplace=False
            )
        else:
            n = stats.count.to(m2.dtype)
            cov = m2 / n if n > 0 else torch.zeros_like(m2)
        return self._sym(cov + torch.outer(stats.mean, stats.mean.conj()))

    def delta_matrix_for_source(
        self, name: str, *, moment: str, shrinkage: bool = False
    ) -> Tensor:
        return self._delta_matrix(name, moment=moment, shrinkage=shrinkage)

    def combined_delta_matrix(
        self,
        source_specs: Sequence[DeltaSourceSpec | Mapping[str, Any]],
        *,
        normalize_source_weights: bool = True,
    ) -> Tensor:
        specs = [DeltaSourceSpec.from_value(s) for s in source_specs]
        total_w = sum(s.weight for s in specs if s.weight > 0)
        B = torch.zeros(
            self.x_dim, self.x_dim, device=self.mean_x.device, dtype=self.mean_x.dtype
        )
        for spec in specs:
            if spec.weight == 0:
                continue
            mat = self._delta_matrix(
                spec.name, moment=spec.moment, shrinkage=spec.shrinkage
            )
            if spec.normalization == "trace":
                mat = mat / torch.real(torch.trace(mat)).abs().clamp_min(1e-12)
            elif spec.normalization == "frobenius":
                mat = mat / torch.linalg.matrix_norm(mat, ord="fro").clamp_min(1e-12)
            w = spec.weight / total_w if normalize_source_weights else spec.weight
            B.add_(mat, alpha=float(w))
        return self._sym(B)

    def source_diagnostics(
        self, source_specs: Sequence[DeltaSourceSpec | Mapping[str, Any]]
    ) -> dict[str, Any]:
        out = {}
        for value in source_specs:
            spec = DeltaSourceSpec.from_value(value)
            raw = self._delta_matrix(
                spec.name, moment=spec.moment, shrinkage=spec.shrinkage
            )
            out[spec.name] = {
                "n_delta": self.delta_count(spec.name),
                "weight": spec.weight,
                "moment": spec.moment,
                "shrinkage": spec.shrinkage,
                "normalization": spec.normalization,
                "raw_trace": float(torch.real(torch.trace(raw)).item()),
                "raw_frobenius": float(torch.linalg.matrix_norm(raw, ord="fro").item()),
            }
        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _whitening_matrices(
        self, *, shrink_A: bool, ridge: float, svd_tol: float
    ) -> tuple[Tensor, Tensor]:
        A = self.covariance_x(shrinkage=shrink_A)
        eye = torch.eye(self.x_dim, device=A.device, dtype=A.dtype)
        vals, vecs = self._eigh(A + ridge * eye)
        thr = svd_tol * vals.max().clamp_min(1.0)
        inv_sqrt = self._sym(
            (vecs * torch.where(vals > thr, vals.rsqrt(), torch.zeros_like(vals)))
            @ vecs.mH
        )
        sqrt = self._sym(
            (vecs * torch.where(vals > thr, vals.sqrt(), torch.zeros_like(vals)))
            @ vecs.mH
        )
        return inv_sqrt, sqrt

    def _build_B(
        self,
        *,
        source_specs: Sequence[DeltaSourceSpec | Mapping[str, Any]] | None,
        delta_moment: str,
        shrink_B: bool,
        source_normalization: str,
        normalize_source_weights: bool,
        joint_normalization: str,
        A: Tensor,
    ) -> Tensor:
        if source_specs is None:
            source_specs = [
                DeltaSourceSpec(
                    self.DEFAULT_SOURCE_NAME,
                    moment=delta_moment,
                    shrinkage=shrink_B,
                    normalization=source_normalization,
                )
            ]
        B = self.combined_delta_matrix(
            source_specs, normalize_source_weights=normalize_source_weights
        )
        if joint_normalization == "match_x_trace":
            B = B * (
                torch.real(torch.trace(A)).abs()
                / torch.real(torch.trace(B)).abs().clamp_min(1e-12)
            )
        return B

    # ------------------------------------------------------------------
    # Eraser builders
    # ------------------------------------------------------------------

    @torch.no_grad()
    def make_pca_eraser(
        self,
        *,
        rank: int,
        whitening: bool = False,
        affine: bool = True,
        delta_sources: Sequence[DeltaSourceSpec | Mapping[str, Any]] | None = None,
        normalize_source_weights: bool = True,
        delta_moment: str = "second_moment",
        shrink_A: bool = True,
        shrink_B: bool = False,
        source_normalization: str = "none",
        ridge: float = 1e-4,
        svd_tol: float = 1e-7,
    ) -> LinearProjectionEraser:
        A = self.covariance_x(shrinkage=shrink_A)
        B = self._build_B(
            source_specs=delta_sources,
            delta_moment=delta_moment,
            shrink_B=shrink_B,
            source_normalization=source_normalization,
            normalize_source_weights=normalize_source_weights,
            joint_normalization="none",
            A=A,
        )
        if not whitening:
            vals, vecs = self._eigh(B)
            idx = torch.argsort(vals, descending=True)[:rank]
            keep = idx[vals[idx] > svd_tol * vals.max().clamp_min(1.0)]
            proj_left, proj_right = vecs[:, keep], vecs[:, keep].mH
        else:
            inv_sqrt, sqrt = self._whitening_matrices(
                shrink_A=shrink_A, ridge=ridge, svd_tol=svd_tol
            )
            vals, vecs = self._eigh(self._sym(inv_sqrt @ B @ inv_sqrt.mH))
            idx = torch.argsort(vals, descending=True)[:rank]
            keep = idx[vals[idx] > svd_tol * vals.max().clamp_min(1.0)]
            proj_left = sqrt @ vecs[:, keep]
            proj_right = vecs[:, keep].mH @ inv_sqrt
        return LinearProjectionEraser(
            proj_left=proj_left,
            proj_right=proj_right,
            bias=self.mean_x.clone() if affine else None,
            eigenvalues=vals[torch.argsort(vals, descending=True)[:rank]],
        )

    @torch.no_grad()
    def make_hard_eraser(
        self,
        *,
        rank: int | None = None,
        affine: bool = True,
        delta_sources: Sequence[DeltaSourceSpec | Mapping[str, Any]] | None = None,
        normalize_source_weights: bool = True,
        delta_moment: str = "second_moment",
        shrink_A: bool = True,
        shrink_B: bool = False,
        source_normalization: str = "none",
        joint_normalization: str = "none",
        ridge: float = 1e-4,
        svd_tol: float = 1e-7,
    ) -> LinearProjectionEraser:
        A = self.covariance_x(shrinkage=shrink_A)
        B = self._build_B(
            source_specs=delta_sources,
            delta_moment=delta_moment,
            shrink_B=shrink_B,
            source_normalization=source_normalization,
            normalize_source_weights=normalize_source_weights,
            joint_normalization=joint_normalization,
            A=A,
        )
        inv_sqrt, sqrt = self._whitening_matrices(
            shrink_A=shrink_A, ridge=ridge, svd_tol=svd_tol
        )
        vals, vecs = self._eigh(self._sym(inv_sqrt @ B @ inv_sqrt.mH))
        r = self.x_dim if rank is None else int(rank)
        idx = torch.argsort(vals, descending=True)[:r]
        keep = idx[vals[idx] > svd_tol * vals.max().clamp_min(1.0)]
        return LinearProjectionEraser(
            proj_left=sqrt @ vecs[:, keep],
            proj_right=vecs[:, keep].mH @ inv_sqrt,
            bias=self.mean_x.clone() if affine else None,
            eigenvalues=vals[torch.argsort(vals, descending=True)[:r]],
        )

    @torch.no_grad()
    def prepare_soft_eraser_family(
        self,
        *,
        affine: bool = True,
        delta_sources: Sequence[DeltaSourceSpec | Mapping[str, Any]] | None = None,
        normalize_source_weights: bool = True,
        delta_moment: str = "second_moment",
        shrink_A: bool = True,
        shrink_B: bool = False,
        source_normalization: str = "none",
        joint_normalization: str = "none",
        ridge: float = 1e-4,
        svd_tol: float = 1e-7,
    ) -> PreparedSoftDeltaFamily:
        A = self.covariance_x(shrinkage=shrink_A)
        B = self._build_B(
            source_specs=delta_sources,
            delta_moment=delta_moment,
            shrink_B=shrink_B,
            source_normalization=source_normalization,
            normalize_source_weights=normalize_source_weights,
            joint_normalization=joint_normalization,
            A=A,
        )
        eye = torch.eye(self.x_dim, device=A.device, dtype=A.dtype)
        A_reg = self._sym(A + ridge * eye)
        a_vals, a_vecs = torch.linalg.eigh(A_reg)
        a_vals = a_vals.clamp_min(
            torch.finfo(A.dtype).eps * a_vals.abs().max().clamp_min(1.0)
        )
        A_sqrt = self._sym((a_vecs * a_vals.sqrt()) @ a_vecs.mH)
        A_inv_sqrt = self._sym((a_vecs * a_vals.rsqrt()) @ a_vecs.mH)
        vals, vecs = self._eigh(self._sym(A_inv_sqrt @ B @ A_inv_sqrt.mH))
        order = torch.argsort(vals, descending=True)
        return PreparedSoftDeltaFamily(
            basis_left=A_sqrt @ vecs[:, order],
            basis_right=vecs[:, order].mH @ A_inv_sqrt,
            generalized_eigenvalues=vals[order],
            bias=self.mean_x.clone() if affine else None,
        )

    @torch.no_grad()
    def make_soft_eraser(
        self,
        *,
        lam: float,
        rank: int | None = None,
        affine: bool = True,
        delta_sources: Sequence[DeltaSourceSpec | Mapping[str, Any]] | None = None,
        normalize_source_weights: bool = True,
        delta_moment: str = "second_moment",
        shrink_A: bool = True,
        shrink_B: bool = False,
        source_normalization: str = "none",
        joint_normalization: str = "none",
        ridge: float = 1e-4,
        svd_tol: float = 1e-7,
    ) -> SoftDeltaProjectionEraser:
        A = self.covariance_x(shrinkage=shrink_A)
        B = self._build_B(
            source_specs=delta_sources,
            delta_moment=delta_moment,
            shrink_B=shrink_B,
            source_normalization=source_normalization,
            normalize_source_weights=normalize_source_weights,
            joint_normalization=joint_normalization,
            A=A,
        )
        eye = torch.eye(self.x_dim, device=A.device, dtype=A.dtype)
        A_reg = A + ridge * eye
        P = torch.linalg.solve(self._sym(A_reg + float(lam) * B).mH, A_reg.mH).mH.to(
            dtype=A.dtype
        )
        return SoftDeltaProjectionEraser(
            P=P, bias=self.mean_x.clone() if affine else None, lam=float(lam)
        )

    @torch.no_grad()
    def make_soft_erasers(
        self, lams: Sequence[float], **kwargs: Any
    ) -> list[SoftDeltaProjectionEraser]:
        """Build multiple soft erasers sharing one eigendecomposition."""
        family = self.prepare_soft_eraser_family(
            **{k: v for k, v in kwargs.items() if k != "lam"}
        )
        return [family.make_eraser(float(lam)) for lam in lams]
