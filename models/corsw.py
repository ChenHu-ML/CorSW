"""Correlation Sliced-Wasserstein distances and training loss.

The public API accepts either full-rank correlation matrices or features that
have already been mapped into OLM/LSM pullback coordinates by CorAtt.
"""

from __future__ import annotations

from typing import Literal, Union

import numpy as np
import torch
from torch import Tensor

from .CorAtt import lsm_diffeomorphism, olm_diffeomorphism


Metric = Literal["olm", "lsm"]
InputSpace = Literal["correlation", "pullback"]


def _metric_name(metric: str) -> Metric:
    metric = str(metric).lower()
    if metric not in {"olm", "lsm"}:
        raise ValueError(f"metric must be 'olm' or 'lsm'; got {metric!r}.")
    return metric  # type: ignore[return-value]


def correlation_pullback(
    x: Tensor,
    *,
    metric: Union[Metric, str] = "olm",
) -> Tensor:
    """Map full-rank correlation matrices into OLM or LSM coordinates.

    Inputs are assumed to be symmetric positive-definite with unit diagonal.
    """
    metric = _metric_name(metric)
    if metric == "olm":
        return olm_diffeomorphism(x)
    return lsm_diffeomorphism(x)


def emd1d(u_values: Tensor, v_values: Tensor, p: float = 2) -> Tensor:
    """Return empirical one-dimensional Wasserstein costs.

    Samples occupy the last dimension. Preceding dimensions are independent
    batches, such as CorSW slicing directions. Both empirical measures use
    uniform weights and may contain different numbers of samples.
    """
    u_values = torch.sort(u_values, dim=-1).values
    v_values = torch.sort(v_values, dim=-1).values
    n, m = u_values.shape[-1], v_values.shape[-1]

    u_cdf = torch.cumsum(torch.full_like(u_values, 1.0 / n), dim=-1)
    v_cdf = torch.cumsum(torch.full_like(v_values, 1.0 / m), dim=-1)
    cdf_axis = torch.sort(torch.cat((u_cdf, v_cdf), dim=-1), dim=-1).values.contiguous()

    u_index = torch.searchsorted(u_cdf.contiguous(), cdf_axis)
    v_index = torch.searchsorted(v_cdf.contiguous(), cdf_axis)
    u_icdf = torch.gather(u_values, -1, u_index.clamp(0, n - 1))
    v_icdf = torch.gather(v_values, -1, v_index.clamp(0, m - 1))

    padded_axis = torch.nn.functional.pad(cdf_axis, (1, 0))
    delta = padded_axis[..., 1:] - padded_axis[..., :-1]
    return torch.sum(delta * torch.abs(u_icdf - v_icdf).pow(p), dim=-1)


def _random_slicing_directions(
    num_projections: int,
    matrix_size: int,
    *,
    metric: Metric,
    dtype: torch.dtype,
    device: torch.device,
    random_state: int,
) -> Tensor:
    rng = np.random.default_rng(random_state)
    raw = torch.as_tensor(
        rng.normal(size=(num_projections, matrix_size, matrix_size)),
        dtype=dtype,
        device=device,
    )
    if metric == "olm":
        lower = raw.tril(-1)
        directions = lower + lower.transpose(-1, -2)
    else:
        # Orthogonal projection onto symmetric matrices whose row sums vanish,
        # the Euclidean pullback subspace associated with LSM.
        symmetric = (raw + raw.transpose(-1, -2)) / 2
        identity = torch.eye(matrix_size, dtype=dtype, device=device)
        ones = torch.ones((matrix_size, matrix_size), dtype=dtype, device=device)
        projector = identity - ones / matrix_size
        directions = projector @ symmetric @ projector
    return directions / directions.norm(dim=(1, 2), keepdim=True).clamp_min(1e-12)


def _corsw_from_pullbacks(
    x: Tensor,
    x_ref: Tensor,
    *,
    metric: Metric,
    num_projections: int,
    p: float,
    random_state: int,
) -> Tensor:
    x_ref = x_ref.to(device=x.device, dtype=x.dtype)
    directions = _random_slicing_directions(
        num_projections,
        x.shape[-1],
        metric=metric,
        dtype=x.dtype,
        device=x.device,
        random_state=random_state,
    )
    x_projected = torch.einsum("pij,nij->pn", directions, x)
    ref_projected = torch.einsum("pij,nij->pn", directions, x_ref)
    return emd1d(x_projected, ref_projected, p=p).mean()


def corsw_distance(
    x: Tensor,
    x_ref: Tensor,
    *,
    metric: Union[Metric, str] = "olm",
    input_space: Union[InputSpace, str] = "correlation",
    num_projections: int = 1000,
    p: float = 2,
    random_state: int = 123456,
) -> Tensor:
    """Compute CorSW for correlation matrices or pullback coordinates.

    Args:
        x: First empirical distribution with shape ``(samples, d, d)``.
        x_ref: Second empirical distribution with shape ``(samples, d, d)``.
        metric: Correlation geometry, ``"olm"`` or ``"lsm"``.
        input_space: ``"correlation"`` for genuine full-rank correlation
            matrices, or ``"pullback"`` for features already mapped by CorAtt.

    Inputs in correlation space are assumed to be full-rank correlation
    matrices; no runtime manifold validation is performed.
    """
    metric = _metric_name(metric)
    input_space = str(input_space).lower()
    if input_space == "correlation":
        x = correlation_pullback(x, metric=metric)
        x_ref = correlation_pullback(x_ref, metric=metric)
    elif input_space != "pullback":
        raise ValueError(f"Unknown input_space: {input_space!r}.")
    return _corsw_from_pullbacks(
        x,
        x_ref,
        metric=metric,
        num_projections=num_projections,
        p=p,
        random_state=random_state,
    )


def _gaussian_pullback_reference(
    num_samples: int,
    matrix_size: int,
    *,
    metric: Metric,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    raw = torch.randn((num_samples, matrix_size, matrix_size), dtype=dtype, device=device)
    if metric == "olm":
        lower = raw.tril(-1)
        reference = lower + lower.transpose(-1, -2)
    else:
        symmetric = (raw + raw.transpose(-1, -2)) / 2
        identity = torch.eye(matrix_size, dtype=dtype, device=device)
        ones = torch.ones((matrix_size, matrix_size), dtype=dtype, device=device)
        projector = identity - ones / matrix_size
        reference = projector @ symmetric @ projector
    return reference / reference.norm(dim=(1, 2), keepdim=True).clamp_min(1e-8)


def swd_loss(
    x: Tensor,
    domains: Tensor,
    *,
    metric: Union[Metric, str] = "olm",
    input_space: Union[InputSpace, str] = "pullback",
    num_projections: int = 1000,
    p: float = 2,
    random_state: int = 123456,
) -> Tensor:
    """Sum domain-wise CorSW losses against metric-aware Gaussian references."""
    metric = _metric_name(metric)
    input_space = str(input_space).lower()
    if input_space == "correlation":
        x = correlation_pullback(x, metric=metric)
    elif input_space != "pullback":
        raise ValueError(f"Unknown input_space: {input_space!r}.")

    domains = domains.reshape(-1).to(x.device)

    loss = x.new_zeros(())
    for domain in domains.unique(sorted=True):
        domain_features = x[domains == domain]
        reference = _gaussian_pullback_reference(
            domain_features.shape[0],
            x.shape[-1],
            metric=metric,
            dtype=x.dtype,
            device=x.device,
        )
        loss = loss + _corsw_from_pullbacks(
            domain_features,
            reference,
            metric=metric,
            num_projections=num_projections,
            p=p,
            random_state=random_state,
        )
    return loss
