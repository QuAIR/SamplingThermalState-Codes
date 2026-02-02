#!/usr/bin/env python3
# Copyright (c) 2026 QuAIR team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import numpy as np
from typing import List, Tuple, Dict, Optional, Literal
from scipy.linalg import expm
from scipy.ndimage import gaussian_filter, gaussian_filter1d

import quairkit as qkit
from quairkit import State, Hamiltonian
from quairkit.database import *
from quairkit.qinfo import *

import os
from concurrent.futures import ThreadPoolExecutor

LN2 = np.log(2.0)

MAX_WORKERS = min(8, os.cpu_count() or 1)

__all__ = ['build_2d_tfim', 'build_2d_heisenberg', 'negativity_spectrum', 'r_stats_from_gibbs_states', 'compare_terms_full_dim_mc',]

def idx(x, y, Ly):
    return x * Ly + y

def build_2d_tfim(Lx, Ly, J=1.0, h=1.0):
    n = Lx * Ly
    terms = []

    # transverse field: -h * sum_i X_i
    for x in range(Lx):
        for y in range(Ly):
            i = idx(x, y, Ly)
            terms.append((-h, f"X{i}"))

    # Ising couplings: -J * sum_<i,j> Z_i Z_j (open boundary)
    for x in range(Lx):
        for y in range(Ly):
            i = idx(x, y, Ly)

            if x + 1 < Lx:  # down neighbor
                j = idx(x + 1, y, Ly)
                terms.append((-J, f"Z{i},Z{j}"))

            if y + 1 < Ly:  # right neighbor
                j = idx(x, y + 1, Ly)
                terms.append((-J, f"Z{i},Z{j}"))

    return n, Hamiltonian(terms)

def build_2d_heisenberg(Lx, Ly, Jxy=1.0, Jz=None, hx=0.0, hy=0.0, hz=0.0):
    """
    Construct a 2D Heisenberg Hamiltonian on an Lx x Ly square lattice
    with open boundary conditions.

    The Hamiltonian is:
        H = sum_<ij> [ Jxy (X_i X_j + Y_i Y_j) + Jz Z_i Z_j ]
            - sum_i (hx X_i + hy Y_i + hz Z_i)

    where:
        - <ij> denotes nearest-neighbor pairs (up/down/left/right)
        - Jxy is the coupling strength for XX and YY terms
        - Jz is the coupling strength for ZZ terms
        - hx, hy, hz are optional on-site (Zeeman) fields

    Special cases:
        - If Jz is None, it is set to Jxy (isotropic XXX model)
        - Setting hx = hy = hz = 0 gives a pure XXX/XXZ model
    """

    n = Lx * Ly                 # total number of spins
    terms = []

    # If Jz is not specified, use the isotropic XXX limit
    if Jz is None:
        Jz = Jxy

    # On-site Zeeman field terms (with a minus sign convention)
    for x in range(Lx):
        for y in range(Ly):
            i = idx(x, y, Ly)

            if hx != 0.0:
                terms.append((-hx, f"X{i}"))
            if hy != 0.0:
                terms.append((-hy, f"Y{i}"))
            if hz != 0.0:
                terms.append((-hz, f"Z{i}"))

    # Nearest-neighbor interactions (open boundary conditions)
    # Each bond is added once: (x+1, y) and (x, y+1)
    for x in range(Lx):
        for y in range(Ly):
            i = idx(x, y, Ly)

            # Neighbor in the +x direction
            if x + 1 < Lx:
                j = idx(x + 1, y, Ly)

                if Jxy != 0.0:
                    terms.append(( Jxy, f"X{i},X{j}"))
                    terms.append(( Jxy, f"Y{i},Y{j}"))
                if Jz != 0.0:
                    terms.append(( Jz,  f"Z{i},Z{j}"))

            # Neighbor in the +y direction
            if y + 1 < Ly:
                j = idx(x, y + 1, Ly)

                if Jxy != 0.0:
                    terms.append(( Jxy, f"X{i},X{j}"))
                    terms.append(( Jxy, f"Y{i},Y{j}"))
                if Jz != 0.0:
                    terms.append(( Jz,  f"Z{i},Z{j}"))

    return n, Hamiltonian(terms)


def _to_numpy(rho):
    # Compatible with torch / quairkit / numpy
    if hasattr(rho, "detach"):
        rho = rho.detach().cpu().numpy()
    else:
        rho = np.asarray(rho)
    return rho.astype(np.complex128)

def partial_transpose(rho, dims, sys):
    """
    rho: (D, D) density matrix
    dims: [d0, d1, ..., d_{N-1}] local dimensions for each subsystem
    sys: list of subsystem (qubit) indices to transpose (e.g. A1)
    """
    rho = _to_numpy(rho)
    N = len(dims)
    D = int(np.prod(dims))
    # print(rho.shape)
    assert rho.shape == (D, D)

    T = rho.reshape(dims + dims)  # indices: (i0..iN-1, j0..jN-1)
    for k in sys:
        # swap i_k <-> j_k
        T = T.swapaxes(k, N + k)
    return T.reshape(D, D)

def partial_trace(rho, keep, dims):
    """
    Compute Tr_{trace_out}(rho). 'keep' are the subsystem indices to keep.
    """
    rho = _to_numpy(rho)
    N = len(dims)
    D = int(np.prod(dims))
    # print(rho.shape)
    assert rho.shape == (D, D)

    keep = list(keep)
    trace_out = [i for i in range(N) if i not in keep]

    T = rho.reshape(dims + dims)
    # Trace out subsystems one by one.
    # Note: axes shift after each trace, so iterate from larger to smaller indices.
    for t in sorted(trace_out, reverse=True):
        T = np.trace(T, axis1=t, axis2=N + t)
        dims.pop(t)
        N -= 1
    D_keep = int(np.prod(dims)) if dims else 1
    return T.reshape(D_keep, D_keep)


def negativity_spectrum(rho, A1, A2=None, B=None, *, N=None, dims=None, tol=1e-12, log_base=2):
    """
    Returns: (eigvals, neg, log_neg)

    Notes:
    - If B is None: by default, A1 ∪ A2 is treated as the "kept" region and the rest as B.
      If you truly have "no B", set A1, A2 to cover a bipartition of the full system.
    - If you already have an N-qubit mixed state and do NOT want to trace anything out:
      set B = [] or make A1 ∪ A2 cover all qubits.
    """
    rho = _to_numpy(rho)
    D = rho.shape[0]
    # print(rho.shape)
    assert rho.shape == (D, D)

    if dims is None:
        if N is None:
            N = int(round(np.log2(D)))
        dims = [2] * N
    else:
        N = len(dims)

    A1 = list(A1)
    A2 = [] if A2 is None else list(A2)
    A = A1 + A2

    if B is None:
        B = [q for q in range(N) if q not in set(A)]
    else:
        B = list(B)

    # Physicality / numerical robustness fixes
    rho = (rho + rho.conj().T) / 2
    tr = np.trace(rho)
    if abs(tr) > 0:
        rho = rho / tr

    # 1) Get rho_A
    if len(B) == 0:
        rho_A = rho
        dims_A = dims
        A1_in_A = A1  # here A is the full system indices
    else:
        keep = A
        rho_A = partial_trace(rho, keep=keep, dims=dims.copy())
        dims_A = [dims[i] for i in keep]
        # Remap indices inside A to 0..|A|-1
        pos = {q: i for i, q in enumerate(keep)}
        A1_in_A = [pos[q] for q in A1]

    # 2) Partial transpose w.r.t. A1
    rho_A_pt = partial_transpose(rho_A, dims=dims_A, sys=A1_in_A)

    # 3) Negativity spectrum = eigenvalues of rho_A^{T_{A1}}
    eigvals = np.linalg.eigvalsh((rho_A_pt + rho_A_pt.conj().T) / 2).real
    eigvals[np.abs(eigvals) < tol] = 0.0

    neg = float(np.sum(np.abs(eigvals[eigvals < 0.0])))
    log_neg = float(np.log(2 * neg + 1) / np.log(log_base))
    return eigvals, neg, log_neg


def _robust_scale_from_gaps(x_sorted: torch.Tensor) -> torch.Tensor:
    """
    Return a robust characteristic scale for level spacings,
    defined as the median of strictly positive adjacent gaps.

    This scale is used to set adaptive tolerances for detecting
    (near-)degenerate levels.
    """
    if x_sorted.numel() < 2:
        return torch.tensor(0.0, device=x_sorted.device, dtype=x_sorted.dtype)
    dif = x_sorted[1:] - x_sorted[:-1]
    dif = dif[dif > 0]
    if dif.numel() == 0:
        return torch.tensor(0.0, device=x_sorted.device, dtype=x_sorted.dtype)
    return dif.median()


def _make_tol(scale: torch.Tensor, tol_rel: float, tol_abs: float) -> torch.Tensor:
    """
    Construct an absolute tolerance from a characteristic scale.

    The tolerance is defined as:
        tol = max(tol_abs, tol_rel * scale)

    If scale == 0, the tolerance falls back to tol_abs.
    """
    t = torch.tensor(tol_abs, device=scale.device, dtype=scale.dtype)
    if tol_rel is not None and tol_rel > 0 and torch.isfinite(scale) and scale > 0:
        t = torch.maximum(t, scale * tol_rel)
    return t


def _dedup_sorted_levels(
    x_sorted: torch.Tensor,
    tol: torch.Tensor,
    keep: Literal["first", "last", "mean"] = "first",
) -> torch.Tensor:
    """
    Deduplicate a sorted spectrum by merging adjacent values whose
    differences are smaller than or equal to `tol`.

    The input tensor must be sorted in ascending order.

    Parameters
    ----------
    x_sorted:
        Sorted 1D tensor of levels.
    tol:
        Absolute tolerance for identifying near-degenerate levels.
    keep:
        Strategy for choosing a representative value for each cluster:
        - "first": keep the first element in the cluster
        - "last":  keep the last element in the cluster
        - "mean":  keep the mean value of the cluster (numerically smoother)

    Returns
    -------
    torch.Tensor
        Deduplicated 1D tensor of representative levels.
    """
    n = x_sorted.numel()
    if n <= 1 or tol <= 0:
        return x_sorted

    dif = x_sorted[1:] - x_sorted[:-1]
    new_cluster = torch.ones((n,), device=x_sorted.device, dtype=torch.bool)
    new_cluster[1:] = dif > tol

    starts = torch.nonzero(new_cluster, as_tuple=False).flatten()
    ends = torch.cat([starts[1:] - 1, torch.tensor([n - 1], device=x_sorted.device)])

    reps = []
    for st, ed in zip(starts.tolist(), ends.tolist()):
        seg = x_sorted[st:ed + 1]
        if keep == "first":
            reps.append(seg[0])
        elif keep == "last":
            reps.append(seg[-1])
        elif keep == "mean":
            reps.append(seg.mean())
        else:
            raise ValueError("keep must be 'first', 'last', or 'mean'")
    return torch.stack(reps, dim=0)


@torch.no_grad()
def r_stats_from_gibbs_states(
    list_state: torch.Tensor,
    eps: float = 1e-12,
    bulk_keep: float = 0.8,
    enforce_hermitian: bool = True,
    renormalize_trace: bool = True,
    jitter: float = 0.0,
    bins: int = 60,
    device: Optional[str] = None,

    # ---- PSD / numerical stabilization ----
    psd_clip: bool = True,          # clip negative eigenvalues to zero and renormalize

    # ---- Remove near-zero eigenvalues (avoid log divergence) ----
    drop_small_eigs: bool = True,
    rank_tol_rel: float = 1e-12,    # keep λ > max(rank_tol_abs, rank_tol_rel * λ_max)
    rank_tol_abs: float = 0.0,

    # ---- Deduplication of (near-)degenerate levels ----
    dedup: bool = True,
    dedup_on: Literal["evals", "mu"] = "mu",
    dedup_keep: Literal["first", "last", "mean"] = "first",
    dedup_tol_rel: float = 1e-6,    # tol = max(abs, rel * median_positive_gap)
    dedup_tol_abs: float = 0.0,

    # ---- Optional removal of near-degenerate gaps ----
    drop_small_gaps: bool = False,
    gap_tol_rel: float = 1e-6,
    gap_tol_abs: float = 0.0,

    # ---- Minimum number of levels required to compute r ----
    min_levels: int = 3,

    # ---- Debug output ----
    return_debug: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Compute the level-spacing ratio statistics

        r_k = min(s_k, s_{k+1}) / max(s_k, s_{k+1}) ∈ [0, 1],

    where
        s_k = μ_{k+1} - μ_k,
        μ = -log(λ),
    and λ are the eigenvalues of the density matrix.

    Treatment of degeneracies:
    --------------------------
    If `dedup=True`, (near-)degenerate levels are merged into a single
    representative level before computing spacings. This suppresses
    artificial peaks near r ≈ 0 caused by exact or numerical degeneracies.
    """

    if device is None:
        device = list_state.device
    rho = list_state.to(device=device)

    if rho.ndim == 2:
        rho = rho.unsqueeze(0)
    assert rho.ndim == 3 and rho.shape[-1] == rho.shape[-2]

    # promote precision
    rho = rho.to(torch.complex128) if rho.is_complex() else rho.to(torch.float64)

    S, d, _ = rho.shape
    r_list = []
    r_per_state = torch.full((S,), float("nan"), device=device, dtype=torch.float64)

    # debug bookkeeping
    kept_levels = torch.zeros((S,), device=device, dtype=torch.int64)
    kept_levels_after_dedup = torch.zeros((S,), device=device, dtype=torch.int64)

    for s in range(S):
        M = rho[s]

        if enforce_hermitian:
            M = (M + M.conj().transpose(-1, -2)) / 2

        if renormalize_trace:
            tr = torch.real(torch.trace(M))
            if torch.isfinite(tr) and torch.abs(tr) > eps:
                M = M / tr

        # eigenvalues of rho (ascending order)
        evals = torch.real(torch.linalg.eigvalsh(M)).to(torch.float64)

        if psd_clip:
            evals = torch.clamp(evals, min=0.0)
            ssum = evals.sum()
            if ssum <= eps:
                continue
            evals = evals / ssum

        # remove near-zero eigenvalues to avoid log divergence
        if drop_small_eigs:
            lam_max = evals.max()
            thr = max(rank_tol_abs, float(lam_max) * rank_tol_rel)
            evals = evals[evals > thr]

        kept_levels[s] = int(evals.numel())
        if evals.numel() < min_levels:
            continue

        # ---- deduplicate either in λ-space or μ-space ----
        if dedup and dedup_on == "evals":
            evals_sorted, _ = torch.sort(evals)
            scale = _robust_scale_from_gaps(evals_sorted)
            tol = _make_tol(scale, dedup_tol_rel, dedup_tol_abs)
            evals_sorted = _dedup_sorted_levels(evals_sorted, tol, keep=dedup_keep)

            if evals_sorted.numel() < min_levels:
                kept_levels_after_dedup[s] = int(evals_sorted.numel())
                continue

            mu = -torch.log(evals_sorted)

        else:
            mu = -torch.log(evals)
            mu, _ = torch.sort(mu)

            if dedup and dedup_on == "mu":
                scale = _robust_scale_from_gaps(mu)
                tol = _make_tol(scale, dedup_tol_rel, dedup_tol_abs)
                mu = _dedup_sorted_levels(mu, tol, keep=dedup_keep)

        kept_levels_after_dedup[s] = int(mu.numel())
        if mu.numel() < min_levels:
            continue

        if jitter and jitter > 0:
            mu = mu + jitter * torch.randn_like(mu)

        # bulk spectrum selection
        if bulk_keep is not None and 0 < bulk_keep < 1 and mu.numel() >= 6:
            m = mu.numel()
            cut = int(((1.0 - bulk_keep) / 2.0) * m)
            mu = mu[cut: m - cut]
            if mu.numel() < min_levels:
                continue

        spac = mu[1:] - mu[:-1]
        if spac.numel() < 2:
            continue

        s1 = spac[:-1]
        s2 = spac[1:]

        # optionally discard near-degenerate gaps
        if drop_small_gaps:
            pos = spac[spac > 0]
            scale = pos.median() if pos.numel() > 0 else torch.tensor(
                0.0, device=device, dtype=spac.dtype
            )
            gtol = _make_tol(scale, gap_tol_rel, gap_tol_abs)
            if gtol > 0:
                good = (s1 > gtol) & (s2 > gtol)
                s1, s2 = s1[good], s2[good]
                if s1.numel() == 0:
                    continue

        denom = torch.maximum(s1, s2)
        numer = torch.minimum(s1, s2)
        r = torch.where(denom > 0, numer / denom, torch.zeros_like(denom))
        r = r[torch.isfinite(r)]
        if r.numel() == 0:
            continue

        r_list.append(r)
        r_per_state[s] = r.mean()

    edges = torch.linspace(0, 1, bins + 1, device=device, dtype=torch.float64)

    if len(r_list) == 0:
        out = {
            "r_all": torch.empty((0,), device=device, dtype=torch.float64),
            "r_mean": torch.tensor(float("nan"), device=device, dtype=torch.float64),
            "r_per_state": r_per_state,
            "hist_counts": torch.zeros((bins,), device=device, dtype=torch.float64),
            "hist_edges": edges,
        }
        if return_debug:
            out["levels_kept_per_state"] = kept_levels
            out["levels_after_dedup_per_state"] = kept_levels_after_dedup
        return out

    r_all = torch.cat(r_list, dim=0)

    # histogram (GPU-safe)
    idx = torch.bucketize(r_all, edges, right=False) - 1
    idx = torch.clamp(idx, 0, bins - 1)
    counts = torch.bincount(idx, minlength=bins).to(torch.float64)
    bin_width = edges[1] - edges[0]
    density = counts / (counts.sum() * bin_width + 1e-30)

    out = {
        "r_all": r_all,
        "r_mean": r_all.mean(),
        "r_per_state": r_per_state,
        "hist_counts": density,
        "hist_edges": edges,
    }
    if return_debug:
        out["levels_kept_per_state"] = kept_levels
        out["levels_after_dedup_per_state"] = kept_levels_after_dedup
    return out



def _logsumexp(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return -np.inf
    m = np.max(x)
    if not np.isfinite(m):
        return -np.inf
    return float(m + np.log(np.sum(np.exp(x - m))))


def _pauli_char_matrix(ch: str, dtype=np.complex128) -> np.ndarray:
    ch = ch.upper()
    if ch == "I":
        return np.array([[1, 0], [0, 1]], dtype=dtype)
    if ch == "X":
        return np.array([[0, 1], [1, 0]], dtype=dtype)
    if ch == "Y":
        return np.array([[0, -1j], [1j, 0]], dtype=dtype)
    if ch == "Z":
        return np.array([[1, 0], [0, -1]], dtype=dtype)
    raise ValueError(f"Invalid Pauli character: {ch!r}. Allowed: I,X,Y,Z")

def pauli_string_to_matrix(pauli: str, dtype=np.complex128) -> np.ndarray:
    s = str(pauli).strip().upper()
    if len(s) == 0:
        raise ValueError("Empty Pauli string.")
    mats = [_pauli_char_matrix(ch, dtype=dtype) for ch in s]
    M = mats[0]
    for A in mats[1:]:
        M = np.kron(M, A)
    return M

def build_pauli_ops(pauli_strings, dtype=np.complex128):
    pauli_strings = [str(p).strip().upper() for p in pauli_strings]
    if len(pauli_strings) == 0:
        raise ValueError("pauli_strings must be non-empty.")
    n = len(pauli_strings[0])
    for p in pauli_strings:
        if len(p) != n:
            raise ValueError("All Pauli strings must have the same length (n qubits).")
        for ch in p:
            if ch not in "IXYZ":
                raise ValueError(f"Bad Pauli string {p!r}: contains {ch!r}.")
    return [pauli_string_to_matrix(p, dtype=dtype) for p in pauli_strings]


def log_tr_expm_hermitian(H: np.ndarray, beta: float) -> float:
    evals = np.linalg.eigvalsh(H)  # real for Hermitian
    return _logsumexp(-beta * evals)


def find_pauli_index(pauli_list, term: str) -> int:
    t = str(term).strip().upper()
    pauli_norm = [str(p).strip().upper() for p in pauli_list]
    if t not in pauli_norm:
        raise ValueError(f"Pauli term {t!r} not found in hamiltonian.pauli_words.")
    return pauli_norm.index(t)


def grid_distribution_terms(sample_coef: np.ndarray, ix: int, iy: int, D: int, c_lim=None):
    coef = np.asarray(sample_coef)
    x = coef[:, ix].astype(float)
    y = coef[:, iy].astype(float)

    if c_lim is None:
        m = max(np.max(np.abs(x)), np.max(np.abs(y)))
        c_lim = max(0.2, 1.02 * float(m))

    x_edges = np.linspace(-c_lim, c_lim, D + 1)
    y_edges = np.linspace(-c_lim, c_lim, D + 1)

    H, xe, ye = np.histogram2d(x, y, bins=[x_edges, y_edges])  # shape (Dx, Dy)
    cx = 0.5 * (xe[:-1] + xe[1:])  # length Dx
    cy = 0.5 * (ye[:-1] + ye[1:])  # length Dy
    return H, cx, cy, xe, ye

def prepare_heat_from_samples_terms(sample_coef, ix: int, iy: int, D: int, sigma: float, c_lim=None):
    P, cx, cy, x_edges, y_edges = grid_distribution_terms(sample_coef, ix=ix, iy=iy, D=D, c_lim=c_lim)

    if sigma and sigma > 0:
        mask = P <= 0.0
        num = np.where(mask, 0.0, P)
        weights = (~mask).astype(float)

        P_num = gaussian_filter(num, sigma=sigma, mode="constant", cval=0.0)
        P_den = gaussian_filter(weights, sigma=sigma, mode="constant", cval=0.0)

        with np.errstate(divide="ignore", invalid="ignore"):
            P_smooth = np.divide(P_num, P_den, out=np.zeros_like(P_num), where=P_den > 1e-15)

        P_plot = np.where(P_den > 1e-15, P_smooth, P)
    else:
        P_plot = P

    return {
        "P_plot": P_plot.astype(float),
        "cX_unique": cx.astype(float),
        "cY_unique": cy.astype(float),
        "x_edges": x_edges.astype(float),
        "y_edges": y_edges.astype(float),
    }

def prepare_heat_from_gridP(P, cx, cy, x_edges, y_edges, sigma):
    P = np.asarray(P, float)

    if sigma and sigma > 0:
        mask = P <= 0.0
        num = np.where(mask, 0.0, P)
        weights = (~mask).astype(float)

        P_num = gaussian_filter(num, sigma=sigma, mode="constant", cval=0.0)
        P_den = gaussian_filter(weights, sigma=sigma, mode="constant", cval=0.0)

        with np.errstate(divide="ignore", invalid="ignore"):
            P_smooth = np.divide(P_num, P_den, out=np.zeros_like(P_num), where=P_den > 1e-15)

        P_plot = np.where(P_den > 1e-15, P_smooth, P)
    else:
        P_plot = P

    return {
        "P_plot": P_plot.astype(float),
        "cX_unique": np.asarray(cx, float),
        "cY_unique": np.asarray(cy, float),
        "x_edges": np.asarray(x_edges, float),
        "y_edges": np.asarray(y_edges, float),
    }

def empirical_1d_hist(sample_coef, idx: int, D: int, sigma_1d: float = 1.0, c_lim=None):
    x = np.asarray(sample_coef)[:, idx].astype(float)
    if c_lim is None:
        m = np.max(np.abs(x))
        c_lim = max(0.2, 1.02 * float(m))

    edges = np.linspace(-c_lim, c_lim, D + 1)
    h, ed = np.histogram(x, bins=edges)

    if sigma_1d and sigma_1d > 0:
        h = gaussian_filter1d(h.astype(float), sigma=sigma_1d, mode="constant", cval=0.0)

    s = h.sum()
    if s > 0:
        h = h / s

    centers = 0.5 * (ed[:-1] + ed[1:])
    return centers.astype(float), h.astype(float), ed.astype(float)

def crop_heat_nonzero(res, thresh_rel=0.0, pad=1):
    P = np.asarray(res["P_plot"], float)
    cx = np.asarray(res["cX_unique"], float)  # x-bin centers
    cy = np.asarray(res["cY_unique"], float)  # y-bin centers
    x_edges = np.asarray(res["x_edges"], float)
    y_edges = np.asarray(res["y_edges"], float)

    if P.size == 0:
        return res

    thresh = float(P.max()) * float(thresh_rel)
    mask = P > thresh
    if not np.any(mask):
        return res

    # P shape: (Dx, Dy). axis0 corresponds to x bins, axis1 corresponds to y bins.
    rows = np.where(mask.any(axis=1))[0]  # x indices
    cols = np.where(mask.any(axis=0))[0]  # y indices

    i0, i1 = int(rows[0]), int(rows[-1])
    j0, j1 = int(cols[0]), int(cols[-1])

    i0 = max(0, i0 - pad); i1 = min(P.shape[0] - 1, i1 + pad)
    j0 = max(0, j0 - pad); j1 = min(P.shape[1] - 1, j1 + pad)

    out = dict(res)
    out["P_plot"] = P[i0:i1+1, j0:j1+1]
    out["cX_unique"] = cx[i0:i1+1]
    out["cY_unique"] = cy[j0:j1+1]
    out["x_edges"] = x_edges[i0:i1+2]
    out["y_edges"] = y_edges[j0:j1+2]
    return out

def crop_slice_nonzero(axis_vals, p, thresh_rel=0.0, pad=1):
    p = np.asarray(p, float)
    a = np.asarray(axis_vals, float)

    thr = float(p.max()) * float(thresh_rel)
    idx = np.where(p > thr)[0]
    if idx.size == 0:
        return a, p

    i0, i1 = int(idx[0]), int(idx[-1])
    i0 = max(0, i0 - pad); i1 = min(len(p) - 1, i1 + pad)
    return a[i0:i1+1], p[i0:i1+1]


def sample_c_full_gaussian(L, M, N, xi, SigmaX, rng):
    SigmaX = np.asarray(SigmaX, float)
    if SigmaX.shape != (L, L):
        raise ValueError("SigmaX must be (L,L).")

    Cov = (xi * xi / float(N)) * SigmaX

    # Diagonal fast path
    if np.allclose(Cov, np.diag(np.diag(Cov))):
        std = np.sqrt(np.clip(np.diag(Cov), 0.0, np.inf))
        return rng.normal(0.0, std, size=(M, L)).astype(float)

    return rng.multivariate_normal(mean=np.zeros(L), cov=Cov, size=M).astype(float)

def build_H_from_ops_stack(c: np.ndarray, ops_stack: np.ndarray) -> np.ndarray:
    return np.tensordot(c.astype(float), ops_stack, axes=(0, 0))

def _compute_logW_threaded(C_full: np.ndarray, ops_stack: np.ndarray, beta: float) -> np.ndarray:
    r"""
    Compute logW[m] = log Tr exp(-beta * H(c_m)) using a thread pool.

    Notes:
        - Uses threads to avoid copying ops_stack into subprocesses.
        - If your BLAS (MKL/OpenBLAS) already uses many threads, too many workers can oversubscribe.
          Adjust MAX_WORKERS accordingly.
    """
    C_full = np.asarray(C_full, float)
    M = C_full.shape[0]
    logW = np.empty(M, dtype=float)

    def worker(m: int) -> float:
        H = build_H_from_ops_stack(C_full[m], ops_stack)
        return log_tr_expm_hermitian(H, beta=float(beta))

    if MAX_WORKERS <= 1:
        for m in range(M):
            logW[m] = worker(m)
        return logW

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for m, val in enumerate(ex.map(worker, range(M))):
            logW[m] = val

    return logW

def theory_mc_projected_hists(
    *,
    C_full,          # (M,L)
    logW,            # (M,)
    ix, iy,          # indices for 2D
    x_edges, y_edges,
    idx_slice,
    slice_edges,
):
    logW = np.asarray(logW, float)
    logW = logW - np.max(logW)
    w = np.exp(logW)

    H2d, _, _ = np.histogram2d(
        C_full[:, ix], C_full[:, iy],
        bins=[x_edges, y_edges],
        weights=w
    )
    s2d = H2d.sum()
    if s2d > 0:
        H2d = H2d / s2d

    h1d, _ = np.histogram(C_full[:, idx_slice], bins=slice_edges, weights=w)
    s1d = h1d.sum()
    if s1d > 0:
        h1d = h1d / s1d

    return H2d.astype(float), h1d.astype(float)


def compute_num_step(beta, row_key, error, xi):
    if row_key == "beta":
        return int(np.ceil((xi ** 2) * (beta ** 1) / (error ** (2 / 3))))
    if row_key == "beta2":
        return int(np.ceil((xi ** 2) * (beta ** 2) / (error ** (2 / 3))))
    if row_key == "beta3":
        return int(np.ceil((xi ** 2) * (beta ** 3) / (error ** (2 / 3))))
    raise ValueError(f"Unknown row_key: {row_key}")


def compare_terms_full_dim_mc(
    hamiltonian,
    *,
    term_x: str,
    term_y: str,
    slice_term: str,
    beta: float = 2.0,
    row_key: str = "beta2",
    error_level: int = 3,
    path: str,
    max_empirical_samples: int = 100000,
    D_heat: int = 80,
    sigma_heat: float = 1.0,
    D_slice: int = 160,
    sigma_slice: float = 1.0,
    M_theory: int = 400,
    seed: int = 0,
    crop_zeros: bool = True,
    crop_thresh_rel: float = 0.0,
    crop_pad: int = 1,
    log_color: bool = True,
    heat_floor: float = 1e-300,
    slice_floor: float = 1e-10,
):
    r"""
    Prepare plotting data for comparing:
        (A) empirical samples (from saved sample_coef)
    vs  (B) full-dimension theorem-style MC (importance sampling) projected onto:
        - a 2D marginal over (term_x, term_y)
        - a 1D marginal over (slice_term)

    Args:
        hamiltonian: Object with attributes:
            - pauli_words: list of Pauli strings (length L)
            - decompose_with_sites(): returns (h, ..., ...) where h has length L
        term_x: Pauli term used as x-axis in the 2D heatmap.
        term_y: Pauli term used as y-axis in the 2D heatmap.
        slice_term: Pauli term used for the 1D marginal slice.
        beta: Inverse temperature used in Tr exp(-beta H(c)).
        row_key: One of {"beta", "beta2", "beta3"} controlling num_step scaling.
        error_level: error = 10^(-error_level).
        path: .npz path that contains sample_coef.
        max_empirical_samples: Truncate empirical sample count to this number.
        D_heat: Number of bins along each axis in 2D heat.
        sigma_heat: Gaussian smoothing sigma for 2D heat.
        D_slice: Number of bins for 1D marginal.
        sigma_slice: Gaussian smoothing sigma for 1D marginal.
        M_theory: Number of full-dim MC samples.
        seed: RNG seed for theory sampling.
        crop_zeros: Crop near-zero regions in heat/slice to improve visualization.
        crop_thresh_rel: Relative threshold (w.r.t. max) for cropping.
        crop_pad: Padding (in bins) around the detected nonzero region.
        log_color: Suggest log color normalization for heatmaps.
        heat_floor: Lower bound used when building log-scale vmin.
        slice_floor: Lower bound used when plotting semilogy.

    Returns:
        out: dict containing:
            - meta: indices, xi, N, etc.
            - emp_heat / th_heat: {P_plot, edges, centers, extent}
            - emp_slice / th_slice: {centers, prob, prob_plot, edges}
            - color_scale: {log_color, vmin, vmax}
    """
    # ---- Hamiltonian info ----
    pauli_list = [str(p).strip().upper() for p in hamiltonian.pauli_words]
    h, _, _ = hamiltonian.decompose_with_sites()
    h = np.asarray(h, dtype=float)
    L = len(pauli_list)

    xi = float(np.sum(np.abs(h)))
    error = 10 ** (-int(error_level))
    N = compute_num_step(beta, row_key, error, xi=xi)

    # ---- indices ----
    ix = find_pauli_index(pauli_list, term_x)
    iy = find_pauli_index(pauli_list, term_y)
    iz = find_pauli_index(pauli_list, slice_term)

    # ---- load empirical samples ----
    data = np.load(path)
    sample_coef = np.asarray(data["sample_coef"])
    sample_coef = sample_coef[: int(max_empirical_samples)]

    if sample_coef.ndim != 2 or sample_coef.shape[1] != L:
        raise ValueError(f"sample_coef shape {sample_coef.shape} but need (num_samples, L={L}).")

    # ---- empirical 2D heat ----
    emp_heat = prepare_heat_from_samples_terms(
        sample_coef, ix=ix, iy=iy, D=D_heat, sigma=sigma_heat
    )
    P_emp = emp_heat["P_plot"].astype(float)
    s_emp = P_emp.sum()
    if s_emp > 0:
        P_emp /= s_emp
    emp_heat["P_plot"] = P_emp

    # ---- empirical 1D slice ----
    centers_e, p1d_e, edges_slice = empirical_1d_hist(
        sample_coef, idx=iz, D=D_slice, sigma_1d=sigma_slice
    )

    # ---- theory proposal covariance: p = abs(h)/xi ----
    # (Diagonal SigmaX = diag(p).)
    p_vec = np.abs(h) / float(xi) if xi != 0 else np.ones(L, dtype=float) / float(L)
    SigmaX = np.diag(p_vec.astype(float))

    rng = np.random.default_rng(seed)
    C_full = sample_c_full_gaussian(L=L, M=M_theory, N=N, xi=xi, SigmaX=SigmaX, rng=rng)

    # ---- build dense operator stack once ----
    ops = build_pauli_ops(pauli_list, dtype=np.complex128)
    ops_stack = np.stack(ops, axis=0)  # (L, dim, dim)

    # ---- parallel log weights ----
    logW = _compute_logW_threaded(C_full, ops_stack, beta=float(beta))

    # ---- project theory to (term_x, term_y) heat and slice_term 1D ----
    H2d_th, h1d_th = theory_mc_projected_hists(
        C_full=C_full,
        logW=logW,
        ix=ix, iy=iy,
        x_edges=emp_heat["x_edges"], y_edges=emp_heat["y_edges"],
        idx_slice=iz,
        slice_edges=edges_slice
    )

    # ---- smooth/normalize theory heat ----
    th_heat = prepare_heat_from_gridP(
        H2d_th,
        emp_heat["cX_unique"],
        emp_heat["cY_unique"],
        emp_heat["x_edges"],
        emp_heat["y_edges"],
        sigma=sigma_heat
    )
    P_th = th_heat["P_plot"].astype(float)
    s_th = P_th.sum()
    if s_th > 0:
        P_th /= s_th
    th_heat["P_plot"] = P_th

    # ---- smooth/normalize theory slice ----
    if sigma_slice and sigma_slice > 0:
        h1d_th = gaussian_filter1d(h1d_th.astype(float), sigma=sigma_slice, mode="constant", cval=0.0)
    s1 = h1d_th.sum()
    if s1 > 0:
        h1d_th = h1d_th / s1

    centers_t = 0.5 * (edges_slice[:-1] + edges_slice[1:])
    p1d_t = h1d_th.astype(float)

    # ---- crop near-zero regions ----
    if crop_zeros:
        emp_heat = crop_heat_nonzero(emp_heat, thresh_rel=crop_thresh_rel, pad=crop_pad)
        th_heat  = crop_heat_nonzero(th_heat,  thresh_rel=crop_thresh_rel, pad=crop_pad)
        centers_e, p1d_e = crop_slice_nonzero(centers_e, p1d_e, thresh_rel=crop_thresh_rel, pad=crop_pad)
        centers_t, p1d_t = crop_slice_nonzero(centers_t, p1d_t, thresh_rel=crop_thresh_rel, pad=crop_pad)

    # ---- extents for imshow ----
    emp_extent = [emp_heat["x_edges"][0], emp_heat["x_edges"][-1], emp_heat["y_edges"][0], emp_heat["y_edges"][-1]]
    th_extent  = [th_heat["x_edges"][0],  th_heat["x_edges"][-1],  th_heat["y_edges"][0],  th_heat["y_edges"][-1]]

    # ---- shared vmin/vmax suggestion ----
    Ppos = np.concatenate([
        emp_heat["P_plot"][emp_heat["P_plot"] > 0],
        th_heat["P_plot"][th_heat["P_plot"] > 0]
    ])
    if Ppos.size == 0:
        vmin = heat_floor
        vmax = heat_floor
    else:
        vmin = max(float(Ppos.min()), float(heat_floor))
        vmax = float(Ppos.max())

    emp_slice_plot = np.maximum(p1d_e.astype(float), float(slice_floor))
    th_slice_plot  = np.maximum(p1d_t.astype(float), float(slice_floor))

    out = {
        "meta": {
            "term_x": str(term_x).strip().upper(),
            "term_y": str(term_y).strip().upper(),
            "slice_term": str(slice_term).strip().upper(),
            "ix": int(ix), "iy": int(iy), "iz": int(iz),
            "L": int(L),
            "beta": float(beta),
            "row_key": str(row_key),
            "error_level": int(error_level),
            "error": float(error),
            "xi": float(xi),
            "N_num_step": int(N),
            "M_theory": int(M_theory),
            "seed": int(seed),
            "MAX_WORKERS": int(MAX_WORKERS),
        },
        "emp_heat": {**emp_heat, "extent": emp_extent},
        "th_heat":  {**th_heat,  "extent": th_extent},
        "emp_slice": {"centers": centers_e, "prob": p1d_e, "prob_plot": emp_slice_plot, "edges": edges_slice},
        "th_slice":  {"centers": centers_t, "prob": p1d_t, "prob_plot": th_slice_plot,  "edges": edges_slice},
        "color_scale": {"log_color": bool(log_color), "vmin": float(vmin), "vmax": float(vmax)},
    }
    return out
