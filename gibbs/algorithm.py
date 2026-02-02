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

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

import os
import time

import numpy as np
import quairkit as qkit
import torch
from quairkit import Hamiltonian, State, to_state
from quairkit.database import completely_mixed_computational, eye, x, y, z
from quairkit.qinfo import nkron, trace

__all__ = ["algorithm1", "grid_distribution"]

_GPU_CHUNK_CAP = 131072
_CPU_CHUNK_CAP = 65536
_SAFETY_RATIO = 0.95


@dataclass
class _AlgorithmContext:
    n: int
    beta: float
    dim: int
    device: torch.device
    state_dtype: torch.dtype
    real_dtype: torch.dtype
    state_elem_bytes: int
    real_elem_bytes: int
    num_step: int
    step_size: float
    tau: float
    list_pauli: List[torch.Tensor]
    list_pauli_expand: torch.Tensor
    list_pauli_prob: torch.Tensor
    pauli_site_ids: torch.Tensor
    site_positions: torch.Tensor
    site_operator_cache: List[torch.Tensor]
    site_id_to_site: List[List[int]]
    base_dm_init: torch.Tensor
    mix_dm: torch.Tensor
    eps_real: float

    @property
    def num_pauli(self) -> int:
        return len(self.list_pauli)


@dataclass
class _BatchPlan:
    total_sample: int
    per_sample_bytes: int
    mem_budget_bytes: Optional[int]
    chunks: List[int]
    need_split: bool

    @property
    def num_batches(self) -> int:
        return len(self.chunks)

    @property
    def max_chunk(self) -> int:
        return max(self.chunks) if self.chunks else 0


def _set_gpu(min_free_gb: float = 4.0, max_utilization_pct: int = 50) -> None:
    r'''
    Choose a CUDA device with enough free memory and low utilization when possible.
    Fall back to CPU if CUDA is unavailable or no suitable device is found.
    '''
    if not torch.cuda.is_available():
        qkit.set_device("cpu")
        return

    try:
        _efficient_pick(min_free_gb, max_utilization_pct)
        return
    except Exception:
        best_idx, best_free_gb = None, -1.0
        for i in range(torch.cuda.device_count()):
            try:
                free, _ = torch.cuda.mem_get_info(i)
            except TypeError:
                torch.cuda.set_device(i)
                free, _ = torch.cuda.mem_get_info()
            free_gb = free / (1024**3)
            if free_gb > best_free_gb:
                best_idx, best_free_gb = i, free_gb

        if best_idx is None or best_free_gb < min_free_gb:
            qkit.set_device("cpu")
            return

        qkit.set_device(f"cuda:{best_idx}")


def _efficient_pick(min_free_gb: float, max_utilization_pct: int) -> None:
    import pynvml as nvml

    nvml.nvmlInit()
    count = nvml.nvmlDeviceGetCount()

    candidates = []
    for idx in range(count):
        h = nvml.nvmlDeviceGetHandleByIndex(idx)
        mem = nvml.nvmlDeviceGetMemoryInfo(h)
        util = nvml.nvmlDeviceGetUtilizationRates(h).gpu
        free_gb = mem.free / (1024**3)

        procs = 0
        for name in (
            "nvmlDeviceGetComputeRunningProcesses_v3",
            "nvmlDeviceGetComputeRunningProcesses_v2",
            "nvmlDeviceGetComputeRunningProcesses",
        ):
            fn = getattr(nvml, name, None)
            if fn is None:
                continue
            try:
                procs = len(fn(h))
            except nvml.NVMLError:
                procs = 0
            break

        if free_gb >= min_free_gb and util <= max_utilization_pct:
            candidates.append((procs, util, -free_gb, idx))

    if candidates:
        candidates.sort()
        chosen = candidates[0][-1]
    else:
        freemem = []
        for idx in range(count):
            h = nvml.nvmlDeviceGetHandleByIndex(idx)
            mem = nvml.nvmlDeviceGetMemoryInfo(h)
            freemem.append((-mem.free, idx))
        chosen = sorted(freemem)[0][1]

    nvml.nvmlShutdown()
    qkit.set_device(f"cuda:{chosen}")


def _pauli_str_to_matrix(pauli_str: str) -> torch.Tensor:
    r'''
    Convert a Pauli-word string (e.g., "IXZ") to its full matrix via Kronecker products.
    '''
    mats = []
    for s in pauli_str:
        if s == "X":
            mats.append(x())
        elif s == "Y":
            mats.append(y())
        elif s == "Z":
            mats.append(z())
        else:
            mats.append(eye())
    return nkron(*mats)


def _decompose_hamiltonian(
    hamiltonian: Hamiltonian,
) -> Tuple[List[torch.Tensor], torch.Tensor, List[List[int]], torch.Tensor]:
    r'''
    Decompose the Hamiltonian into Pauli components, expanded components, site metadata, and bounds.
    '''
    list_bound, list_pauli_str_simplified, list_pauli_sites = hamiltonian.decompose_with_sites()
    list_pauli_str_expand = hamiltonian.pauli_words

    list_pauli = [_pauli_str_to_matrix(s) for s in list_pauli_str_simplified]
    list_pauli_expand = torch.stack([_pauli_str_to_matrix(s) for s in list_pauli_str_expand])
    list_bound = torch.abs(torch.tensor(list_bound, dtype=torch.float64))

    return list_pauli, list_pauli_expand, list_pauli_sites, list_bound


def _group_pauli_by_site(list_pauli_sites: List[List[int]]):
    r'''
    Group Pauli terms by the lattice sites they act on, and build index maps.
    '''
    site_map = {}
    site_id_to_site: List[List[int]] = []
    site_pauli_lists: List[List[int]] = []
    pauli_site_ids = torch.empty(len(list_pauli_sites), dtype=torch.long)
    site_positions = torch.empty(len(list_pauli_sites), dtype=torch.long)

    for idx, site in enumerate(list_pauli_sites):
        key = tuple(site)
        if key not in site_map:
            site_map[key] = len(site_id_to_site)
            site_id_to_site.append(list(site))
            site_pauli_lists.append([])
        site_id = site_map[key]
        site_positions[idx] = len(site_pauli_lists[site_id])
        site_pauli_lists[site_id].append(idx)
        pauli_site_ids[idx] = site_id

    return pauli_site_ids, site_positions, site_id_to_site, site_pauli_lists


def _current_device() -> torch.device:
    r'''
    Normalize QuAIRKit's device specification to a torch.device.
    '''
    dev = qkit.get_device()
    return dev if isinstance(dev, torch.device) else torch.device(dev)


def _cuda_mem_get_info(device: torch.device) -> Tuple[int, int]:
    r'''
    Get CUDA free/total bytes for a specific device, compatible across PyTorch versions.
    '''
    if device.type != "cuda":
        raise ValueError("device must be a CUDA device")
    try:
        return torch.cuda.mem_get_info(device)
    except Exception:
        current = torch.cuda.current_device()
        try:
            torch.cuda.set_device(device)
            info = torch.cuda.mem_get_info()
        finally:
            torch.cuda.set_device(current)
        return info


def _to_torch(init_state, device: torch.device) -> torch.Tensor:
    r'''
    Convert init_state to a torch.Tensor on the specified device.

    Accepted:
      - torch.Tensor
      - numpy.ndarray
      - quairkit State / MixedState (or any object with `.density_matrix`)
    '''
    if isinstance(init_state, torch.Tensor):
        t = init_state
    elif isinstance(init_state, np.ndarray):
        t = torch.from_numpy(init_state)
    else:
        if hasattr(init_state, "density_matrix"):
            t = init_state.density_matrix
        else:
            raise TypeError(
                "init_state must be torch.Tensor / np.ndarray / quairkit State(MixedState), "
                "or an object with `.density_matrix`."
            )
    return t.to(device=device)


def _normalize_init_state(
    init_state,
    mix_dm: torch.Tensor,
    state_dtype: torch.dtype,
    eps_real: float,
) -> torch.Tensor:
    r'''
    Build the single initial density matrix (shape [1, dim, dim]) used for all samples.

    Accepted init_state formats:
      - None -> I/d
      - density matrix: [dim, dim]
      - statevector (ket): [dim]
      - quairkit State/MixedState (uses `.density_matrix`)
    '''
    device = mix_dm.device
    dim = mix_dm.shape[-1]

    if init_state is None:
        base = mix_dm.clone()
    else:
        t = _to_torch(init_state, device=device)

        if t.dtype not in (torch.complex64, torch.complex128):
            t = t.to(torch.complex128)

        if t.dim() == 1:
            if t.numel() != dim:
                raise ValueError(f"init_state ket dim mismatch: got {t.numel()}, expected {dim}")
            psi = t.view(1, dim)
            norm = torch.linalg.norm(psi, dim=-1).clamp_min(eps_real)
            psi = psi / norm[:, None]
            base = psi[:, :, None] * psi.conj()[:, None, :]
        elif t.dim() == 2:
            if t.shape != (dim, dim):
                raise ValueError(f"init_state density dim mismatch: got {tuple(t.shape)}, expected {(dim, dim)}")
            base = t.unsqueeze(0)
        else:
            raise ValueError(f"init_state rank not supported: got t.dim()={t.dim()}")

    base = (base + base.conj().transpose(-2, -1)) * 0.5
    tr = trace(base, -2, -1).real.clamp_min(eps_real)
    base = base / tr.view(-1, 1, 1)

    return base.to(dtype=state_dtype, device=device).contiguous()


def _prepare_algorithm_context(
    n: int,
    beta: float,
    hamiltonian: Hamiltonian,
    error: float,
    num_step: Optional[int],
    init_state=None,
) -> _AlgorithmContext:
    r'''
    Precompute Hamiltonian decomposition, sampling probabilities, and cached local ITE operators.
    '''
    device = _current_device()
    state_dtype = qkit.get_dtype()
    real_dtype = torch.float64 if state_dtype in (torch.complex128, torch.float64) else torch.float32
    eps_real = torch.finfo(real_dtype).tiny

    list_pauli, list_pauli_expand, list_pauli_sites, list_bound = _decompose_hamiltonian(hamiltonian)
    list_pauli = [p.to(device=device, dtype=state_dtype) for p in list_pauli]
    list_pauli_expand = list_pauli_expand.to(device=device, dtype=state_dtype)
    list_bound = list_bound.to(device=device, dtype=real_dtype)

    Lambda = list_bound.sum().clamp_min(eps_real)
    list_pauli_prob = (list_bound / Lambda).clamp_min(eps_real)
    list_pauli_prob = (list_pauli_prob / list_pauli_prob.sum()).contiguous()

    if num_step is None:
        num_step = max(int(np.ceil((beta**2) / (error ** (2 / 3)))), 10)
    num_step = max(int(num_step), 1)

    step_size = Lambda.item() / num_step
    tau = step_size * beta

    pauli_site_ids, site_positions, site_id_to_site, site_pauli_lists = _group_pauli_by_site(list_pauli_sites)
    pauli_site_ids = pauli_site_ids.to(device)
    site_positions = site_positions.to(device)

    scales = torch.tensor([-tau / 2, tau / 2], dtype=state_dtype, device=device).view(1, 2, 1, 1)

    site_operator_cache: List[torch.Tensor] = []
    for idx_list in site_pauli_lists:
        local_pauli = torch.stack([list_pauli[i] for i in idx_list], dim=0)
        ite_pairs_local = torch.matrix_exp(local_pauli.unsqueeze(1) * scales)
        site_operator_cache.append(ite_pairs_local.contiguous())

    mix_state = completely_mixed_computational(n).to(device=device, dtype=state_dtype)
    mix_dm = mix_state.density_matrix
    if mix_dm.dim() == 2:
        mix_dm = mix_dm.unsqueeze(0)
    dim = mix_dm.shape[-1]

    base_dm_init = _normalize_init_state(
        init_state=init_state,
        mix_dm=mix_dm,
        state_dtype=state_dtype,
        eps_real=eps_real,
    )

    state_elem_bytes = torch.zeros((), dtype=state_dtype).element_size()
    real_elem_bytes = torch.zeros((), dtype=real_dtype).element_size()

    return _AlgorithmContext(
        n=n,
        beta=beta,
        dim=dim,
        device=device,
        state_dtype=state_dtype,
        real_dtype=real_dtype,
        state_elem_bytes=state_elem_bytes,
        real_elem_bytes=real_elem_bytes,
        num_step=num_step,
        step_size=step_size,
        tau=tau,
        list_pauli=list_pauli,
        list_pauli_expand=list_pauli_expand,
        list_pauli_prob=list_pauli_prob,
        pauli_site_ids=pauli_site_ids,
        site_positions=site_positions,
        site_operator_cache=site_operator_cache,
        site_id_to_site=site_id_to_site,
        base_dm_init=base_dm_init,
        mix_dm=mix_dm,
        eps_real=eps_real,
    )


def _estimate_per_sample_bytes(ctx: _AlgorithmContext) -> int:
    r'''
    Estimate per-sample memory use for planning chunk sizes (heuristic).
    '''
    dim_sq = ctx.dim * ctx.dim
    base_bytes = dim_sq * ctx.state_elem_bytes
    base_bytes += ctx.num_pauli * ctx.real_elem_bytes
    base_bytes += ctx.real_elem_bytes
    overhead = 6.0 + 0.04 * ctx.num_step
    overhead = max(8.0, min(overhead, 24.0))
    estimate = int(math.ceil(base_bytes * overhead))
    return max(estimate, ctx.state_elem_bytes)


def _plan_batch_schedule(num_sample: int, ctx: _AlgorithmContext) -> _BatchPlan:
    r'''
    Plan batch chunk sizes for sampling, using CUDA free-memory heuristics when on GPU.
    '''
    if num_sample <= 0:
        raise ValueError("num_sample must be positive")

    per_sample = _estimate_per_sample_bytes(ctx)
    chunks: List[int] = []
    mem_budget: Optional[int] = None
    max_chunk = num_sample
    need_split = False

    if ctx.device.type == "cuda":
        free_mem, _ = _cuda_mem_get_info(ctx.device)
        mem_budget = int(free_mem * _SAFETY_RATIO)
        if per_sample > 0:
            max_chunk = max(1, mem_budget // per_sample)
        max_chunk = min(max_chunk, _GPU_CHUNK_CAP)
        need_split = (per_sample * num_sample > mem_budget) or (num_sample > max_chunk)
    else:
        max_chunk = min(num_sample, _CPU_CHUNK_CAP)
        need_split = num_sample > max_chunk

    max_chunk = max(1, int(max_chunk))
    remaining = num_sample
    while remaining > 0:
        chunk = min(max_chunk, remaining)
        chunks.append(chunk)
        remaining -= chunk

    return _BatchPlan(
        total_sample=num_sample,
        per_sample_bytes=per_sample,
        mem_budget_bytes=mem_budget,
        chunks=chunks,
        need_split=need_split or len(chunks) > 1,
    )


def _log_batch_plan(plan: _BatchPlan, ctx: _AlgorithmContext) -> None:
    r'''
    Print a concise summary of the batch plan for debugging/monitoring.
    '''
    budget_str = "N/A"
    if plan.mem_budget_bytes is not None:
        budget_str = f"{plan.mem_budget_bytes / (1024**3):.2f} GB"
    per_sample_kb = plan.per_sample_bytes / 1024
    split_flag = "yes" if plan.need_split else "no"
    print(
        "[algorithm1] batch-plan | device="
        f"{ctx.device} | budget={budget_str} | per-sample≈{per_sample_kb:.1f} KB | "
        f"planned_batches={plan.num_batches} | split_required={split_flag}"
    )


def _is_cuda_oom(err: RuntimeError) -> bool:
    r'''
    Check whether a RuntimeError looks like a CUDA out-of-memory failure.
    '''
    msg = str(err).lower()
    return ("out of memory" in msg) and ("cuda" in msg or "cublas" in msg or "cudnn" in msg)


def _execute_batch_plan(ctx: _AlgorithmContext, plan: _BatchPlan):
    r'''
    Execute sampling over (possibly) multiple chunks; move results to CPU per chunk to avoid accumulation.
    '''
    rho_chunks: List[torch.Tensor] = []
    coef_chunks: List[torch.Tensor] = []
    log_prob_chunks: List[torch.Tensor] = []

    queue = deque(plan.chunks)
    processed = executed = max_seen = 0

    while processed < plan.total_sample:
        remaining = plan.total_sample - processed
        if not queue:
            queue.append(min(plan.max_chunk or remaining, remaining))
        batch_size = min(queue.popleft(), remaining)

        try:
            rho_batch, coef_batch, log_prob_batch = _run_sampling_batch(ctx, batch_size)
        except RuntimeError as err:
            if not (_is_cuda_oom(err) and ctx.device.type == "cuda"):
                raise
            torch.cuda.empty_cache()
            if batch_size == 1:
                raise
            a = batch_size // 2
            b = batch_size - a
            queue.appendleft(b)
            queue.appendleft(a)
            print(f"[algorithm1] OOM fallback | batch={batch_size} -> {a}+{b}")
            continue

        rho_chunks.append(rho_batch.detach().cpu())
        coef_chunks.append(coef_batch.detach().cpu())
        log_prob_chunks.append(log_prob_batch.detach().cpu())

        del rho_batch, coef_batch, log_prob_batch

        processed += batch_size
        executed += 1
        max_seen = max(max_seen, batch_size)

    rho_dm = torch.cat(rho_chunks, dim=0)
    list_coef = torch.cat(coef_chunks, dim=0)
    log_prob = torch.cat(log_prob_chunks, dim=0)

    stats = dict(
        planned_batches=plan.num_batches,
        executed_batches=executed,
        max_batch_seen=max_seen,
        per_sample_bytes=plan.per_sample_bytes,
        need_split=plan.need_split,
    )
    return rho_dm, list_coef, log_prob, stats


def _suggest_finalize_chunk_size(total_sample: int, ctx: _AlgorithmContext, hard_cap: int = 64) -> int:
    r'''
    Suggest a chunk size for the finalize stage (matrix_exp + eigvalsh), based on available GPU memory.
    '''
    if total_sample <= 0:
        return 0
    if ctx.device.type != "cuda":
        return 1

    try:
        free_mem, _ = _cuda_mem_get_info(ctx.device)
    except Exception:
        return 1

    budget = max(int(free_mem * 0.30), 1)
    per = int(ctx.dim * ctx.dim * ctx.state_elem_bytes * 12)
    if per <= 0:
        return 1
    est = budget // per
    return max(1, min(int(est), total_sample, hard_cap))


def _trace_distance_chunked(
    rho: torch.Tensor,
    sigma: torch.Tensor,
    ctx: _AlgorithmContext,
    chunk_hint: Optional[int] = None,
) -> torch.Tensor:
    r'''
    Compute trace distances D(rho_i, sigma_i) for a batch, using chunked Hermitian eigendecompositions.
    '''
    if rho.shape != sigma.shape:
        raise ValueError(f"shape mismatch: rho{rho.shape} vs sigma{sigma.shape}")
    num_sample = rho.shape[0]
    if num_sample == 0:
        return torch.empty(0, dtype=ctx.real_dtype, device=ctx.device)

    chunk = max(1, int(chunk_hint or 1))
    distances: List[torch.Tensor] = []
    start = 0

    while start < num_sample:
        end = min(start + chunk, num_sample)
        block = rho[start:end] - sigma[start:end]
        try:
            eigvals = torch.linalg.eigvalsh(block)
            dist_block = 0.5 * torch.sum(torch.abs(eigvals), dim=-1)
            distances.append(dist_block.to(dtype=ctx.real_dtype))
            start = end
        except RuntimeError as err:
            if ctx.device.type == "cuda" and "cusolver" in str(err).lower():
                if chunk > 1:
                    chunk = max(1, chunk // 2)
                    print(f"[algorithm1] trace-distance chunk fallback: chunk_size={chunk}")
                    continue
                block_cpu = block.detach().cpu()
                eigvals = torch.linalg.eigvalsh(block_cpu)
                dist_block = 0.5 * torch.sum(torch.abs(eigvals), dim=-1)
                distances.append(dist_block.to(ctx.real_dtype).to(ctx.device))
                start = end
            else:
                raise

    return torch.cat(distances, dim=0)


def _run_sampling_batch(ctx: _AlgorithmContext, batch_size: int):
    r'''
    Run the stochastic ITE sampling for a single batch (size = batch_size) on ctx.device.

    This assumes a single init_state is used for all samples; the batch is initialized by cloning
    the same base density matrix.
    '''
    rho_dm = ctx.base_dm_init.expand(batch_size, -1, -1).clone()
    list_coef = torch.zeros((batch_size, ctx.num_pauli), dtype=ctx.real_dtype, device=ctx.device)
    log_prob = torch.zeros(batch_size, dtype=ctx.real_dtype, device=ctx.device)

    site_micro_cap = 4096 if ctx.device.type == "cuda" else batch_size

    for _ in range(ctx.num_step):
        pauli_choice = torch.multinomial(ctx.list_pauli_prob, batch_size, replacement=True)
        site_labels = ctx.pauli_site_ids.index_select(0, pauli_choice)

        site_sorted, perm = torch.sort(site_labels)
        pauli_sorted = pauli_choice.index_select(0, perm)

        unique_sites, counts = torch.unique_consecutive(site_sorted, return_counts=True)
        offset = 0

        for site_label, count in zip(unique_sites.tolist(), counts.tolist()):
            if count <= 0:
                continue

            end = offset + count
            batch_ids_full = perm[offset:end]
            pauli_subset_full = pauli_sorted[offset:end]
            offset = end

            sub_start = 0
            while sub_start < count:
                sub_end = min(sub_start + site_micro_cap, count)
                batch_ids = batch_ids_full[sub_start:sub_end]
                pauli_subset = pauli_subset_full[sub_start:sub_end]

                positions = ctx.site_positions.index_select(0, pauli_subset)
                ops_cache = ctx.site_operator_cache[int(site_label)]
                ops = ops_cache.index_select(0, positions).contiguous()
                d_loc = ops.shape[-1]

                subset_dm = rho_dm.index_select(0, batch_ids)
                subset_pairs = subset_dm.unsqueeze(1).expand(-1, 2, -1, -1).contiguous()
                subset_pairs_flat = subset_pairs.view(-1, ctx.dim, ctx.dim)

                subset_state = to_state(subset_pairs_flat, eps=None)
                evolved = subset_state.evolve(ops.view(-1, d_loc, d_loc), ctx.site_id_to_site[int(site_label)])
                evolved_dm = evolved.density_matrix.view(batch_ids.numel(), 2, ctx.dim, ctx.dim)

                weights = trace(evolved_dm, -2, -1).real.clamp_min(ctx.eps_real)
                prob_pair = (weights / weights.sum(dim=1, keepdim=True).clamp_min(ctx.eps_real)).clamp_min(ctx.eps_real)

                direction = torch.multinomial(prob_pair, 1).squeeze(-1)
                local_rows = torch.arange(batch_ids.numel(), device=ctx.device)

                selected_dm = evolved_dm[local_rows, direction]
                selected_norm = weights[local_rows, direction].clamp_min(ctx.eps_real)
                normalized_dm = selected_dm / selected_norm.view(-1, 1, 1)
                rho_dm.index_copy_(0, batch_ids, normalized_dm)

                sign = (1.0 - 2.0 * direction.to(ctx.real_dtype))
                list_coef[batch_ids, pauli_subset] += sign * ctx.step_size

                branch_prob = prob_pair[local_rows, direction].clamp_min(ctx.eps_real)
                pauli_prob = ctx.list_pauli_prob.index_select(0, pauli_subset).clamp_min(ctx.eps_real)
                log_prob[batch_ids] += torch.log(pauli_prob * branch_prob)

                del subset_dm, subset_pairs, subset_pairs_flat, subset_state, evolved, evolved_dm
                sub_start = sub_end

    return rho_dm, list_coef, log_prob


def _finalize_outputs(
    ctx: _AlgorithmContext, rho_dm_cpu: torch.Tensor, list_coef_cpu: torch.Tensor, log_prob_cpu: torch.Tensor
):
    r'''
    Finalize outputs:
      - Construct the Gibbs-map target from coefficients and the fixed init_state (rho0),
      - Compute per-sample trace-distance errors,
      - Return CPU outputs for downstream analysis.
    '''
    num = rho_dm_cpu.shape[0]
    list_error_cpu = torch.empty(num, dtype=ctx.real_dtype, device="cpu")
    dist_to_mix_cpu = torch.empty(num, dtype=ctx.real_dtype, device="cpu")

    rho0_cpu = ctx.base_dm_init.detach().cpu().expand(num, -1, -1).contiguous()

    chunk = _suggest_finalize_chunk_size(num, ctx)
    start = 0
    while start < num:
        end = min(start + chunk, num)
        sl = slice(start, end)

        rho = rho_dm_cpu[sl].to(ctx.device, dtype=ctx.state_dtype)
        rho0 = rho0_cpu[sl].to(ctx.device, dtype=ctx.state_dtype)
        coef = list_coef_cpu[sl].to(ctx.device, dtype=ctx.real_dtype)

        H = torch.einsum("bk,kij->bij", coef.to(ctx.state_dtype), ctx.list_pauli_expand)
        K = torch.matrix_exp((-0.5 * ctx.beta) * H)

        target = K @ rho0 @ K.conj().transpose(-2, -1)
        target = (target + target.conj().transpose(-2, -1)) * 0.5

        tr = trace(target, -2, -1).real.clamp_min(ctx.eps_real)
        target = target / tr.view(-1, 1, 1)

        err = _trace_distance_chunked(rho, target, ctx, chunk_hint=1)
        mix = ctx.mix_dm.expand(rho.shape[0], -1, -1)
        dmx = _trace_distance_chunked(rho, mix, ctx, chunk_hint=1)

        list_error_cpu[sl] = err.detach().cpu()
        dist_to_mix_cpu[sl] = dmx.detach().cpu()

        del rho, rho0, coef, H, K, target, err, dmx, mix
        start = end

    list_prob_cpu = log_prob_cpu.exp()
    qkit.set_device("cpu")
    list_state = to_state(rho_dm_cpu, eps=None)

    coef_abs_sum = torch.sum(torch.abs(list_coef_cpu), dim=-1)

    metrics = dict(
        num_step=ctx.num_step,
        tau=ctx.tau,
        max_error=float(list_error_cpu.max().item()),
        max_dist_mix=float(dist_to_mix_cpu.max().item()),
        max_coef_abs_sum=float(coef_abs_sum.max().item()),
    )

    return (
        list_state,
        list_coef_cpu.numpy(),
        list_error_cpu.numpy(),
        list_prob_cpu.numpy(),
        metrics,
    )

def _state_to_density_numpy(list_state) -> np.ndarray:
    r"""
    Convert a returned state container to a NumPy density-matrix batch [B, d, d].

    Supports:
      - quairkit State/MixedState with `.density_matrix` (batched)
      - torch.Tensor (assumed [B,d,d] or [d,d])
      - numpy.ndarray (assumed [B,d,d] or [d,d])
      - list/tuple of quairkit states (each has `.density_matrix`)
    """
    if hasattr(list_state, "density_matrix"):
        rho = list_state.density_matrix
    elif isinstance(list_state, (list, tuple)):
        if len(list_state) == 0:
            raise ValueError("list_state is an empty list/tuple")
        mats = []
        for i, s in enumerate(list_state):
            if not hasattr(s, "density_matrix"):
                raise TypeError(f"Element {i} has no .density_matrix: {type(s)}")
            dm = s.density_matrix
            if isinstance(dm, torch.Tensor):
                dm = dm.detach().cpu().numpy()
            elif not isinstance(dm, np.ndarray):
                dm = np.asarray(dm)
            mats.append(dm)
        rho = np.stack(mats, axis=0)
    elif isinstance(list_state, torch.Tensor):
        rho = list_state.detach().cpu().numpy()
    elif isinstance(list_state, np.ndarray):
        rho = list_state
    else:
        raise TypeError(f"Unsupported list_state type: {type(list_state)}")

    if rho.ndim == 2:
        rho = rho[None, :, :]
    if rho.ndim != 3 or rho.shape[-1] != rho.shape[-2]:
        raise ValueError(f"Density matrix must have shape [B,d,d] or [d,d], got {rho.shape}")

    if not np.iscomplexobj(rho):
        rho = rho.astype(np.complex128, copy=False)
    return rho


def _save_algorithm1_npz(
    path: str,
    list_state: State,
    sample_coef: np.ndarray,
    sample_error: np.ndarray,
    list_prob: np.ndarray,
    meta: dict,
    save_state: bool = True,
    compressed: bool = True,
) -> None:
    r"""
    Save algorithm1 outputs to a .npz file.

    Notes:
      - If save_state=True, store `rho` as a NumPy complex array [B,d,d].
      - Always stores coef/error/prob and meta fields.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    payload = dict(
        sample_coef=np.asarray(sample_coef),
        sample_error=np.asarray(sample_error),
        sample_prob=np.asarray(list_prob),
        **meta,
    )

    if save_state:
        payload["rho"] = _state_to_density_numpy(list_state)

    if compressed:
        np.savez_compressed(path, **payload)
    else:
        np.savez(path, **payload)


def algorithm1(
    n: int,
    beta: float,
    hamiltonian: Hamiltonian,
    error: float = 1e-2,
    num_sample: int = 1,
    num_step: Optional[int] = None,
    init_state=None,
    # ---- new: saving options ----
    save_dir: Optional[str] = None,
    save_tag: Optional[str] = None,
    save_state: bool = True,
    save_compressed: bool = True,
) -> Tuple[State, np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Sample approximate Gibbs states via a stochastic imaginary-time evolution scheme.

    Saving:
      - If save_dir is not None, this function will save outputs to a .npz file automatically.
      - The .npz includes: sample_coef, sample_error, sample_prob, and optionally rho (density matrices).
    """
    _set_gpu()
    ctx = _prepare_algorithm_context(n, beta, hamiltonian, error, num_step, init_state=init_state)
    batch_plan = _plan_batch_schedule(num_sample, ctx)
    _log_batch_plan(batch_plan, ctx)

    rho_dm, list_coef, log_prob, batch_stats = _execute_batch_plan(ctx, batch_plan)
    list_state, coef_np, error_np, prob_np, metrics = _finalize_outputs(ctx, rho_dm, list_coef, log_prob)

    print(
        f"N: {metrics['num_step']}, tau: {metrics['tau']:.2E}, "
        f"(max) error: {metrics['max_error']:.2E}, "
        f"(max) D(rho, I): {metrics['max_dist_mix']:.3f}, "
        f"(max) coef abs sum: {metrics['max_coef_abs_sum']:.3f}, "
        f"batches: {batch_stats['executed_batches']} (planned {batch_stats['planned_batches']})"
    )

    if ctx.device.type == "cuda":
        torch.cuda.empty_cache()
    qkit.set_device("cpu")

    # ---- save inside algorithm1 ----
    if save_dir is not None:
        # Make filenames stable & filesystem-safe
        beta_tag = f"{float(beta):.6g}"
        err_tag = f"{float(error):.6g}"
        tag = save_tag if (save_tag is not None and len(save_tag) > 0) else "run"

        out_path = os.path.join(
            save_dir,
            f"{tag}_b{beta_tag}_e{err_tag}_sample{int(num_sample)}_step{int(metrics['num_step'])}.npz",
        )


        meta = dict(
            beta=float(beta),
            error=float(error),
            num_sample=int(num_sample),
            num_step=int(metrics["num_step"]),
            n=int(n),
            tau=float(metrics["tau"]),
            max_error=float(metrics["max_error"]),
            max_dist_mix=float(metrics["max_dist_mix"]),
            max_coef_abs_sum=float(metrics["max_coef_abs_sum"]),
            planned_batches=int(batch_stats["planned_batches"]),
            executed_batches=int(batch_stats["executed_batches"]),
            per_sample_bytes=int(batch_stats["per_sample_bytes"]),
        )

        _save_algorithm1_npz(
            path=out_path,
            list_state=list_state,
            sample_coef=coef_np,
            sample_error=error_np,
            list_prob=prob_np,
            meta=meta,
            save_state=save_state,
            compressed=save_compressed,
        )
        print(f"[algorithm1] saved -> {out_path} (state_saved={save_state}, compressed={save_compressed})")

    return list_state, coef_np, error_np, prob_np



def grid_distribution(list_coef: np.ndarray, D: int) -> np.ndarray:
    r'''
    Build a 2D/3D histogram of sampled coefficients on a regular grid.

    Input:
      - list_coef: shape [num_sample, L] where L is 2 or 3
      - D: number of bins per dimension

    Output:
      - array of shape [D^L, L+1] containing (grid coordinates, probability)
    '''
    _, L = list_coef.shape
    if not (2 <= L <= 3):
        raise ValueError(f"only 2D or 3D distributions are supported; got L={L}")

    num_decimal = 2
    bounds = np.round(np.max(np.abs(list_coef), axis=0).reshape(-1) + 10 ** (-num_decimal - 1), decimals=num_decimal)

    edges = [np.linspace(-b, b, D + 1) for b in bounds]
    centers = [(e[:-1] + e[1:]) / 2.0 for e in edges]

    idx_per_dim = []
    for j in range(L):
        e = edges[j]
        xj = list_coef[:, j]
        idx = np.searchsorted(e, xj, side="right") - 1
        idx = np.clip(idx, 0, D - 1)
        idx_per_dim.append(idx)

    idx_flat = np.ravel_multi_index(idx_per_dim, dims=(D,) * L)
    counts = np.bincount(idx_flat, minlength=D**L).astype(float)

    total = counts.sum()
    if total <= 0:
        raise RuntimeError("no samples; cannot build histogram")
    p = counts / total

    meshes = np.meshgrid(*centers, indexing="ij")
    coords = np.stack([m.reshape(-1) for m in meshes], axis=1)

    return np.concatenate([coords, p[:, None]], axis=1)
