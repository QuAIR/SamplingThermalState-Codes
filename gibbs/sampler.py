import logging
import os
import sys
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, List, Optional, Tuple

import numpy as np
import quairkit as qkit
import torch
from quairkit import Hamiltonian, to_state
from quairkit.database import *
from quairkit.qinfo import *

# Assuming algorithm.py exists in the same directory as per the original notebook
from .algorithm import algorithm1
from .property import pauli_str_to_matrix

__all__ = ["ThermalStateSampler"]

class StreamToLogger:
    """
    File-like object that redirects writes to a logger (line buffered).

    This class is used to capture stdout/stderr and send it to a log file.
    """

    def __init__(self, logger: logging.Logger, level: int = logging.INFO):
        """
        Args:
            logger: The logging.Logger instance to write to.
            level: The logging level to use (e.g., logging.INFO, logging.ERROR).
        """
        self.logger = logger
        self.level = level
        self._buf = ""

    def write(self, msg: str):
        """
        Writes a message to the buffer and logs complete lines.

        Args:
            msg: The string message to write.
        """
        # Many libs call .write() in chunks; buffer until newline.
        self._buf += msg
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            # Remove carriage returns if present
            if line := line.rstrip("\r"):
                self.logger.log(self.level, line)

    def flush(self):
        """Flushes any remaining data in the buffer to the logger."""
        if leftover := self._buf.strip("\r\n"):
            self.logger.log(self.level, leftover)
        self._buf = ""


def make_logger(log_file: str = "run.log") -> logging.Logger:
    """
    Creates and configures a logger that writes to a file.

    Args:
        log_file: The name of the file to save logs to. Defaults to "run.log".

    Returns:
        A configured logging.Logger instance.
    """
    logger = logging.getLogger("captured_prints")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()  # Avoid duplicate handlers on re-runs (e.g., notebooks)

    # mode="w" => overwrite/clean old log each run
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)
    return logger


def coef_to_gibbs(coef: np.ndarray, list_pauli_matrix: torch.Tensor, beta: float) -> torch.Tensor:
    """
    Converts Hamiltonian coefficients to a Gibbs state density matrix.

    Args:
        coef: A numpy array of coefficients.
        list_pauli_matrix: A tensor stack of Pauli matrices corresponding to the coefficients.
        beta: Inverse temperature parameter.

    Returns:
        The normalized Gibbs state density matrix.
    """
    # Reshape coefficients for broadcasting
    coef_tensor = torch.from_numpy(coef).unsqueeze(-1).unsqueeze(-1)
    # Sum(coef * Pauli_Matrix)
    matrix = torch.sum(coef_tensor * list_pauli_matrix, dim=-3)
    # Calculate Matrix Exponential: exp(-beta * H)
    rho = torch.matrix_exp(-beta * matrix)
    # Normalize by trace
    return rho / trace(rho).view([-1, 1, 1])


def coef_weight(filter_coef: torch.Tensor, barrier: float, xi: float) -> torch.Tensor:
    """
    Calculates the weight of the coefficients based on a sigmoid filter.

    Args:
        filter_coef: The coefficient tensor to evaluate (usually the last column).
        barrier: The barrier threshold parameter.
        xi: The normalization factor (sum of absolute coefficients).

    Returns:
        A tensor of weights.
    """
    w_min = 0.05
    # Sigmoid function centered around barrier * xi
    w = torch.sigmoid((torch.abs(filter_coef) - barrier * xi) / (0.25 * barrier * xi))
    return w_min + (1.0 - w_min) * w


class ThermalStateSampler:
    r"""Thermal state sampler.

    - `get_test_sample`: uniform sampling in the outer box excluding the inner box
    - `get_train_sample`: buffered samples produced by `algorithm1` and consumed without replacement;
      automatically refills when depleted.
    """

    def __init__(
        self,
        *,
        n: int,
        beta: float,
        hamiltonian,
        error,
        num_step: int,
        h: torch.Tensor,
        xi: float,
        barrier: float,
        prefetch_size: int = 10_000,
        max_chunk: int = 1_00_000,
        w_min: float = 0.05,
    ):
        self.n = n
        self.beta = beta
        self.hamiltonian = hamiltonian
        self.error = error
        self.num_step = num_step

        self.h = h
        self.xi = xi
        self.barrier = barrier

        self.prefetch_size = int(prefetch_size)
        self.max_chunk = int(max_chunk)
        
        logger = make_logger("run.log")
        out = StreamToLogger(logger, logging.INFO)
        err = StreamToLogger(logger, logging.ERROR)
        self.out = out
        self.err = err

        self.w_min = float(w_min)

        self._train_dm_buf: Optional[torch.Tensor] = None
        self._train_label_buf: Optional[torch.Tensor] = None
        self._train_weight_buf: Optional[torch.Tensor] = None
        self._cursor = 0

        if self.prefetch_size > 0:
            self._refill_train_buffer(self.prefetch_size)

    def _coef_weight(self, filter_coef: torch.Tensor) -> torch.Tensor:
        w = torch.sigmoid(
            (torch.abs(filter_coef) - self.barrier * self.xi)
            / (0.25 * self.barrier * self.xi)
        )
        return self.w_min + (1.0 - self.w_min) * w

    def _train_remaining(self) -> int:
        if self._train_label_buf is None:
            return 0
        return int(self._train_label_buf.numel() - self._cursor)

    def _maybe_compact(self) -> None:
        """Drop the consumed prefix to limit long-lived memory usage."""
        if self._train_label_buf is None:
            return
        total = int(self._train_label_buf.numel())
        if total == 0:
            self._train_dm_buf = None
            self._train_label_buf = None
            self._train_weight_buf = None
            self._cursor = 0
            return

        if self._cursor >= max(1024, total // 2):
            self._train_dm_buf = self._train_dm_buf[self._cursor :]
            self._train_label_buf = self._train_label_buf[self._cursor :]
            self._train_weight_buf = self._train_weight_buf[self._cursor :]
            self._cursor = 0

    def _append_train(self, dm: torch.Tensor, label: torch.Tensor, weight: torch.Tensor) -> None:
        if self._train_dm_buf is None:
            self._train_dm_buf = dm
            self._train_label_buf = label
            self._train_weight_buf = weight
        else:
            self._train_dm_buf = torch.cat([self._train_dm_buf, dm], dim=0)
            self._train_label_buf = torch.cat([self._train_label_buf, label], dim=0)
            self._train_weight_buf = torch.cat([self._train_weight_buf, weight], dim=0)

    def _refill_train_buffer(self, target_size: int) -> None:
        """Generate and append at least `target_size` accepted training samples via `algorithm1`."""
        target_size = int(target_size)
        if target_size <= 0:
            return

        sample_ratio = 2
        accepted = 0

        def _loop():
            nonlocal sample_ratio, accepted
            with torch.no_grad():
                while accepted < target_size:
                    remain = target_size - accepted
                    remain_num = min(self.max_chunk, max(10, remain * sample_ratio))
                    print(f"Sampling {remain_num} states...")

                    list_state, sample_coef, _, _ = algorithm1(
                        n=self.n,
                        beta=self.beta,
                        hamiltonian=self.hamiltonian,
                        error=self.error,
                        num_sample=remain_num,
                        num_step=self.num_step,
                        save_state=False,
                        save_compressed=False,
                    )

                    sample_coef = torch.from_numpy(sample_coef)
                    satisfy = torch.abs(sample_coef)[:, -1] >= self.barrier * self.xi

                    if torch.any(satisfy):
                        dm = list_state[satisfy].density_matrix
                        if not isinstance(dm, torch.Tensor):
                            dm = torch.from_numpy(dm)

                        label = torch.sign(sample_coef[satisfy, -1])
                        weight = self._coef_weight(sample_coef[satisfy, -1])
                        self._append_train(dm, label, weight)

                        accepted += int(torch.sum(satisfy).item())

                    sample_ratio *= 2

        with redirect_stdout(self.out), redirect_stderr(self.err):
            _loop()

    def get_train_sample(self, size: int) -> Tuple[qkit.State, torch.Tensor, torch.Tensor]:
        """Return `size` training samples (consumed without replacement)."""
        size = int(size)
        if size <= 0:
            raise ValueError(f"size must be positive, got {size}")

        self._maybe_compact()

        with torch.no_grad():
            while self._train_remaining() < size:
                need = size - self._train_remaining()
                self._refill_train_buffer(max(self.prefetch_size, need))

            s = self._cursor
            e = s + size
            dm = self._train_dm_buf[s:e]
            label = self._train_label_buf[s:e]
            weight = self._train_weight_buf[s:e]
            self._cursor = e

            # When depleted, prefetch the next batch for future calls.
            if self._train_remaining() == 0 and self.prefetch_size > 0:
                self._train_dm_buf = None
                self._train_label_buf = None
                self._train_weight_buf = None
                self._cursor = 0
                self._refill_train_buffer(self.prefetch_size)

        return to_state(dm), label, weight

    def get_test_sample(self, size: int) -> Tuple[qkit.State, torch.Tensor, torch.Tensor]:
        """Uniform sampling for test set (same logic as before)."""
        sample_size = int(size)
        if sample_size <= 0:
            raise ValueError(f"size must be positive, got {sample_size}")

        a_row = self.h.view(1, len(self.h))

        # acceptance rate = 1 - y^L (use it to pick a reasonable batch size)
        acc = 1.0 - (self.barrier ** len(self.h))
        batch = max(1024, int((sample_size / max(acc, 1e-6)) * 1.2))

        list_coef = []
        i = 0
        with torch.no_grad():
            while i < sample_size:
                u = torch.rand((batch, len(self.h)))
                x = (u * 2 - 1) * a_row  # uniform in outer box
                keep = (x.abs() >= (a_row * self.barrier)).any(dim=1)  # outside inner box
                x = x[keep]
                list_coef.append(x)
                i += x.size(0)
                batch = max(1024, sample_size - i)  # adjust a bit as we get close

        list_pauli_str = self.hamiltonian.pauli_words
        list_pauli_matrix = torch.stack([pauli_str_to_matrix(p_str, self.n) for p_str in list_pauli_str])
        
        sample_coef = torch.cat(list_coef, dim=0)[:sample_size]
        sampled_state = to_state(coef_to_gibbs(sample_coef.numpy(), list_pauli_matrix, self.beta))
        sampled_label = torch.sign(sample_coef[:, -1])
        sampled_weight = self._coef_weight(sample_coef[:, -1])
        return sampled_state, sampled_label, sampled_weight
