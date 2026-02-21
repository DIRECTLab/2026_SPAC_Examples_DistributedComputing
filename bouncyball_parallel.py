import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
import os


# -----------------------------
# Helpers: log odds not needed here
# -----------------------------

# -----------------------------
# 1) Pure branching process (Poisson offspring)
# -----------------------------
def simulate_branching_poisson_once(args: Tuple[float, int, int]) -> int:
    """
    Worker-safe: pure function. Args = (R, max_events, seed)
    Returns cascade size capped by max_events.
    """
    R, max_events, seed = args
    np.random.seed(seed)

    total = 0
    q = deque([1])

    while q and total < max_events:
        q.popleft()
        total += 1
        k = np.random.poisson(R)
        # append k children
        # (cap growth so we do not exceed max_events too badly)
        remaining = max_events - total - len(q)
        if remaining <= 0:
            break
        if k > remaining:
            k = remaining
        for _ in range(int(k)):
            q.append(1)

    return total


def estimate_extinction_and_sizes_parallel(
    R_values: Iterable[float],
    trials: int = 2000,
    max_events: int = 20000,
    extinct_cutoff: int = 500,
    base_seed: int = 12345,
    workers: int = None,
    chunks_per_R: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parallel estimate for each R:
      - extinction probability: P(size < extinct_cutoff)
      - mean cascade size
    """
    R_values = list(R_values)
    if workers is None:
        workers = max(1, os.cpu_count() or 1)

    p_ext = np.zeros(len(R_values), dtype=float)
    mean_size = np.zeros(len(R_values), dtype=float)

    # We submit all tasks for each R then reduce results.
    # chunks_per_R can be >1 if you want to reduce overhead by batching trials inside worker,
    # but simplest is 1 (one trial per task).
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, R in enumerate(R_values):
            futures = []
            # deterministic unique seeds per trial
            for t in range(trials):
                seed = base_seed + (i + 1) * 1_000_000 + t
                futures.append(ex.submit(simulate_branching_poisson_once, (float(R), int(max_events), int(seed))))

            sizes = []
            for f in as_completed(futures):
                sizes.append(f.result())

            sizes = np.array(sizes, dtype=int)
            p_ext[i] = np.mean(sizes < extinct_cutoff)
            mean_size[i] = np.mean(sizes)

    return p_ext, mean_size


# -----------------------------
# 2) Geometric "large balls emit small balls"
# -----------------------------
def random_unit_vector_2d(rng: random.Random) -> Tuple[float, float]:
    ang = rng.random() * 2.0 * math.pi
    return math.cos(ang), math.sin(ang)


def simulate_geometric_cascade_once(args: Tuple[int, float, float, float, int, int, int]) -> int:
    """
    Worker-safe: Args =
      (N_targets, domain_size, target_radius, step_length, emitted_per_active, max_events, seed)
    Returns cascade size.
    """
    N_targets, domain_size, target_radius, step_length, emitted_per_active, max_events, seed = args

    rng = random.Random(seed)
    np.random.seed(seed)

    centers = np.random.rand(int(N_targets), 2) * float(domain_size)

    # Grid accel
    cell_size = max(float(target_radius) * 2.5, 1e-6)
    grid_w = int(math.ceil(float(domain_size) / cell_size))
    grid = {}
    for idx, (x, y) in enumerate(centers):
        cx = int(x / cell_size)
        cy = int(y / cell_size)
        grid.setdefault((cx, cy), []).append(idx)

    def nearby_candidates(px: float, py: float):
        cx = int(px / cell_size)
        cy = int(py / cell_size)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                yield from grid.get((cx + dx, cy + dy), [])

    activated = np.zeros(int(N_targets), dtype=bool)
    start = np.random.randint(0, int(N_targets))
    activated[start] = True
    q = deque([start])
    total = 1

    r2 = float(target_radius) * float(target_radius)

    while q and total < min(int(max_events), int(N_targets)):
        idx = q.popleft()
        x0, y0 = centers[idx]

        for _ in range(int(emitted_per_active)):
            ux, uy = random_unit_vector_2d(rng)
            px = x0 + float(step_length) * ux
            py = y0 + float(step_length) * uy

            if not (0.0 <= px <= float(domain_size) and 0.0 <= py <= float(domain_size)):
                continue

            for j in nearby_candidates(px, py):
                if activated[j]:
                    continue
                dx = centers[j, 0] - px
                dy = centers[j, 1] - py
                if dx * dx + dy * dy <= r2:
                    activated[j] = True
                    q.append(j)
                    total += 1
                    break

        if total >= int(N_targets):
            break

    return total


def scan_density_threshold_parallel(
    N_targets_values: Iterable[int],
    trials: int = 250,
    domain_size: float = 1.0,
    target_radius: float = 0.020,
    step_length: float = 0.070,
    emitted_per_active: int = 6,
    max_events: int = 20000,
    extinct_cutoff: int = 50,
    base_seed: int = 54321,
    workers: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parallel sweep over density proxy N_targets.
    """
    N_targets_values = list(map(int, N_targets_values))
    if workers is None:
        workers = max(1, os.cpu_count() or 1)

    p_ext = np.zeros(len(N_targets_values), dtype=float)
    mean_size = np.zeros(len(N_targets_values), dtype=float)

    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, n in enumerate(N_targets_values):
            futures = []
            for t in range(trials):
                seed = base_seed + (i + 1) * 1_000_000 + t
                futures.append(
                    ex.submit(
                        simulate_geometric_cascade_once,
                        (int(n), float(domain_size), float(target_radius), float(step_length),
                         int(emitted_per_active), int(max_events), int(seed))
                    )
                )

            sizes = []
            for f in as_completed(futures):
                sizes.append(f.result())

            sizes = np.array(sizes, dtype=int)
            p_ext[i] = np.mean(sizes < extinct_cutoff)
            mean_size[i] = np.mean(sizes)

    return p_ext, mean_size


# -----------------------------
# Plot helper
# -----------------------------
def plot_threshold_curve(x, p_ext, mean_size, x_label, title_prefix, vline_at=None):
    plt.figure(figsize=(9, 4.5))
    plt.plot(x, p_ext, marker="o")
    if vline_at is not None:
        plt.axvline(vline_at, linestyle="--")
    plt.ylim(-0.02, 1.02)
    plt.xlabel(x_label)
    plt.ylabel("Estimated extinction probability")
    plt.title(f"{title_prefix}: Extinction probability vs {x_label}")
    plt.grid(True, alpha=0.3)

    plt.figure(figsize=(9, 4.5))
    plt.plot(x, mean_size, marker="o")
    if vline_at is not None:
        plt.axvline(vline_at, linestyle="--")
    plt.xlabel(x_label)
    plt.ylabel("Mean cascade size (capped)")
    plt.title(f"{title_prefix}: Mean cascade size vs {x_label}")
    plt.grid(True, alpha=0.3)


# -----------------------------
# Main
# -----------------------------
def main():
    workers = max(1, (os.cpu_count() or 1) - 1)

    # ---- Demo A: Pure branching threshold ----
    R_values = np.linspace(0.2, 1.8, 17)
    p_ext, mean_size = estimate_extinction_and_sizes_parallel(
        R_values,
        trials=2000,
        max_events=20000,
        extinct_cutoff=500,
        base_seed=12345,
        workers=workers
    )
    plot_threshold_curve(
        R_values, p_ext, mean_size,
        x_label="R (mean offspring)",
        title_prefix="Pure branching (Poisson, parallel)",
        vline_at=1.0
    )

    # ---- Demo B: Geometric cascade threshold by density ----
    N_targets_values = np.linspace(150, 1400, 14).astype(int)
    p_ext_g, mean_size_g = scan_density_threshold_parallel(
        N_targets_values,
        trials=250,
        domain_size=1.0,
        target_radius=0.020,
        step_length=0.070,
        emitted_per_active=6,
        max_events=20000,
        extinct_cutoff=50,
        base_seed=54321,
        workers=workers
    )
    plot_threshold_curve(
        N_targets_values, p_ext_g, mean_size_g,
        x_label="N_targets (density proxy)",
        title_prefix="Geometric cascade (parallel)",
        vline_at=None
    )

    plt.show()


if __name__ == "__main__":
    main()