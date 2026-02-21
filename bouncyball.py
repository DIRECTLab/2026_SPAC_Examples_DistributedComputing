import math
import random
from collections import deque, Counter

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1) Pure branching process
# -----------------------------
def simulate_branching_poisson(R: float, max_events: int = 20000) -> int:
    """
    Galton–Watson branching where each event spawns K ~ Poisson(R) children.
    Returns total number of events (cascade size), capped by max_events.
    """
    total = 0
    q = deque([1])  # start with 1 initial event

    while q and total < max_events:
        q.popleft()
        total += 1
        k = np.random.poisson(R)
        for _ in range(int(k)):
            if total + len(q) >= max_events:
                break
            q.append(1)

    return total

def estimate_extinction_and_sizes(R_values, trials=2000, max_events=20000, extinct_cutoff=500):
    """
    For each R, estimate:
      - extinction probability (cascade size < extinct_cutoff)
      - mean cascade size (capped)
    """
    p_ext = []
    mean_size = []
    for R in R_values:
        sizes = [simulate_branching_poisson(R, max_events=max_events) for _ in range(trials)]
        p_ext.append(np.mean([s < extinct_cutoff for s in sizes]))
        mean_size.append(np.mean(sizes))
    return np.array(p_ext), np.array(mean_size)


# -----------------------------
# 2) Geometric "large balls emit small balls"
# -----------------------------
def random_unit_vector_2d():
    ang = random.random() * 2.0 * math.pi
    return math.cos(ang), math.sin(ang)

def simulate_geometric_cascade(
    N_targets: int = 600,
    domain_size: float = 1.0,
    target_radius: float = 0.025,
    step_length: float = 0.08,
    emitted_per_active: int = 6,
    max_events: int = 20000,
    seed: int = 0
) -> int:
    """
    Place N_targets discs ("large balls") uniformly in a 2D square domain.
    Start with one activated target. Each active target emits M small balls.
    Each small ball travels a fixed step_length in a random direction from the center.
    If it lands within target_radius of a *not-yet-activated* target center, it activates it.

    Returns cascade size (number of activated targets), capped by max_events.

    This is a geometric way to generate an emergent branching process.
    """
    random.seed(seed)
    np.random.seed(seed)

    # Target centers
    centers = np.random.rand(N_targets, 2) * domain_size

    # Simple grid acceleration (optional, but keeps it fast)
    cell_size = max(target_radius * 2.5, 1e-6)
    grid_w = int(math.ceil(domain_size / cell_size))
    grid = {}
    for i, (x, y) in enumerate(centers):
        cx = int(x / cell_size)
        cy = int(y / cell_size)
        grid.setdefault((cx, cy), []).append(i)

    def nearby_candidates(px, py):
        cx = int(px / cell_size)
        cy = int(py / cell_size)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                yield from grid.get((cx + dx, cy + dy), [])

    activated = np.zeros(N_targets, dtype=bool)
    # Start from a random seed target
    start = np.random.randint(0, N_targets)
    activated[start] = True

    q = deque([start])
    total = 1

    r2 = target_radius * target_radius

    while q and total < min(max_events, N_targets):
        idx = q.popleft()
        x0, y0 = centers[idx]

        for _ in range(emitted_per_active):
            ux, uy = random_unit_vector_2d()
            px = x0 + step_length * ux
            py = y0 + step_length * uy

            # If particle goes out of bounds, ignore (or you could wrap; ignore is simpler)
            if not (0.0 <= px <= domain_size and 0.0 <= py <= domain_size):
                continue

            # Check for hits: does particle land within target_radius of a target center?
            for j in nearby_candidates(px, py):
                if activated[j]:
                    continue
                dx = centers[j, 0] - px
                dy = centers[j, 1] - py
                if dx*dx + dy*dy <= r2:
                    activated[j] = True
                    q.append(j)
                    total += 1
                    break  # one particle activates at most one new target

        # Early stop if fully activated
        if total >= N_targets:
            break

    return total


def scan_density_threshold(
    N_targets_values,
    trials=250,
    domain_size=1.0,
    target_radius=0.025,
    step_length=0.08,
    emitted_per_active=6,
    max_events=20000,
    extinct_cutoff=50
):
    """
    Sweep over N_targets (controls density). For each, run trials and estimate:
      - extinction probability
      - mean cascade size
    """
    p_ext = []
    mean_size = []
    for n in N_targets_values:
        sizes = []
        for t in range(trials):
            sizes.append(simulate_geometric_cascade(
                N_targets=int(n),
                domain_size=domain_size,
                target_radius=target_radius,
                step_length=step_length,
                emitted_per_active=emitted_per_active,
                max_events=max_events,
                seed=1000 + 17*t + int(n)
            ))
        p_ext.append(np.mean([s < extinct_cutoff for s in sizes]))
        mean_size.append(np.mean(sizes))
    return np.array(p_ext), np.array(mean_size)


# -----------------------------
# Plot helpers
# -----------------------------
def plot_threshold_curve(x, p_ext, mean_size, x_label, title_prefix):
    fig1 = plt.figure(figsize=(9, 4.5))
    plt.plot(x, p_ext, marker="o")
    plt.axvline(1.0, linestyle="--") if x_label == "R (mean offspring)" else None
    plt.ylim(-0.02, 1.02)
    plt.xlabel(x_label)
    plt.ylabel("Estimated extinction probability")
    plt.title(f"{title_prefix}: Extinction probability vs {x_label}")
    plt.grid(True, alpha=0.3)

    fig2 = plt.figure(figsize=(9, 4.5))
    plt.plot(x, mean_size, marker="o")
    plt.axvline(1.0, linestyle="--") if x_label == "R (mean offspring)" else None
    plt.xlabel(x_label)
    plt.ylabel("Mean cascade size (capped)")
    plt.title(f"{title_prefix}: Mean cascade size vs {x_label}")
    plt.grid(True, alpha=0.3)

    return fig1, fig2


# -----------------------------
# Main: run both demos
# -----------------------------
def main():
    # ---- Demo A: Pure branching threshold at R ~ 1 ----
    R_values = np.linspace(0.2, 1.8, 17)
    p_ext, mean_size = estimate_extinction_and_sizes(
        R_values, trials=2000, max_events=20000, extinct_cutoff=500
    )
    plot_threshold_curve(R_values, p_ext, mean_size, "R (mean offspring)", "Pure branching (Poisson)")

    # ---- Demo B: Geometry-based threshold by changing density (N_targets) ----
    # Fix geometry parameters; sweep density via number of targets
    N_targets_values = np.linspace(150, 1400, 14).astype(int)

    # Parameters you can mention in the workshop:
    # - target_radius: how "big" the large balls are (cross-section)
    # - step_length: how far the emitted small balls travel (spacing)
    # - emitted_per_active: how many small balls each active ball emits
    # Density is controlled by N_targets / domain_size^2
    p_ext_g, mean_size_g = scan_density_threshold(
        N_targets_values,
        trials=250,
        domain_size=1.0,
        target_radius=0.020,
        step_length=0.070,
        emitted_per_active=6,
        max_events=20000,
        extinct_cutoff=50
    )

    plot_threshold_curve(N_targets_values, p_ext_g, mean_size_g, "N_targets (density proxy)", "Geometric cascade")

    plt.show()


if __name__ == "__main__":
    main()