import numpy as np
import matplotlib.pyplot as plt

def _try_import_cupy():
    try:
        import cupy as cp
        return cp
    except Exception as e:
        raise RuntimeError(
            "CuPy is not available. Install a matching wheel, for example:\n"
            "  pip install cupy-cuda12x\n"
            "Then rerun."
        ) from e


def branching_poisson_gpu(
    R: float,
    trials: int = 200_000,
    max_total: int = 20_000,
    extinct_cutoff_total: int = 500,
    max_generations: int = 2_000,
    seed: int = 12345,
):
    """
    GPU-accelerated Galton–Watson branching:
      X_{g+1} = sum_{i=1..X_g} Poisson(R)
    Uses the property: sum of independent Poisson(R) is Poisson(R * X_g).

    We simulate many trials at once:
      active[g, trial] = number of individuals/events in generation g.

    Stopping conditions per trial:
      - extinct when X_g == 0
      - capped when cumulative total >= max_total

    Returns:
      p_ext: extinction probability (total < extinct_cutoff_total)
      mean_total: mean total cascade size (capped at max_total)
    """
    cp = _try_import_cupy()
    rs = cp.random.RandomState(seed)

    # State vectors for all trials
    x = cp.ones(trials, dtype=cp.int32)          # current generation population
    total = cp.ones(trials, dtype=cp.int32)      # cumulative size so far
    alive = cp.ones(trials, dtype=cp.bool_)      # still running

    # Iterate generations
    for _ in range(max_generations):
        if not bool(alive.any()):
            break

        # For each trial: next generation is Poisson(R * x)
        lam = R * x.astype(cp.float32)

        # Only update trials still alive
        nxt = rs.poisson(lam)
        nxt = nxt.astype(cp.int32)

        # Update totals only for alive trials
        nxt = cp.where(alive, nxt, cp.int32(0))

        total = total + nxt
        # Cap the total
        total = cp.minimum(total, cp.int32(max_total))

        # Next generation
        x = nxt

        # Update alive mask: extinct or capped trials stop
        alive = alive & (x > 0) & (total < max_total)

    # Bring back to CPU for statistics
    total_cpu = cp.asnumpy(total)

    p_ext = float(np.mean(total_cpu < extinct_cutoff_total))
    mean_total = float(np.mean(total_cpu))
    return p_ext, mean_total


def sweep_R_gpu(
    R_values,
    trials: int = 200_000,
    max_total: int = 20_000,
    extinct_cutoff_total: int = 500,
    seed: int = 12345,
):
    p_ext = []
    mean_size = []
    for i, R in enumerate(R_values):
        pe, ms = branching_poisson_gpu(
            float(R),
            trials=trials,
            max_total=max_total,
            extinct_cutoff_total=extinct_cutoff_total,
            seed=seed + 1000 * i,
        )
        p_ext.append(pe)
        mean_size.append(ms)
    return np.array(p_ext), np.array(mean_size)


def main():
    # R sweep around the threshold
    R_values = np.linspace(0.2, 1.8, 17)

    # For a live demo, start smaller then scale up:
    # trials=50_000 is already smooth, 200_000 is very smooth
    p_ext, mean_size = sweep_R_gpu(
        R_values,
        trials=200_000,
        max_total=20_000,
        extinct_cutoff_total=500,
        seed=12345,
    )

    plt.figure(figsize=(9, 4.5))
    plt.plot(R_values, p_ext, marker="o")
    plt.axvline(1.0, linestyle="--")
    plt.ylim(-0.02, 1.02)
    plt.xlabel("R (mean offspring)")
    plt.ylabel("Estimated extinction probability")
    plt.title("GPU branching (Poisson): extinction probability vs R")
    plt.grid(True, alpha=0.3)

    plt.figure(figsize=(9, 4.5))
    plt.plot(R_values, mean_size, marker="o")
    plt.axvline(1.0, linestyle="--")
    plt.xlabel("R (mean offspring)")
    plt.ylabel("Mean cascade size (capped)")
    plt.title("GPU branching (Poisson): mean cascade size vs R")
    plt.grid(True, alpha=0.3)

    plt.show()


if __name__ == "__main__":
    main()