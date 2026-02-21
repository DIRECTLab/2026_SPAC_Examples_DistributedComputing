import numpy as np
import matplotlib.pyplot as plt

def _try_import_cupy():
    try:
        import cupy as cp
        return cp
    except Exception as e:
        raise RuntimeError(
            "CuPy not found. Install a matching wheel, e.g.\n"
            "  pip install cupy-cuda12x\n"
            "and rerun."
        ) from e


def cascade_gpu_mixed_emitters(
    p_burst: float,
    lambda_burst: float,
    p_single_fail: float = 0.0,
    # Optional second burst mode (super-burst)
    p_superburst: float = 0.0,
    lambda_superburst: float = 25.0,
    trials: int = 200_000,
    max_total: int = 20_000,
    extinct_cutoff_total: int = 500,
    max_generations: int = 2000,
    seed: int = 12345,
):
    """
    GPU Galton–Watson style cascade with "balls that when hit release many balls or sometimes only one".

    Per active ball in a generation:
      - With probability p_burst -> burst emitter
          - With probability p_superburst -> Poisson(lambda_superburst)
            else -> Poisson(lambda_burst)
      - With probability (1 - p_burst) -> single emitter
          - With probability p_single_fail -> 0
            else -> 1

    We simulate MANY trials in parallel, generation-by-generation, using:
      nb ~ Binomial(x, p_burst)
      ns = x - nb
      offspring_from_burst ~ Poisson(lambda_burst * nb) [or split into super/regular]
      offspring_from_single = ns * Bernoulli(1 - p_single_fail)

    Returns:
      p_extinction, mean_total_size  (both on CPU floats)
    """
    cp = _try_import_cupy()
    rs = cp.random.RandomState(seed)

    # State per trial
    x = cp.ones(trials, dtype=cp.int32)      # population in current generation
    total = cp.ones(trials, dtype=cp.int32)  # cumulative cascade size
    alive = cp.ones(trials, dtype=cp.bool_)

    p_burst = float(p_burst)
    lambda_burst = float(lambda_burst)
    p_single_fail = float(p_single_fail)
    p_superburst = float(p_superburst)
    lambda_superburst = float(lambda_superburst)

    for _ in range(max_generations):
        if not bool(alive.any()):
            break

        # Only compute where alive; otherwise treat x as 0
        x_alive = cp.where(alive, x, cp.int32(0))

        # Split: burst vs single
        nb = rs.binomial(x_alive, p_burst).astype(cp.int32)
        ns = (x_alive - nb).astype(cp.int32)

        # Single emitters: each produces 1 with prob (1 - p_single_fail)
        if p_single_fail <= 0.0:
            single_offspring = ns
        elif p_single_fail >= 1.0:
            single_offspring = cp.zeros_like(ns)
        else:
            single_success = rs.binomial(ns, 1.0 - p_single_fail).astype(cp.int32)
            single_offspring = single_success

        # Burst emitters: optional split into superburst vs regular burst
        if p_superburst > 0.0:
            nsuper = rs.binomial(nb, p_superburst).astype(cp.int32)
            nreg = (nb - nsuper).astype(cp.int32)

            burst_reg = rs.poisson(lambda_burst * nreg.astype(cp.float32)).astype(cp.int32)
            burst_sup = rs.poisson(lambda_superburst * nsuper.astype(cp.float32)).astype(cp.int32)
            burst_offspring = burst_reg + burst_sup
        else:
            burst_offspring = rs.poisson(lambda_burst * nb.astype(cp.float32)).astype(cp.int32)

        nxt = (single_offspring + burst_offspring).astype(cp.int32)

        # Update totals (cap)
        total = total + nxt
        total = cp.minimum(total, cp.int32(max_total))

        # Advance generation
        x = nxt

        # Alive update: stop if extinct or capped
        alive = alive & (x > 0) & (total < max_total)

    total_cpu = cp.asnumpy(total)
    p_ext = float(np.mean(total_cpu < extinct_cutoff_total))
    mean_total = float(np.mean(total_cpu))
    return p_ext, mean_total


def sweep_parameter_gpu(
    param_values,
    mode: str = "p_burst",
    lambda_burst: float = 8.0,
    p_single_fail: float = 0.0,
    p_superburst: float = 0.0,
    lambda_superburst: float = 25.0,
    trials: int = 200_000,
    max_total: int = 20_000,
    extinct_cutoff_total: int = 500,
    seed: int = 12345,
):
    """
    Sweep either p_burst or lambda_burst and plot extinction / mean size.
    mode: "p_burst" or "lambda_burst"
    """
    p_ext = []
    mean_size = []

    for i, v in enumerate(param_values):
        if mode == "p_burst":
            pb = float(v)
            lb = float(lambda_burst)
        elif mode == "lambda_burst":
            pb = float(0.35)  # default if sweeping lambda
            lb = float(v)
        else:
            raise ValueError("mode must be 'p_burst' or 'lambda_burst'")

        pe, ms = cascade_gpu_mixed_emitters(
            p_burst=pb,
            lambda_burst=lb,
            p_single_fail=p_single_fail,
            p_superburst=p_superburst,
            lambda_superburst=lambda_superburst,
            trials=trials,
            max_total=max_total,
            extinct_cutoff_total=extinct_cutoff_total,
            seed=seed + 1000 * i,
        )
        p_ext.append(pe)
        mean_size.append(ms)

    return np.array(p_ext), np.array(mean_size)


def main():
    # --- Sweep p_burst: "how often does a hit cause a burst?" ---
    p_values = np.linspace(0.0, 0.9, 19)

    # Tune these for the workshop:
    # - lambda_burst is "how many small balls a burst releases on average"
    # - p_single_fail is "sometimes the single-emitter releases none"
    # - p_superburst adds occasional very large bursts
    p_ext, mean_size = sweep_parameter_gpu(
        p_values,
        mode="p_burst",
        lambda_burst=8.0,
        p_single_fail=0.05,
        p_superburst=0.05,
        lambda_superburst=30.0,
        trials=20_000_000,          # reduce to 50_000 if you want faster live
        max_total=20_000,
        extinct_cutoff_total=500,
        seed=12345
    )

    plt.figure(figsize=(9, 4.5))
    plt.plot(p_values, p_ext, marker="o")
    plt.ylim(-0.02, 1.02)
    plt.xlabel("p_burst (probability a hit triggers a burst)")
    plt.ylabel("Estimated extinction probability")
    plt.title("GPU cascade: extinction vs p_burst (sometimes 1, sometimes many)")
    plt.grid(True, alpha=0.3)

    plt.figure(figsize=(9, 4.5))
    plt.plot(p_values, mean_size, marker="o")
    plt.xlabel("p_burst (probability a hit triggers a burst)")
    plt.ylabel("Mean cascade size (capped)")
    plt.title("GPU cascade: mean size vs p_burst (sometimes 1, sometimes many)")
    plt.grid(True, alpha=0.3)

    plt.show()


if __name__ == "__main__":
    main()