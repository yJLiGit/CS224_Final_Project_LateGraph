"""Small statistics helpers: bootstrap confidence intervals and seed aggregation."""
import numpy as np


def bootstrap_mean_ci(x, n_boot=2000, seed=0, alpha=0.05):
    """Mean of x and a percentile bootstrap CI (resampling the entries of x)."""
    x = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (n_boot, len(x)))
    m = x[idx].mean(1)
    return float(x.mean()), float(np.quantile(m, alpha / 2)), float(np.quantile(m, 1 - alpha / 2))


def paired_diff_ci(a, b, n_boot=2000, seed=0, alpha=0.05):
    """Mean of (a - b) over paired entries (e.g. per-query reciprocal ranks of two methods) with a bootstrap CI."""
    return bootstrap_mean_ci(np.asarray(a, np.float64) - np.asarray(b, np.float64), n_boot, seed, alpha)


def seed_summary(values):
    """mean, sample std and the number of seeds."""
    v = np.asarray(values, dtype=np.float64)
    return dict(mean=float(v.mean()), std=float(v.std(ddof=1)) if len(v) > 1 else 0.0, n=int(len(v)),
                values=[float(x) for x in v])


def t_interval(values, alpha=0.05):
    """mean +- t-based 95% interval over seeds (few seeds, so the t quantile matters)."""
    from scipy import stats
    v = np.asarray(values, dtype=np.float64)
    if len(v) < 2:
        return float(v.mean()), float("nan"), float("nan")
    h = stats.t.ppf(1 - alpha / 2, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return float(v.mean()), float(v.mean() - h), float(v.mean() + h)
