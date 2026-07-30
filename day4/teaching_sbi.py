"""Small, transparent helpers for the Day 5 SBI teaching notebooks.

The functions here deliberately contain no neural-network machinery. Students
implement the ABC and NPE steps in the notebook, while these helpers provide a
single trusted toy simulator and independent reference calculations.
"""

from __future__ import annotations

import numpy as np


TOY_PRIOR_LOW = -3.0
TOY_PRIOR_HIGH = 3.0
TOY_NOISE_SIGMA = 0.25


def sample_toy_prior(
    n: int,
    rng: np.random.Generator,
    *,
    low: float = TOY_PRIOR_LOW,
    high: float = TOY_PRIOR_HIGH,
) -> np.ndarray:
    """Draw ``n`` scalar parameters and return shape ``(n, 1)``."""
    if n <= 0:
        raise ValueError("n must be positive")
    return rng.uniform(low, high, size=(n, 1))


def simulate_toy_reference(
    theta: np.ndarray,
    rng: np.random.Generator,
    *,
    noise_sigma: float = TOY_NOISE_SIGMA,
) -> np.ndarray:
    """Simulate ``x = theta**2 + Normal(0, noise_sigma)``.

    Parameters and returned summaries both use shape ``(n, 1)``. The
    many-to-one square is what makes the posterior bimodal for positive
    observations; SBI does not manufacture the two modes.
    """
    value = np.asarray(theta, dtype=float)
    if value.ndim == 1:
        value = value[:, None]
    if value.ndim != 2 or value.shape[1] != 1:
        raise ValueError("theta must have shape (n, 1)")
    if noise_sigma <= 0:
        raise ValueError("noise_sigma must be positive")
    return value**2 + rng.normal(0.0, noise_sigma, size=value.shape)


def exact_toy_posterior(
    theta_grid: np.ndarray,
    x_observed: float,
    *,
    noise_sigma: float = TOY_NOISE_SIGMA,
    low: float = TOY_PRIOR_LOW,
    high: float = TOY_PRIOR_HIGH,
) -> np.ndarray:
    """Evaluate and normalize the analytic toy posterior on a 1D grid."""
    grid = np.asarray(theta_grid, dtype=float)
    if grid.ndim != 1 or grid.size < 2:
        raise ValueError("theta_grid must be a one-dimensional grid")
    inside = (grid >= low) & (grid <= high)
    log_likelihood = -0.5 * ((float(x_observed) - grid**2) / noise_sigma) ** 2
    log_likelihood -= np.max(log_likelihood[inside])
    density = np.where(inside, np.exp(log_likelihood), 0.0)
    try:
        normalization = np.trapezoid(density, grid)
    except:
        normalization = np.trapz(density, grid)
    if not np.isfinite(normalization) or normalization <= 0:
        raise ValueError("posterior normalization failed")
    return density / normalization


def central_interval(samples: np.ndarray, probability: float) -> tuple[np.ndarray, np.ndarray]:
    """Return an equal-tailed central interval along the sample axis."""
    if not 0.0 < probability < 1.0:
        raise ValueError("probability must lie between zero and one")
    values = np.asarray(samples, dtype=float)
    alpha = 0.5 * (1.0 - probability)
    return (
        np.quantile(values, alpha, axis=0),
        np.quantile(values, 1.0 - alpha, axis=0),
    )


def empirical_marginal_coverage(
    posterior_samples: np.ndarray,
    theta_true: np.ndarray,
    probability: float,
) -> np.ndarray:
    """Marginal equal-tailed coverage for batched posterior samples.

    ``posterior_samples`` has shape ``(n_cases, n_draws, n_parameters)`` and
    ``theta_true`` has shape ``(n_cases, n_parameters)``.
    """
    samples = np.asarray(posterior_samples, dtype=float)
    truth = np.asarray(theta_true, dtype=float)
    if samples.ndim != 3:
        raise ValueError("posterior_samples must have shape (cases, draws, parameters)")
    if truth.shape != (samples.shape[0], samples.shape[2]):
        raise ValueError("theta_true has incompatible shape")
    low, high = central_interval(np.swapaxes(samples, 0, 1), probability)
    return np.mean((truth >= low) & (truth <= high), axis=0)
