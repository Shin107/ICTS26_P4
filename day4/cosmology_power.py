"""Small, explicit physics helpers for the Day 5 cosmology SBI tutorial.

The teaching experiment varies only

``theta = (Omega_m, sigma_8)``

and fixes ``Omega_b``, ``h``, ``n_s``, and redshift.  Linear matter power
spectra come from the wide-prior symbolic emulator of Bartlett & Pandey (2025),
via ``symbolic_pofk.wider_syren.linear.symbolic_pklin``.

Units used throughout
---------------------
* wavenumber: ``h / Mpc``
* power spectrum: ``(Mpc / h)^3``
* survey volume: ``(Mpc / h)^3``
* number density: ``(h / Mpc)^3``

The Gaussian covariance is deliberately evaluated once at a fiducial
cosmology and then held fixed.  This controlled model is useful for comparing
inference methods, but it is not a survey likelihood.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

import symbolic_pk_linear as wlin

DAY5_SEED = 2605
PARAMETER_NAMES = ("Omega_m", "sigma_8")

# Published calibration limits of the wide-prior symbolic model.
OMEGA_M_RANGE = (0.1, 0.5)
SIGMA_8_RANGE = (0.6, 1.0)
K_RANGE_H_MPC = (1.0e-4, 1.0e2)


@dataclass(frozen=True)
class FixedCosmology:
    """Cosmological parameters that are not inferred in the Day 5 exercise."""

    omega_b: float = 0.049
    h: float = 0.674
    n_s: float = 0.965
    redshift: float = 0.0

    def validate(self) -> None:
        """Check the published calibration range of the symbolic emulator."""

        _require_in_closed_interval("omega_b", self.omega_b, 0.03, 0.07)
        _require_in_closed_interval("h", self.h, 0.5, 0.9)
        _require_in_closed_interval("n_s", self.n_s, 0.8, 1.2)
        _require_in_closed_interval("redshift", self.redshift, 0.0, 3.0)


DEFAULT_FIXED_COSMOLOGY = FixedCosmology()


def _require_in_closed_interval(
    name: str,
    value: float,
    lower: float,
    upper: float,
) -> None:
    value = float(value)
    if not np.isfinite(value) or not lower <= value <= upper:
        raise ValueError(f"{name} must lie in [{lower}, {upper}], got {value}")


def _as_finite_vector(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validate_k(k_h_mpc: np.ndarray) -> np.ndarray:
    k = _as_finite_vector("k_h_mpc", k_h_mpc)
    if np.any(k <= 0):
        raise ValueError("all wavenumbers must be positive")
    if np.any(np.diff(k) <= 0):
        raise ValueError("wavenumbers must be strictly increasing")
    if k[0] < K_RANGE_H_MPC[0] or k[-1] > K_RANGE_H_MPC[1]:
        raise ValueError(
            "wavenumbers must remain inside the symbolic model calibration "
            f"range {K_RANGE_H_MPC} h/Mpc"
        )
    return k


def _validate_parameters(omega_m: float, sigma_8: float, fixed: FixedCosmology) -> None:
    fixed.validate()
    _require_in_closed_interval("omega_m", omega_m, *OMEGA_M_RANGE)
    _require_in_closed_interval("sigma_8", sigma_8, *SIGMA_8_RANGE)
    if float(omega_m) <= fixed.omega_b:
        raise ValueError("omega_m must exceed the fixed baryon fraction omega_b")


def symbolic_pofk_available() -> bool:
    """Return whether the pinned symbolic power-spectrum dependency is importable."""

    return _symbolic_pklin is not None


def _power_function() -> Callable[..., np.ndarray]:
    if _symbolic_pklin is None:
        raise ImportError(
            "The Day 5 power-spectrum model requires symbolic_pofk. "
            "Install day5/requirements-day5.txt before calling this function."
        ) from _SYMBOLIC_POFK_IMPORT_ERROR
    return _symbolic_pklin


def make_k_bins(
    k_min_h_mpc: float = 0.02,
    k_max_h_mpc: float = 0.25,
    n_bins: int = 20,
    *,
    spacing: str = "log",
) -> tuple[np.ndarray, np.ndarray]:
    """Return bin edges and representative wavenumbers.

    The representative wavenumber is the mean of ``k`` under the isotropic
    Fourier-shell measure ``k^2 dk``.  This keeps the center definition
    consistent for both linearly and logarithmically spaced edges.
    """

    if isinstance(n_bins, bool) or not isinstance(n_bins, (int, np.integer)):
        raise TypeError("n_bins must be an integer")
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    _require_in_closed_interval("k_min_h_mpc", k_min_h_mpc, *K_RANGE_H_MPC)
    _require_in_closed_interval("k_max_h_mpc", k_max_h_mpc, *K_RANGE_H_MPC)
    if k_max_h_mpc <= k_min_h_mpc:
        raise ValueError("k_max_h_mpc must exceed k_min_h_mpc")

    if spacing == "log":
        edges = np.geomspace(k_min_h_mpc, k_max_h_mpc, n_bins + 1)
    elif spacing == "linear":
        edges = np.linspace(k_min_h_mpc, k_max_h_mpc, n_bins + 1)
    else:
        raise ValueError("spacing must be 'log' or 'linear'")

    lower = edges[:-1]
    upper = edges[1:]
    centers = 0.75 * (upper**4 - lower**4) / (upper**3 - lower**3)
    return edges, centers


def mode_counts(k_edges_h_mpc: np.ndarray, volume_mpc_h3: float) -> np.ndarray:
    r"""Count the effective Fourier lattice modes in each spherical shell.

    The continuum count is

    ``N_i = V / (6 pi^2) * (k_high^3 - k_low^3)``.

    The conventional count includes the conjugate ``k`` and ``-k``
    wavevectors; :func:`fixed_gaussian_covariance` uses the matching
    ``2 P**2 / N`` convention. A floating-point effective count is returned
    because the formula is a continuum approximation rather than a lattice
    enumeration.
    """

    edges = _as_finite_vector("k_edges_h_mpc", k_edges_h_mpc)
    if edges.size < 2 or np.any(edges <= 0) or np.any(np.diff(edges) <= 0):
        raise ValueError("k_edges_h_mpc must contain increasing positive edges")
    volume_mpc_h3 = float(volume_mpc_h3)
    if not np.isfinite(volume_mpc_h3) or volume_mpc_h3 <= 0:
        raise ValueError("volume_mpc_h3 must be finite and positive")

    counts = (
        volume_mpc_h3
        * (edges[1:] ** 3 - edges[:-1] ** 3)
        / (6.0 * np.pi**2)
    )
    if not np.all(counts > 0):
        raise ValueError("every k bin must contain a positive effective mode count")
    return counts


def linear_power_spectrum(
    omega_m: float,
    sigma_8: float,
    k_h_mpc: np.ndarray,
    *,
    fixed: FixedCosmology = DEFAULT_FIXED_COSMOLOGY,
) -> np.ndarray:
    """Evaluate the linear matter power spectrum for one parameter point."""

    k = _validate_k(k_h_mpc)
    _validate_parameters(omega_m, sigma_8, fixed)

    power = _power_function()(
        float(omega_m),
        fixed.omega_b,
        fixed.h,
        fixed.n_s,
        float(sigma_8),
        fixed.redshift,
        k,
    )
    power = np.asarray(power, dtype=float)
    if power.shape != k.shape:
        raise ValueError(
            "symbolic_pofk returned an unexpected shape: "
            f"expected {k.shape}, got {power.shape}"
        )
    if not np.all(np.isfinite(power)) or not np.all(power > 0):
        raise ValueError("symbolic_pofk returned non-finite or non-positive power")
    return power


def linear_power_spectrum_batch(
    parameters: np.ndarray,
    k_h_mpc: np.ndarray,
    *,
    fixed: FixedCosmology = DEFAULT_FIXED_COSMOLOGY,
) -> np.ndarray:
    """Evaluate ``P(k)`` for a batch with columns ``(Omega_m, sigma_8)``."""

    theta = np.asarray(parameters, dtype=float)
    if theta.ndim != 2 or theta.shape[1] != 2 or theta.shape[0] == 0:
        raise ValueError("parameters must have shape (n_simulations, 2)")
    if not np.all(np.isfinite(theta)):
        raise ValueError("parameters must contain only finite values")

    return np.stack(
        [
            linear_power_spectrum(
                omega_m=row[0],
                sigma_8=row[1],
                k_h_mpc=k_h_mpc,
                fixed=fixed,
            )
            for row in theta
        ],
        axis=0,
    )


def fixed_gaussian_covariance(
    fiducial_power_mpc_h3: np.ndarray,
    counts: np.ndarray,
    *,
    number_density_h3_mpc3: float | None = None,
) -> np.ndarray:
    r"""Build the fixed diagonal Gaussian covariance used in the tutorial.

    For shell ``i``,

    ``Var[P_i] = 2 * (P_fid_i + 1 / nbar)^2 / N_i``.

    Set ``number_density_h3_mpc3=None`` for a matter field without shot noise.
    The input power must be evaluated once at the chosen fiducial cosmology;
    this function does not make the covariance parameter-dependent.
    """

    power = _as_finite_vector("fiducial_power_mpc_h3", fiducial_power_mpc_h3)
    counts = _as_finite_vector("counts", counts)
    if power.shape != counts.shape:
        raise ValueError("fiducial_power_mpc_h3 and counts must have the same shape")
    if np.any(power <= 0):
        raise ValueError("fiducial power must be positive")
    if np.any(counts <= 0):
        raise ValueError("mode counts must be positive")

    shot_noise = 0.0
    if number_density_h3_mpc3 is not None:
        number_density = float(number_density_h3_mpc3)
        if not np.isfinite(number_density) or number_density <= 0:
            raise ValueError("number_density_h3_mpc3 must be finite and positive")
        shot_noise = 1.0 / number_density

    variance = 2.0 * (power + shot_noise) ** 2 / counts
    covariance = np.diag(variance)
    if not np.all(np.isfinite(covariance)) or not np.all(variance > 0):
        raise ValueError("the Gaussian covariance is not finite and positive")
    return covariance


def covariance_cholesky(covariance: np.ndarray) -> np.ndarray:
    """Validate a covariance matrix and return its lower Cholesky factor."""

    matrix = np.asarray(covariance, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("covariance must be a non-empty square matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("covariance must contain only finite values")
    if not np.allclose(matrix, matrix.T, rtol=1.0e-12, atol=1.0e-14):
        raise ValueError("covariance must be symmetric")
    try:
        return np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError("covariance must be positive definite") from exc


def _validate_cholesky(cholesky: np.ndarray, n_features: int) -> np.ndarray:
    factor = np.asarray(cholesky, dtype=float)
    if factor.shape != (n_features, n_features):
        raise ValueError(
            f"cholesky must have shape {(n_features, n_features)}, got {factor.shape}"
        )
    if not np.all(np.isfinite(factor)):
        raise ValueError("cholesky must contain only finite values")
    if not np.allclose(factor, np.tril(factor), rtol=0.0, atol=1.0e-14):
        raise ValueError("cholesky must be lower triangular")
    if np.any(np.diag(factor) <= 0):
        raise ValueError("cholesky must have a positive diagonal")
    return factor


def _rng_from_seed(seed: int) -> np.random.Generator:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be a non-negative integer")
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    return np.random.default_rng(int(seed))


def simulate_power_spectra(
    parameters: np.ndarray,
    k_h_mpc: np.ndarray,
    cholesky: np.ndarray,
    *,
    seed: int = DAY5_SEED,
    fixed: FixedCosmology = DEFAULT_FIXED_COSMOLOGY,
) -> np.ndarray:
    """Simulate a batch from ``P(theta) + L epsilon`` with a fixed ``L``."""

    mean = linear_power_spectrum_batch(parameters, k_h_mpc, fixed=fixed)
    factor = _validate_cholesky(cholesky, mean.shape[1])
    standard_normal = _rng_from_seed(seed).standard_normal(mean.shape)
    return mean + standard_normal @ factor.T


def simulate_power_spectrum(
    parameters: np.ndarray,
    k_h_mpc: np.ndarray,
    cholesky: np.ndarray,
    *,
    seed: int = DAY5_SEED,
    fixed: FixedCosmology = DEFAULT_FIXED_COSMOLOGY,
) -> np.ndarray:
    """Simulate one spectrum for ``parameters = (Omega_m, sigma_8)``."""

    theta = _as_finite_vector("parameters", parameters)
    if theta.shape != (2,):
        raise ValueError("parameters must have shape (2,)")
    return simulate_power_spectra(
        theta[None, :],
        k_h_mpc,
        cholesky,
        seed=seed,
        fixed=fixed,
    )[0]


def whiten(
    values: np.ndarray,
    mean: np.ndarray,
    cholesky: np.ndarray,
) -> np.ndarray:
    """Return ``L^-1 (values - mean)`` for one vector or a batch.

    ``mean`` may be one shared one-dimensional vector or have the same shape
    as ``values``.  A triangular solve is used instead of forming an inverse.
    """

    data = np.asarray(values, dtype=float)
    reference = np.asarray(mean, dtype=float)
    if data.ndim not in (1, 2) or data.shape[-1] == 0:
        raise ValueError("values must have shape (n_features,) or (n_rows, n_features)")
    if reference.ndim not in (1, 2):
        raise ValueError("mean must be one- or two-dimensional")
    if not np.all(np.isfinite(data)) or not np.all(np.isfinite(reference)):
        raise ValueError("values and mean must contain only finite values")
    try:
        residual = data - reference
    except ValueError as exc:
        raise ValueError("mean is not broadcast-compatible with values") from exc
    if residual.shape != data.shape:
        raise ValueError("mean must not add an extra broadcast dimension")

    factor = _validate_cholesky(cholesky, data.shape[-1])
    if data.ndim == 1:
        return np.linalg.solve(factor, residual)
    return np.linalg.solve(factor, residual.T).T


def unwhiten(
    whitened_values: np.ndarray,
    mean: np.ndarray,
    cholesky: np.ndarray,
) -> np.ndarray:
    """Undo :func:`whiten` for one vector or a batch."""

    values = np.asarray(whitened_values, dtype=float)
    reference = np.asarray(mean, dtype=float)
    if values.ndim not in (1, 2) or values.shape[-1] == 0:
        raise ValueError(
            "whitened_values must have shape (n_features,) or "
            "(n_rows, n_features)"
        )
    if reference.ndim not in (1, 2):
        raise ValueError("mean must be one- or two-dimensional")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(reference)):
        raise ValueError("whitened_values and mean must contain only finite values")

    factor = _validate_cholesky(cholesky, values.shape[-1])
    transformed = factor @ values if values.ndim == 1 else values @ factor.T
    try:
        result = transformed + reference
    except ValueError as exc:
        raise ValueError("mean is not broadcast-compatible with values") from exc
    if result.shape != values.shape:
        raise ValueError("mean must not add an extra broadcast dimension")
    return result
