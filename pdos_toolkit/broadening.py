"""Line shapes and the broadening kernel.

Every line shape here is normalised to unit area, so a broadened curve is a
true density: its integral over energy equals the summed weight of the
states that went in, whatever the output grid spacing.

Width conventions
-----------------
Two conventions are in common use and are easy to mix up:

* ``sigma`` -- the Gaussian standard deviation (what this toolkit's ``-s``
  option, and ``cp2k-output-tools``, have always used);
* ``FWHM`` -- the full width at half maximum (what CP2K >= 2026.2 uses for
  its own broadened curves, ``&BROADEN / WIDTH``).

They are related by ``FWHM = 2 sqrt(2 ln 2) sigma ~= 2.3548 sigma``.
:class:`LineShape` accepts either and converts explicitly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = [
    "LineShape",
    "fwhm_to_sigma",
    "sigma_to_fwhm",
    "broaden",
    "make_grid",
    "LINESHAPES",
]

_FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))

LINESHAPES = ("gaussian", "lorentzian", "pseudo-voigt", "voigt")
_ALIASES = {
    "gauss": "gaussian",
    "g": "gaussian",
    "lorentz": "lorentzian",
    "l": "lorentzian",
    "pseudo_voigt": "pseudo-voigt",
    "pseudovoigt": "pseudo-voigt",
    "pv": "pseudo-voigt",
    "v": "voigt",
}


def fwhm_to_sigma(fwhm: float) -> float:
    """Gaussian standard deviation for a given full width at half maximum."""
    return fwhm / _FWHM_PER_SIGMA


def sigma_to_fwhm(sigma: float) -> float:
    """Full width at half maximum of a Gaussian of standard deviation sigma."""
    return sigma * _FWHM_PER_SIGMA


@dataclass(frozen=True)
class LineShape:
    """A normalised broadening function.

    Args:
        kind: ``"gaussian"``, ``"lorentzian"``, ``"pseudo-voigt"`` or
            ``"voigt"``.
        sigma: Gaussian standard deviation, eV. Give this *or* ``fwhm``.
        fwhm: full width at half maximum, eV. For the Gaussian, Lorentzian
            and pseudo-Voigt shapes this is the width of the whole peak (the
            pseudo-Voigt mixes a Gaussian and a Lorentzian of equal FWHM,
            exactly as CP2K does).
        eta: Lorentzian fraction of the pseudo-Voigt, 0..1.
        gamma: Lorentzian half width at half maximum of the true Voigt
            profile, eV (its Gaussian part is set by ``sigma`` / ``fwhm``).
    """

    kind: str = "gaussian"
    sigma: Optional[float] = None
    fwhm: Optional[float] = None
    eta: float = 0.5
    gamma: Optional[float] = None

    def __post_init__(self) -> None:
        kind = _ALIASES.get(self.kind.lower(), self.kind.lower())
        if kind not in LINESHAPES:
            raise ValueError(f"unknown line shape {self.kind!r}; choose from {LINESHAPES}")
        object.__setattr__(self, "kind", kind)
        sigma, fwhm = self.sigma, self.fwhm
        if sigma is None and fwhm is None:
            raise ValueError("give the broadening width as sigma or as fwhm")
        if sigma is not None and fwhm is not None and not math.isclose(
            sigma_to_fwhm(sigma), fwhm, rel_tol=1e-6
        ):
            raise ValueError("sigma and fwhm were both given and disagree")
        if sigma is None:
            sigma = fwhm_to_sigma(fwhm)
        fwhm = sigma_to_fwhm(sigma)
        if not (sigma > 0 and math.isfinite(sigma)):
            raise ValueError("the broadening width must be positive and finite")
        if not 0.0 <= self.eta <= 1.0:
            raise ValueError("eta (Lorentzian fraction) must lie in [0, 1]")
        if kind == "voigt":
            if self.gamma is None or self.gamma <= 0:
                raise ValueError("the Voigt profile needs gamma > 0 (Lorentzian HWHM, eV)")
        object.__setattr__(self, "sigma", float(sigma))
        object.__setattr__(self, "fwhm", float(fwhm))

    # ------------------------------------------------------------------ #
    @property
    def total_fwhm(self) -> float:
        """FWHM of the full profile (Olivero-Longbothum estimate for Voigt)."""
        if self.kind != "voigt":
            return self.fwhm
        fl = 2.0 * self.gamma
        return 0.5346 * fl + math.sqrt(0.2166 * fl * fl + self.fwhm**2)

    @property
    def cutoff(self) -> float:
        """Distance beyond which the tail is negligible for grid padding."""
        if self.kind == "gaussian":
            return 6.0 * self.sigma
        return 25.0 * self.total_fwhm

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the unit-area profile at energy offsets ``x`` (eV)."""
        x = np.asarray(x, dtype=float)
        if self.kind == "gaussian":
            return _gaussian(x, self.sigma)
        hwhm = 0.5 * self.fwhm
        if self.kind == "lorentzian":
            return _lorentzian(x, hwhm)
        if self.kind == "pseudo-voigt":
            return self.eta * _lorentzian(x, hwhm) + (1.0 - self.eta) * _gaussian(x, self.sigma)
        return _voigt(x, self.sigma, self.gamma)

    def describe(self) -> str:
        """One-line human-readable description, for file headers."""
        if self.kind == "gaussian":
            return f"Gaussian, sigma = {self.sigma:.6g} eV (FWHM = {self.fwhm:.6g} eV)"
        if self.kind == "lorentzian":
            return f"Lorentzian, FWHM = {self.fwhm:.6g} eV"
        if self.kind == "pseudo-voigt":
            return f"pseudo-Voigt, FWHM = {self.fwhm:.6g} eV, Lorentzian fraction = {self.eta:.4g}"
        return (
            f"Voigt, Gaussian sigma = {self.sigma:.6g} eV, Lorentzian HWHM = "
            f"{self.gamma:.6g} eV (FWHM ~ {self.total_fwhm:.6g} eV)"
        )


def _gaussian(x: np.ndarray, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * (x / sigma) ** 2) / (sigma * math.sqrt(2.0 * math.pi))


def _lorentzian(x: np.ndarray, hwhm: float) -> np.ndarray:
    return hwhm / (math.pi * (x * x + hwhm * hwhm))


def _voigt(x: np.ndarray, sigma: float, gamma: float) -> np.ndarray:
    """True Voigt profile: Re[w(z)] / (sigma sqrt(2 pi))."""
    z = (x + 1j * gamma) / (sigma * math.sqrt(2.0))
    try:
        from scipy.special import wofz  # type: ignore

        w = wofz(z)
    except ImportError:  # pragma: no cover - exercised only without SciPy
        w = _faddeeva_weideman(z)
    return np.real(w) / (sigma * math.sqrt(2.0 * math.pi))


def _faddeeva_weideman(z: np.ndarray, n: int = 32) -> np.ndarray:
    """Faddeeva function w(z) for Im z > 0 (Weideman 1994, SIAM J. Numer.
    Anal. 31, 1497). Used only when SciPy is not installed; accurate to
    ~1e-12 for the arguments that occur here."""
    m = 2 * n
    m2 = 2 * m
    k = np.arange(-m + 1, m)
    ll = math.sqrt(n / math.sqrt(2.0))
    theta = k * math.pi / m
    t = ll * np.tan(theta / 2.0)
    f = np.exp(-t * t) * (ll * ll + t * t)
    f = np.concatenate(([0.0], f))
    a = np.real(np.fft.fft(np.fft.fftshift(f))) / m2
    a = np.flipud(a[1 : n + 1])
    zz = (ll + 1j * z) / (ll - 1j * z)
    p = np.polyval(a, zz)
    return 2.0 * p / (ll - 1j * z) ** 2 + (1.0 / math.sqrt(math.pi)) / (ll - 1j * z)


# --------------------------------------------------------------------------- #
def make_grid(emin: float, emax: float, de: float) -> np.ndarray:
    """Uniform energy grid that contains both end points.

    The number of points is computed explicitly rather than with
    ``np.arange`` so that floating-point rounding cannot drop ``emax``.
    """
    if de <= 0:
        raise ValueError("the grid spacing de must be positive")
    if emax < emin:
        raise ValueError("emax must not be smaller than emin")
    n = int(math.ceil((emax - emin) / de - 1e-9)) + 1
    return emin + de * np.arange(n)


def broaden(
    energies: np.ndarray,
    weights: np.ndarray,
    grid: np.ndarray,
    shape: LineShape,
    method: str = "auto",
    max_elements: int = 4_000_000,
) -> np.ndarray:
    """Broaden a set of weighted delta peaks onto ``grid``.

    Computes ``D_j(E) = sum_i w_ij L(E - e_i)`` with a unit-area line shape
    ``L``; no factor of the grid spacing enters, so the result is a density
    (states per eV) that does not depend on how finely it is sampled.

    Args:
        energies: (n_states,) peak positions, eV.
        weights: (n_states,) or (n_states, n_columns) peak weights.
        grid: (n_grid,) energies at which to evaluate, eV.
        shape: the line shape.
        method: ``"exact"`` evaluates every peak on every grid point (in
            memory-bounded chunks). ``"fft"`` deposits the peaks on the grid
            by linear interpolation and convolves once -- much faster for
            many states (long MD averages), at the price of a small extra
            smearing of order ``de`` and needing a uniform grid.
            ``"auto"`` picks ``"fft"`` for large problems on a uniform grid
            that resolves the line shape, else ``"exact"``.
        max_elements: cap on the size of the temporary (grid x states)
            matrix of the exact method; controls peak memory.

    Returns:
        Array of shape (n_grid,) or (n_grid, n_columns).
    """
    energies = np.asarray(energies, dtype=float)
    weights = np.asarray(weights, dtype=float)
    grid = np.asarray(grid, dtype=float)
    squeeze = weights.ndim == 1
    if squeeze:
        weights = weights[:, None]
    if weights.shape[0] != energies.shape[0]:
        raise ValueError("energies and weights have different numbers of states")

    if method not in ("auto", "exact", "fft"):
        raise ValueError("method must be 'auto', 'exact' or 'fft'")
    if method == "auto":
        uniform = _is_uniform(grid)
        big = grid.size * energies.size > 5e7
        resolved = uniform and grid.size > 1 and (grid[1] - grid[0]) <= shape.total_fwhm / 8.0
        method = "fft" if (big and resolved) else "exact"

    if method == "fft":
        out = _broaden_fft(energies, weights, grid, shape)
    else:
        out = np.zeros((grid.size, weights.shape[1]))
        step = max(1, int(max_elements // max(grid.size, 1)))
        for start in range(0, energies.size, step):
            sl = slice(start, start + step)
            out += shape(grid[:, None] - energies[None, sl]) @ weights[sl]
    return out[:, 0] if squeeze else out


def _is_uniform(grid: np.ndarray) -> bool:
    if grid.size < 3:
        return True
    d = np.diff(grid)
    return bool(np.allclose(d, d[0], rtol=1e-6, atol=1e-12))


def _broaden_fft(
    energies: np.ndarray, weights: np.ndarray, grid: np.ndarray, shape: LineShape
) -> np.ndarray:
    if not _is_uniform(grid) or grid.size < 2:
        raise ValueError("the 'fft' method needs a uniform grid with at least two points")
    de = float(grid[1] - grid[0])
    n = grid.size
    # Pad so that peaks just outside the window still contribute their
    # tails, and so that the circular convolution cannot wrap around.
    pad = int(math.ceil(shape.cutoff / de)) + 2
    ntot = n + 2 * pad
    pos = (energies - grid[0]) / de + pad
    i0 = np.floor(pos).astype(np.int64)
    frac = pos - i0
    ok = (i0 >= 0) & (i0 < ntot - 1)
    hist = np.zeros((ntot, weights.shape[1]))
    np.add.at(hist, i0[ok], weights[ok] * (1.0 - frac[ok])[:, None])
    np.add.at(hist, i0[ok] + 1, weights[ok] * frac[ok][:, None])

    half = pad
    kernel = shape(de * np.arange(-half, half + 1))
    size = ntot + kernel.size - 1
    nfft = 1 << (size - 1).bit_length()
    fk = np.fft.rfft(kernel, nfft)
    out = np.empty((n, weights.shape[1]))
    for j in range(weights.shape[1]):
        conv = np.fft.irfft(np.fft.rfft(hist[:, j], nfft) * fk, nfft)[:size]
        out[:, j] = conv[half + pad : half + pad + n]
    return out
