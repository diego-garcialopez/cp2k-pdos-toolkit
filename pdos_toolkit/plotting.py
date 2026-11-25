"""Matplotlib figures: PDOS, band structure, and the two side by side.

Design rules applied throughout: one fixed colour per group assigned in
order (so a group keeps its colour across figures), orbitals of a group
share its colour and differ by dash pattern, the total DOS is neutral grey,
text is always in ink colours, and grid/axes stay recessive.

matplotlib is imported lazily so that the rest of the toolkit works without
it.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .bands import BandStructure
from .constants import l_of
from .dos import TOTAL, DOSCurves

__all__ = [
    "PALETTE",
    "plot_dos",
    "plot_bands",
    "plot_bands_dos",
    "plot_comparison",
    "group_colours",
    "reference_label",
    "save_figure",
]

log = logging.getLogger("pdos_toolkit")

#: Categorical colours, used strictly in this order. The order keeps
#: neighbouring colours distinguishable under the common colour-vision
#: deficiencies.
PALETTE = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#6250d6",  # violet
    "#e34948",  # red
)
INK = "#1f1f1e"
INK_MUTED = "#6b6a63"
GRID = "#e4e3dd"
TOTAL_FILL = "#d9d8d2"
TOTAL_LINE = "#4a4a46"

_ORBITAL_DASHES = {
    "s": (0, (5, 2)),
    "p": (0, (1, 1.2)),
    "d": (0, (6, 2, 1, 2)),
    "f": (0, (3, 1.5, 1, 1.5, 1, 1.5)),
    "g": (0, (8, 3)),
}
_EXTRA_DASHES = [(0, (4, 1)), (0, (2, 2)), (0, (7, 2, 2, 2)), (0, (1, 3))]


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "plotting needs matplotlib: pip install matplotlib"
        ) from exc
    return plt


def group_colours(
    groups: Sequence[str], colours: Optional[Mapping[str, str]] = None
) -> Dict[str, str]:
    """Assign palette colours to groups in order; ``colours`` overrides."""
    out: Dict[str, str] = {}
    names = [g for g in groups if g != TOTAL]
    if len(names) > len(PALETTE):
        log.warning(
            "%d groups but only %d distinct colours: colours repeat. Merge kinds "
            "(group by element, or define groups) for a readable figure.",
            len(names), len(PALETTE),
        )
    for i, g in enumerate(names):
        out[g] = PALETTE[i % len(PALETTE)]
    out[TOTAL] = TOTAL_LINE
    if colours:
        out.update(colours)
    return out


def reference_label(reference: str) -> str:
    """Axis label for an energy axis shifted to ``reference``."""
    return {
        "fermi": r"$E - E_\mathrm{F}$ (eV)",
        "vbm": r"$E - E_\mathrm{VBM}$ (eV)",
        "cbm": r"$E - E_\mathrm{CBM}$ (eV)",
        "midgap": r"$E - E_\mathrm{mid\,gap}$ (eV)",
        "absolute": "Energy (eV)",
    }.get(reference, f"Energy (eV), zero at {reference}")


def _style_axes(ax) -> None:
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_MUTED, labelcolor=INK, width=0.8)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)


def _parse_spec(spec: str) -> Tuple[str, Optional[str], Optional[str]]:
    """``"Ni:d"`` -> (Ni, d, None); ``"O"`` -> (O, None, None);
    ``"Ni:d:alpha"`` -> (Ni, d, alpha)."""
    parts = spec.split(":")
    group = parts[0]
    orbital = parts[1] if len(parts) > 1 and parts[1] else None
    spin = parts[2] if len(parts) > 2 and parts[2] else None
    if spin not in (None, "alpha", "beta"):
        raise ValueError(f"bad spin {spin!r} in curve spec {spec!r}")
    return group, orbital, spin


def _visible_max(x: np.ndarray, ys: Sequence[np.ndarray], lim) -> float:
    mask = np.ones(x.shape, dtype=bool) if lim is None else (x >= lim[0]) & (x <= lim[1])
    vals = [float(np.max(np.abs(y[mask]))) for y in ys if mask.any()]
    return max(vals) if vals else 1.0


def plot_dos(
    curves: DOSCurves,
    ax=None,
    *,
    kind: str = "total",
    select: Optional[Sequence[str]] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    fill: bool = False,
    stack: bool = False,
    show_total: Optional[bool] = None,
    orientation: str = "horizontal",
    colours: Optional[Mapping[str, str]] = None,
    title: Optional[str] = None,
    legend: bool = True,
    mark_edges: bool = False,
    figsize: Tuple[float, float] = (8.0, 4.8),
):
    """Plot broadened curves.

    Args:
        curves: what to plot.
        ax: existing axes; a new figure is created otherwise.
        kind: ``"total"`` (one line per group), ``"orbital"`` (one line per
            group and orbital) or ``"both"``.
        select: explicit list of curves instead of ``kind``, e.g.
            ``["Ni:d", "O:p", "Total"]`` (``group[:orbital[:spin]]``).
        xlim, ylim: limits of the energy and DOS axes. The DOS axis is
            auto-scaled to what is visible inside the energy window.
        fill: shade the area under each group's curve.
        stack: stack the group curves on top of each other (composition of
            the total DOS); implies ``kind="total"``.
        show_total: draw the total DOS as a grey backdrop (default: if the
            curves contain one).
        orientation: ``"horizontal"`` (energy on x) or ``"vertical"``
            (energy on y, for placing next to a band structure).
        colours: ``{group: colour}`` overrides.
        mark_edges: draw faint lines at the VBM and CBM.

    Spin-polarised data is drawn mirrored: alpha up, beta down.

    Returns:
        ``(figure, axes)``.
    """
    plt = _plt()
    if kind not in ("total", "orbital", "both"):
        raise ValueError("kind must be 'total', 'orbital' or 'both'")
    if orientation not in ("horizontal", "vertical"):
        raise ValueError("orientation must be 'horizontal' or 'vertical'")
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    vertical = orientation == "vertical"
    energy = curves.energy
    spins: List[Optional[str]] = curves.spins if curves.is_spin_polarised else [None]
    groups = [g for g in curves.groups if g != TOTAL]
    has_total = TOTAL in curves.groups
    if select is not None and TOTAL in [_parse_spec(x)[0] for x in select]:
        # "Total" in an explicit selection is drawn as the grey backdrop.
        select = [x for x in select if _parse_spec(x)[0] != TOTAL]
        if show_total is None:
            show_total = True
        if not has_total:
            raise KeyError("no Total curve: build the curves with total=True (--total-dos)")
    if show_total is None:
        show_total = has_total and select is None
    cmap = group_colours(curves.groups, colours)
    drawn: List[np.ndarray] = []

    def sign(spin: Optional[str]) -> float:
        return -1.0 if spin == "beta" else 1.0

    def line(y, **kw):
        drawn.append(y)
        return ax.plot(y, energy, **kw) if vertical else ax.plot(energy, y, **kw)

    def area(y0, y1, **kw):
        drawn.append(y1)
        if vertical:
            return ax.fill_betweenx(energy, y0, y1, **kw)
        return ax.fill_between(energy, y0, y1, **kw)

    def spin_of(key_spin):
        return "sum" if not curves.is_spin_polarised else key_spin

    if show_total and has_total:
        for i, spin in enumerate(spins):
            y = sign(spin) * curves.get(TOTAL, None, spin_of(spin))
            area(0.0, y, color=TOTAL_FILL, linewidth=0, zorder=1,
                 label="Total" if i == 0 else None)
            line(y, color=TOTAL_LINE, linewidth=0.8, zorder=1.5)

    if select is not None:
        parsed = [_parse_spec(spec) for spec in select]
        for group, _, _ in parsed:
            if group not in curves.groups:
                raise KeyError(f"no group {group!r}; available: {curves.groups}")
        # If one group contributes several curves (Ni d and Ni p, or px/py/pz)
        # the curves themselves are what must be told apart: colour per
        # curve. Otherwise keep each group's own colour.
        names = [g for g, _, _ in parsed]
        per_curve = len(set(names)) < len(names)
        if per_curve and len(parsed) > len(PALETTE):
            raise ValueError(f"at most {len(PALETTE)} curves of one group can be told apart")
        for n, (group, orbital, spin) in enumerate(parsed):
            if per_curve:
                col, ls = PALETTE[n], "-"
            else:
                col = cmap.get(group, INK)
                ls = "-" if orbital is None else _ORBITAL_DASHES.get(l_of(orbital), "-")
            label = group if orbital is None else f"{group} {orbital}"
            for i, s in enumerate([spin] if spin else spins):
                y = sign(s) * curves.get(group, orbital, spin_of(s))
                line(y, color=col, linestyle=ls, linewidth=1.5, zorder=3,
                     label=label if i == 0 else None)
    elif stack:
        for spin in spins:
            base = np.zeros_like(energy)
            for g in groups:
                y = curves.get(g, None, spin_of(spin))
                top = base + y
                # A thin white edge separates neighbouring bands.
                area(sign(spin) * base, sign(spin) * top, color=cmap[g], alpha=0.9,
                     linewidth=0.6, edgecolor="white", zorder=2,
                     label=g if spin in (None, "alpha") else None)
                base = top
    else:
        for g in groups:
            col = cmap[g]
            if kind in ("total", "both"):
                for i, spin in enumerate(spins):
                    y = sign(spin) * curves.get(g, None, spin_of(spin))
                    if fill:
                        area(0.0, y, color=col, alpha=0.15, linewidth=0, zorder=2)
                    line(y, color=col, linewidth=1.6, zorder=3, label=g if i == 0 else None)
            if kind in ("orbital", "both"):
                extra = 0
                for orb in curves.orbitals(g):
                    ls = _ORBITAL_DASHES.get(orb)
                    if ls is None:
                        ls = _EXTRA_DASHES[extra % len(_EXTRA_DASHES)]
                        extra += 1
                    for i, spin in enumerate(spins):
                        y = sign(spin) * curves.get(g, orb, spin_of(spin))
                        line(y, color=col, linestyle=ls, linewidth=1.1, zorder=2.5,
                             label=f"{g} {orb}" if i == 0 else None)

    # Reference line, band edges, zero line for mirrored spin plots.
    ref_line = ax.axhline if vertical else ax.axvline
    zero_line = ax.axvline if vertical else ax.axhline
    ref_line(0.0, color=INK_MUTED, linestyle=(0, (2, 2)), linewidth=0.9, zorder=4)
    if mark_edges:
        for key in ("vbm", "cbm"):
            val = curves.meta.get(key)
            if val is not None and abs(float(val)) > 1e-6:
                ref_line(float(val), color=INK_MUTED, linestyle=(0, (1, 3)),
                         linewidth=0.8, zorder=4)
    if curves.is_spin_polarised:
        zero_line(0.0, color=INK_MUTED, linewidth=0.8, zorder=4)

    elabel = reference_label(curves.reference)
    dlabel = f"DOS ({curves.unit})"
    if curves.is_spin_polarised:
        dlabel += "   [↑ alpha / ↓ beta]"
    top = _visible_max(energy, drawn, xlim) * 1.06 if drawn else 1.0
    auto = (-top, top) if curves.is_spin_polarised else (0.0, top)
    if vertical:
        ax.set_ylabel(elabel)
        ax.set_xlabel(dlabel)
        if xlim is not None:
            ax.set_ylim(xlim)
        ax.set_xlim(ylim if ylim is not None else auto)
    else:
        ax.set_xlabel(elabel)
        ax.set_ylabel(dlabel)
        if xlim is not None:
            ax.set_xlim(xlim)
        else:
            ax.set_xlim(energy[0], energy[-1])
        ax.set_ylim(ylim if ylim is not None else auto)
    if title:
        ax.set_title(title, loc="left", fontsize=11)
    _style_axes(ax)
    if legend:
        handles, labels = ax.get_legend_handles_labels()
        if len(handles) > 1 or (len(handles) == 1 and select is not None):
            ax.legend(
                handles, labels, frameon=False, fontsize=9, labelcolor=INK,
                loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0,
            )
    return fig, ax


def plot_comparison(
    datasets: Mapping[str, DOSCurves],
    spec: str = TOTAL,
    ax=None,
    *,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (8.0, 4.8),
):
    """Overlay the same curve from several calculations.

    Typical uses: a framework with and without adsorbate, a series of
    dopants, or different functionals. Each dataset must already be on the
    energy reference you want to compare on -- remember that aligning two
    *different* calculations at the Fermi level or VBM is a choice, not a
    physical alignment (see ``docs/theory.md``).

    Args:
        datasets: ``{label: curves}``, drawn in this order.
        spec: which curve, ``group[:orbital[:spin]]``.
    """
    plt = _plt()
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    group, orbital, spin = _parse_spec(spec)
    if len(datasets) > len(PALETTE):
        raise ValueError(f"at most {len(PALETTE)} datasets can be told apart in one plot")
    drawn = []
    xs = []
    reference = None
    unit = None
    for i, (label, cur) in enumerate(datasets.items()):
        reference = reference or cur.reference
        unit = unit or cur.unit
        spins: List[Optional[str]] = [spin] if spin else (
            cur.spins if cur.is_spin_polarised else [None]
        )
        for j, s in enumerate(spins):
            y = cur.get(group, orbital, s if cur.is_spin_polarised else "sum")
            y = -y if s == "beta" else y
            ax.plot(cur.energy, y, color=PALETTE[i], linewidth=1.6,
                    label=label if j == 0 else None, zorder=3)
            mask = (
                np.ones(cur.energy.shape, bool) if xlim is None
                else (cur.energy >= xlim[0]) & (cur.energy <= xlim[1])
            )
            if mask.any():
                drawn.append(float(np.abs(y[mask]).max()))
            xs.append(cur.energy)
    mirrored = any(c.is_spin_polarised for c in datasets.values()) and spin is None
    ax.axvline(0.0, color=INK_MUTED, linestyle=(0, (2, 2)), linewidth=0.9, zorder=4)
    if mirrored:
        ax.axhline(0.0, color=INK_MUTED, linewidth=0.8, zorder=4)
    top = (max(drawn) if drawn else 1.0) * 1.06
    ax.set_ylim(ylim if ylim is not None else ((-top, top) if mirrored else (0.0, top)))
    ax.set_xlim(xlim if xlim is not None else (min(x[0] for x in xs), max(x[-1] for x in xs)))
    ax.set_xlabel(reference_label(reference or "absolute"))
    name = group if orbital is None else f"{group} {orbital}"
    ax.set_ylabel(f"{name} DOS ({unit})")
    if title:
        ax.set_title(title, loc="left", fontsize=11)
    _style_axes(ax)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left",
              bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    return fig, ax


def plot_bands(
    bs: BandStructure,
    ax=None,
    *,
    shift: float = 0.0,
    cell: Optional[np.ndarray] = None,
    ylim: Optional[Tuple[float, float]] = None,
    ylabel: Optional[str] = None,
    mark_gap: bool = False,
    efermi: Optional[float] = None,
    n_occupied: Optional[int] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (6.4, 4.8),
):
    """Plot a band structure along its k-path.

    Args:
        shift: energy subtracted from all bands (e.g. the VBM or E_F).
        cell: lattice vectors (3x3, Angstrom) for true segment lengths.
        mark_gap: mark the VBM and CBM with dots.
        efermi, n_occupied: passed to :meth:`BandStructure.band_gap` when
            ``mark_gap`` is set and the file has no occupations.

    Spin-polarised bands are drawn in two colours (alpha solid, beta
    dashed).

    Returns:
        ``(figure, axes)``.
    """
    plt = _plt()
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    dist = bs.distances(cell)
    energies = bs.energies - shift
    # Break the lines where one k-point set ends and the next begins, so
    # that no spurious segment is drawn across a path discontinuity.
    bounds = np.cumsum([0] + [s.nkpoints for s in bs.sets])
    for ispin in range(bs.nspin):
        colour = PALETTE[0] if ispin == 0 else PALETTE[1]
        style = "-" if ispin == 0 else (0, (4, 1.5))
        for a, b in zip(bounds[:-1], bounds[1:]):
            lines = ax.plot(dist[a:b], energies[ispin, a:b, :], color=colour,
                            linestyle=style, linewidth=1.3, zorder=3)
            if a == 0 and bs.nspin == 2:
                lines[0].set_label("alpha" if ispin == 0 else "beta")
    pos, labels = bs.ticks(cell)
    for x in pos:
        ax.axvline(x, color=GRID, linewidth=0.9, zorder=1)
    ax.axhline(0.0, color=INK_MUTED, linestyle=(0, (2, 2)), linewidth=0.9, zorder=2)
    if mark_gap:
        gap = bs.band_gap(efermi=efermi, n_occupied=n_occupied)
        k = bs.kpoints
        for level, kpt, colour in ((gap.vbm, gap.vbm_kpoint, INK), (gap.cbm, gap.cbm_kpoint, INK)):
            idx = int(np.where(np.all(np.isclose(k, kpt, atol=1e-8), axis=1))[0][0])
            ax.plot([dist[idx]], [level - shift], marker="o", markersize=6, color=colour,
                    markeredgecolor="white", markeredgewidth=1.2, zorder=5, linestyle="none",
                    clip_on=False)
    ax.set_xticks(pos)
    ax.set_xticklabels(labels)
    ax.set_xlim(dist[0], dist[-1])
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_ylabel(ylabel or ("Energy (eV)" if shift == 0.0 else "Energy (eV, shifted)"))
    if title:
        ax.set_title(title, loc="left", fontsize=11)
    _style_axes(ax)
    ax.grid(False)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6)
    ax.tick_params(axis="x", length=0)
    if bs.nspin == 2:
        ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper right")
    return fig, ax


def plot_bands_dos(
    bs: BandStructure,
    curves: DOSCurves,
    *,
    shift: float = 0.0,
    cell: Optional[np.ndarray] = None,
    ylim: Optional[Tuple[float, float]] = None,
    dos_kind: str = "total",
    dos_select: Optional[Sequence[str]] = None,
    dos_max: Optional[float] = None,
    mark_gap: bool = False,
    efermi: Optional[float] = None,
    n_occupied: Optional[int] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (9.0, 4.8),
):
    """Band structure and PDOS side by side on a shared energy axis.

    ``shift`` is applied to the bands; the curves must already be on the
    same zero (e.g. both relative to the VBM). The two panels show
    different quantities against the *same* energy axis, which is why they
    are separate panels rather than one plot with two scales.

    Returns:
        ``(figure, (band_axes, dos_axes))``.
    """
    plt = _plt()
    fig, (ax_b, ax_d) = plt.subplots(
        1, 2, figsize=figsize, sharey=True, gridspec_kw={"width_ratios": [2.4, 1.0], "wspace": 0.06}
    )
    plot_bands(bs, ax_b, shift=shift, cell=cell, ylim=ylim, mark_gap=mark_gap,
               efermi=efermi, n_occupied=n_occupied,
               ylabel=reference_label(curves.reference))
    plot_dos(curves, ax_d, kind=dos_kind, select=dos_select, orientation="vertical",
             xlim=ylim, ylim=None if dos_max is None else (0.0, dos_max))
    ax_d.set_ylabel("")
    ax_d.tick_params(axis="y", length=0)
    if title:
        ax_b.set_title(title, loc="left", fontsize=11)
    return fig, (ax_b, ax_d)


def save_figure(fig, path: str, dpi: int = 300) -> str:
    """Save with a tight bounding box (so an outside legend is not cut)."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    return path
