"""Core data model: a set of projections and the broadened curves made from it.

:class:`PDOSSet`
    All state-resolved projections of a calculation (kinds, atom lists, spin
    channels, MD steps). Knows how to pick subsets, find band edges and the
    energy reference, and broaden itself into curves.

:class:`DOSCurves`
    Broadened curves on a common energy grid, keyed by
    ``(group, orbital, spin)``. Can be integrated, saved and plotted.
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import (
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

from .broadening import LineShape, broaden, make_grid
from .constants import HARTREE_TO_EV, element_of, l_of, trapezoid
from .io import CurveFile, Projection, expand_paths, iter_pdos, read_curve, read_pdos_files

__all__ = [
    "PDOSSet",
    "DOSCurves",
    "BandEdges",
    "band_edges",
    "resolve_reference",
    "stream_broaden",
    "curves_from_cp2k",
    "REFERENCES",
    "TOTAL",
]

#: Group name used for the summed (total) DOS.
TOTAL = "Total"

REFERENCES = ("fermi", "vbm", "cbm", "midgap", "auto", "absolute")
_REF_ALIASES = {
    "homo": "vbm",
    "hoco": "vbm",
    "lumo": "cbm",
    "luco": "cbm",
    "none": "absolute",
    "abs": "absolute",
    "ef": "fermi",
    "efermi": "fermi",
}

Key = Tuple[str, str, Optional[str]]  # (group, orbital, spin)
Reference = Union[str, float]


# --------------------------------------------------------------------------- #
# Band edges and energy references
# --------------------------------------------------------------------------- #
@dataclass
class BandEdges:
    """Frontier levels of one spin channel (all energies in eV, absolute)."""

    vbm: Optional[float]  #: highest occupied state (HOMO / HOCO)
    cbm: Optional[float]  #: lowest unoccupied state (LUMO / LUCO)
    efermi: float
    fractional: bool  #: True if occupations are smeared
    spin: Optional[str] = None
    homo_index: Optional[int] = None  #: CP2K MO number of the VBM state

    @property
    def gap(self) -> Optional[float]:
        """CBM - VBM, or ``None`` if no empty state was printed."""
        if self.vbm is None or self.cbm is None:
            return None
        return self.cbm - self.vbm

    @property
    def midgap(self) -> Optional[float]:
        if self.vbm is None or self.cbm is None:
            return None
        return 0.5 * (self.vbm + self.cbm)


def band_edges(
    eigenvalues: np.ndarray,
    occupations: np.ndarray,
    efermi: float,
    spin: Optional[str] = None,
    mo_index: Optional[np.ndarray] = None,
) -> BandEdges:
    """Find VBM and CBM from eigenvalues and occupation numbers.

    A state counts as occupied when its occupation exceeds half the largest
    occupation in the list. For integer occupations this is simply
    "occupation > 0"; for smeared (fractional) occupations it puts the
    dividing line at the Fermi level instead of counting every thermally
    populated tail state as valence band.
    """
    eig = np.asarray(eigenvalues, dtype=float)
    occ = np.asarray(occupations, dtype=float)
    if eig.size == 0:
        return BandEdges(None, None, efermi, False, spin)
    full = float(occ.max())
    if full <= 1e-10:
        return BandEdges(None, float(eig.min()), efermi, False, spin)
    rel = occ / full
    fractional = bool(np.any((rel > 1e-4) & (rel < 1.0 - 1e-4)))
    occupied = rel > 0.5
    vbm = float(eig[occupied].max()) if occupied.any() else None
    cbm = float(eig[~occupied].min()) if (~occupied).any() else None
    homo = None
    if vbm is not None and mo_index is not None:
        homo = int(np.asarray(mo_index)[occupied][np.argmax(eig[occupied])])
    return BandEdges(vbm, cbm, efermi, fractional, spin, homo)


def _normalise_reference(reference: Reference) -> Reference:
    if isinstance(reference, (int, float)) and not isinstance(reference, bool):
        return float(reference)
    text = str(reference).strip().lower()
    text = _REF_ALIASES.get(text, text)
    if text in REFERENCES:
        return text
    try:
        return float(text)
    except ValueError:
        raise ValueError(
            f"unknown energy reference {reference!r}; use one of {REFERENCES} "
            "or a number (eV)"
        ) from None


def resolve_reference(edges: Sequence[BandEdges], reference: Reference) -> Tuple[float, str]:
    """Turn a reference keyword into an energy (eV) for one frame.

    With two spin channels a single, common zero is used for both, following
    CP2K: the higher Fermi level / the higher VBM / the lower CBM.

    Returns:
        ``(energy, label)`` where ``label`` names what was actually used
        (``"auto"`` is resolved to ``"fermi"`` or ``"vbm"``).
    """
    ref = _normalise_reference(reference)
    if isinstance(ref, float):
        return ref, f"{ref:g} eV"
    if ref == "absolute":
        return 0.0, "absolute"
    if ref == "auto":
        ref = "fermi" if any(e.fractional for e in edges) else "vbm"
    if ref == "fermi":
        return max(e.efermi for e in edges), "fermi"
    vbms = [e.vbm for e in edges if e.vbm is not None]
    cbms = [e.cbm for e in edges if e.cbm is not None]
    if ref == "vbm":
        if not vbms:
            raise ValueError("no occupied states found; cannot reference to the VBM")
        return max(vbms), "vbm"
    if not cbms:
        raise ValueError(
            "no unoccupied states in the PDOS files; cannot use the CBM / mid-gap "
            "reference (request empty states with NLUMO or ADDED_MOS in CP2K)"
        )
    if ref == "cbm":
        return min(cbms), "cbm"
    if not vbms:
        raise ValueError("no occupied states found; cannot reference to mid-gap")
    return 0.5 * (max(vbms) + min(cbms)), "midgap"


# --------------------------------------------------------------------------- #
# Curves
# --------------------------------------------------------------------------- #
@dataclass
class DOSCurves:
    """Broadened DOS curves on a common energy grid.

    ``curves`` maps ``(group, orbital, spin)`` to an array on ``energy``.
    ``orbital`` is ``"s"``, ``"p"``, ``"px"``, ... or ``""`` for a curve that
    is not orbital-resolved; ``spin`` is ``"alpha"``, ``"beta"`` or ``None``.
    """

    energy: np.ndarray
    curves: "OrderedDict[Key, np.ndarray]"
    reference: str = "absolute"
    reference_energy: float = 0.0  #: eV, absolute energy of the zero
    unit: str = "states/eV"
    meta: Dict[str, object] = field(default_factory=dict)

    # -- inspection ------------------------------------------------------- #
    @property
    def groups(self) -> List[str]:
        out: List[str] = []
        for g, _, _ in self.curves:
            if g not in out:
                out.append(g)
        return out

    @property
    def spins(self) -> List[Optional[str]]:
        out: List[Optional[str]] = []
        for _, _, s in self.curves:
            if s not in out:
                out.append(s)
        return out

    @property
    def is_spin_polarised(self) -> bool:
        return any(s is not None for s in self.spins)

    @property
    def de(self) -> float:
        return float(self.energy[1] - self.energy[0]) if self.energy.size > 1 else 0.0

    def orbitals(self, group: str) -> List[str]:
        out: List[str] = []
        for g, o, _ in self.curves:
            if g == group and o and o not in out:
                out.append(o)
        return out

    def get(
        self, group: str, orbital: Optional[str] = None, spin: Optional[str] = "sum"
    ) -> np.ndarray:
        """Return one curve, summing what is left unspecified.

        Args:
            group: group name (kind, element, custom group or ``"Total"``).
            orbital: orbital label; an l letter (``"p"``) sums its m
                components; ``None`` sums all orbitals of the group.
            spin: ``"alpha"``, ``"beta"``, or ``"sum"`` (default) for both.
        """
        total = np.zeros_like(self.energy)
        found = False
        for (g, o, s), y in self.curves.items():
            if g != group:
                continue
            if orbital is not None and o != orbital and l_of(o) != orbital:
                continue
            if spin != "sum" and s != spin:
                continue
            total = total + y
            found = True
        if not found:
            raise KeyError(f"no curve for group={group!r}, orbital={orbital!r}, spin={spin!r}")
        return total

    def total(self, spin: Optional[str] = "sum") -> np.ndarray:
        """The total DOS (requires curves built with ``total=True``)."""
        return self.get(TOTAL, None, spin)

    # -- numerics ---------------------------------------------------------- #
    def integrate(
        self, emin: Optional[float] = None, emax: Optional[float] = None
    ) -> "OrderedDict[Key, float]":
        """Trapezoidal integral of every curve over ``[emin, emax]``.

        Limits are on the (shifted) energy axis of the curves. Note that this
        integrates the *broadened* curves, so a sharp limit cuts through the
        tails of nearby peaks; for populations of the underlying states use
        :func:`pdos_toolkit.analysis.populations`, which is broadening-free.
        """
        lo = self.energy[0] if emin is None else emin
        hi = self.energy[-1] if emax is None else emax
        mask = (self.energy >= lo) & (self.energy <= hi)
        x = self.energy[mask]
        return OrderedDict(
            (k, float(trapezoid(y[mask], x)) if x.size > 1 else 0.0)
            for k, y in self.curves.items()
        )

    def value_at(self, energy: float = 0.0) -> "OrderedDict[Key, float]":
        """Linearly interpolated value of every curve at ``energy``."""
        return OrderedDict(
            (k, float(np.interp(energy, self.energy, y))) for k, y in self.curves.items()
        )

    def scaled(self, factor: float) -> "DOSCurves":
        """A copy with every curve multiplied by ``factor``."""
        return DOSCurves(
            self.energy.copy(),
            OrderedDict((k, y * factor) for k, y in self.curves.items()),
            self.reference,
            self.reference_energy,
            self.unit,
            dict(self.meta),
        )

    def window(self, emin: float, emax: float) -> "DOSCurves":
        """A copy restricted to ``emin <= E <= emax``."""
        mask = (self.energy >= emin) & (self.energy <= emax)
        return DOSCurves(
            self.energy[mask],
            OrderedDict((k, y[mask]) for k, y in self.curves.items()),
            self.reference,
            self.reference_energy,
            self.unit,
            dict(self.meta),
        )

    # -- output ------------------------------------------------------------ #
    @staticmethod
    def column_name(key: Key) -> str:
        group, orbital, spin = key
        name = group if not orbital else f"{group}_{orbital}"
        return name if spin is None else f"{name}_{spin}"

    def to_table(self, beta_negative: bool = False) -> Tuple[List[str], np.ndarray]:
        """``(column names, 2-D array)`` with the energy in the first column."""
        names = ["Energy_eV"]
        cols = [self.energy]
        for key, y in self.curves.items():
            names.append(self.column_name(key))
            cols.append(-y if (beta_negative and key[2] == "beta") else y)
        return names, np.column_stack(cols)

    def header_lines(self) -> List[str]:
        """Provenance lines written at the top of text output."""
        lines = [f"reference: {self.reference} (zero at {self.reference_energy:.6f} eV absolute)"]
        lines.append(f"unit: {self.unit}")
        for key in ("lineshape", "n_frames", "weighting", "normalisation", "files"):
            if key in self.meta:
                lines.append(f"{key}: {self.meta[key]}")
        return lines

    def save(self, path: str, fmt: Optional[str] = None, beta_negative: bool = False) -> str:
        """Write the curves to ``path``.

        The format follows the extension unless ``fmt`` is given:

        ``.dat`` / ``.txt``  aligned text with ``#`` comment header (loads
                             with ``numpy.loadtxt`` / gnuplot as is),
        ``.csv``             comma separated with a single header row,
        ``.npz``             compressed NumPy archive,
        ``.json``            JSON with metadata.
        """
        path = str(path)
        fmt = (fmt or os.path.splitext(path)[1].lstrip(".") or "dat").lower()
        names, table = self.to_table(beta_negative)
        if fmt in ("dat", "txt"):
            width = max(16, max(len(n) for n in names) + 1)
            head = [f"# {line}" for line in self.header_lines()]
            head.append("#" + "".join(f"{n:>{width}}" for n in names)[1:])
            with open(path, "w") as fh:
                fh.write("\n".join(head) + "\n")
                np.savetxt(fh, table, fmt=f"%{width}.8f", delimiter="")
        elif fmt == "csv":
            np.savetxt(path, table, delimiter=",", header=",".join(names), comments="", fmt="%.8f")
        elif fmt == "npz":
            np.savez_compressed(
                path,
                energy=self.energy,
                names=np.array(names[1:]),
                curves=table[:, 1:],
                reference=self.reference,
                reference_energy=self.reference_energy,
                unit=self.unit,
            )
        elif fmt == "json":
            payload = {
                "reference": self.reference,
                "reference_energy_eV": self.reference_energy,
                "unit": self.unit,
                "meta": _jsonable(self.meta),
                "energy_eV": self.energy.tolist(),
                "curves": {n: table[:, j + 1].tolist() for j, n in enumerate(names[1:])},
            }
            with open(path, "w") as fh:
                json.dump(payload, fh)
        else:
            raise ValueError(f"unknown output format {fmt!r} (use dat, csv, npz or json)")
        return path


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


# --------------------------------------------------------------------------- #
# Grouping
# --------------------------------------------------------------------------- #
GroupSpec = Union[str, Mapping[str, Sequence[str]], Callable[[Projection], Optional[str]]]


def make_grouper(
    group_by: GroupSpec = "kind", groups: Optional[Mapping[str, Sequence[str]]] = None
) -> Callable[[Projection], Optional[str]]:
    """Build the function that assigns each projection to a named group.

    Args:
        group_by: ``"kind"`` (one group per CP2K kind / list; the default),
            ``"element"`` (kinds of the same element merged, e.g. ``Ni1`` and
            ``Ni2``), ``"file"`` (never merge), or a callable.
        groups: explicit groups, ``{"framework": ["Al", "O", "C", "H"],
            "CO2": ["list1"]}``. Members may be kind names, list names or
            element symbols. Projections named here take the explicit group;
            everything else falls back to ``group_by``.
    """
    explicit: Dict[str, str] = {}
    for gname, members in (groups or {}).items():
        for member in members:
            if member in explicit and explicit[member] != gname:
                raise ValueError(f"{member!r} is assigned to two groups")
            explicit[member] = gname

    if callable(group_by):
        base = group_by
    elif group_by == "kind":
        base = lambda p: p.name  # noqa: E731
    elif group_by == "element":
        base = lambda p: p.element  # noqa: E731
    elif group_by == "file":
        base = lambda p: os.path.basename(p.path) or p.name  # noqa: E731
    else:
        raise ValueError("group_by must be 'kind', 'element', 'file' or a callable")

    def grouper(p: Projection) -> Optional[str]:
        if p.name in explicit:
            return explicit[p.name]
        if p.kind == "kind" and p.element in explicit:
            return explicit[p.element]
        return base(p)

    return grouper


# --------------------------------------------------------------------------- #
# Accumulation of frames onto a grid
# --------------------------------------------------------------------------- #
class _Accumulator:
    """Adds frames (all projections of one step) to curves on a fixed grid."""

    def __init__(
        self,
        grid: np.ndarray,
        shape: LineShape,
        grouper: Callable[[Projection], Optional[str]],
        orbitals: str,
        weighting: str,
        total: bool,
        natoms: Optional[Mapping[str, int]],
        per_atom: bool,
        method: str,
    ) -> None:
        if orbitals not in ("l", "m", "sum"):
            raise ValueError("orbitals must be 'l', 'm' or 'sum'")
        if weighting not in ("all", "occupied", "unoccupied"):
            raise ValueError("weighting must be 'all', 'occupied' or 'unoccupied'")
        self.grid = grid
        self.shape = shape
        self.grouper = grouper
        self.orbitals = orbitals
        self.weighting = weighting
        self.total = total
        self.natoms = dict(natoms or {})
        self.per_atom = per_atom
        self.method = method
        self.curves: "OrderedDict[Key, np.ndarray]" = OrderedDict()
        self.group_atoms: Dict[str, Dict[str, int]] = {}
        self.n_frames = 0

    def _columns(self, p: Projection) -> Tuple[List[str], np.ndarray]:
        if p.kind == "rspace":
            return [""], p.weights[:, :1]
        if self.orbitals == "sum":
            return [""], p.total[:, None]
        if self.orbitals == "l":
            return p.by_l()
        return list(p.labels), p.weights

    def _atoms_of(self, p: Projection) -> Optional[int]:
        for key in (p.name, p.element):
            if key in self.natoms:
                return int(self.natoms[key])
        return p.natoms

    def add(self, frame: Sequence[Projection], reference_energy: float) -> None:
        has_kinds = any(p.kind == "kind" for p in frame)
        for p in frame:
            group = self.grouper(p)
            labels, w = self._columns(p)
            if self.weighting != "all":
                full = float(p.occupations.max()) if p.n_states else 0.0
                f = p.occupations / full if full > 1e-10 else np.zeros_like(p.occupations)
                w = w * (f if self.weighting == "occupied" else 1.0 - f)[:, None]
            y = broaden(p.eigenvalues - reference_energy, w, self.grid, self.shape, self.method)
            if group is not None:
                for j, lab in enumerate(labels):
                    key = (group, lab, p.spin)
                    if key in self.curves:
                        self.curves[key] += y[:, j]
                    else:
                        self.curves[key] = y[:, j].copy()
                self.group_atoms.setdefault(group, {})[p.name] = self._atoms_of(p) or 0
            # The total is the sum over atomic kinds only: atom lists are
            # subsets of the kinds and would be counted twice.
            if self.total and (p.kind == "kind" or not has_kinds):
                key = (TOTAL, "", p.spin)
                tot = y.sum(axis=1)
                if key in self.curves:
                    self.curves[key] += tot
                else:
                    self.curves[key] = tot
        self.n_frames += 1

    def result(self, reference: str, reference_energy: float, meta: Dict[str, object]) -> DOSCurves:
        n = max(self.n_frames, 1)
        curves: "OrderedDict[Key, np.ndarray]" = OrderedDict()
        # Put the total last, like a legend reads best.
        keys = [k for k in self.curves if k[0] != TOTAL] + [k for k in self.curves if k[0] == TOTAL]
        unit = "states/eV"
        for key in keys:
            y = self.curves[key] / n
            if self.per_atom and key[0] != TOTAL:
                members = self.group_atoms.get(key[0], {})
                if not members or any(v <= 0 for v in members.values()):
                    missing = [m for m, v in members.items() if v <= 0]
                    raise ValueError(
                        "per-atom normalisation needs the number of atoms of "
                        f"{missing or key[0]}; pass it as natoms={{name: count}}"
                    )
                y = y / sum(members.values())
            curves[key] = y
        if self.per_atom:
            unit = "states/eV/atom"
        meta = dict(meta)
        meta["n_frames"] = self.n_frames
        meta["lineshape"] = self.shape.describe()
        meta["weighting"] = self.weighting
        if self.per_atom:
            meta["normalisation"] = "per atom (the Total curve is not normalised)"
        return DOSCurves(self.grid, curves, reference, reference_energy, unit, meta)


def _frame_edges(frame: Sequence[Projection]) -> List[BandEdges]:
    """Band edges per spin channel of one frame."""
    seen: Dict[Optional[str], BandEdges] = {}
    for p in frame:
        if p.spin in seen or p.kind == "rspace":
            continue
        seen[p.spin] = band_edges(p.eigenvalues, p.occupations, p.efermi, p.spin, p.mo_index)
    if not seen:
        raise ValueError("no state-resolved kind / list projection to take band edges from")
    return list(seen.values())


def _energy_window(
    frames: Iterable[Tuple[Sequence[Projection], float]], shape: LineShape
) -> Tuple[float, float]:
    lo, hi = np.inf, -np.inf
    for frame, ref in frames:
        for p in frame:
            if p.n_states:
                lo = min(lo, float(p.eigenvalues.min()) - ref)
                hi = max(hi, float(p.eigenvalues.max()) - ref)
    if not np.isfinite(lo):
        raise ValueError("the PDOS files contain no states")
    if shape.kind == "gaussian":
        pad = 10.0 * shape.sigma
    else:
        pad = min(shape.cutoff, 10.0 * shape.total_fwhm)
    return lo - pad, hi + pad


# --------------------------------------------------------------------------- #
# The set of projections
# --------------------------------------------------------------------------- #
class PDOSSet:
    """All projections of one calculation (or one trajectory)."""

    def __init__(self, projections: Iterable[Projection], check: bool = True):
        self.projections: List[Projection] = list(projections)
        if not self.projections:
            raise ValueError("no PDOS data given")
        if check:
            self._check()

    # -- construction ------------------------------------------------------ #
    @classmethod
    def from_files(
        cls,
        paths: Union[str, Iterable[str]],
        steps: Optional[slice] = None,
        last_only: bool = False,
    ) -> "PDOSSet":
        """Load raw ``.pdos`` files; glob patterns are expanded.

        Args:
            paths: one path / pattern or several.
            steps: slice over the steps stored in each file (``APPEND``-ed
                trajectories), e.g. ``slice(50, None, 5)``.
            last_only: keep only the last step of each file.
        """
        if isinstance(paths, (str, os.PathLike)):
            paths = [str(paths)]
        return cls(read_pdos_files(paths, steps=steps, last_only=last_only))

    def _check(self) -> None:
        """Validate that the files plausibly belong to one calculation."""
        for step, frame in self.frames():
            by_spin: Dict[Optional[str], List[Projection]] = {}
            for p in frame:
                by_spin.setdefault(p.spin, []).append(p)
            for spin, plist in by_spin.items():
                ref = plist[0]
                for p in plist[1:]:
                    if not np.isclose(p.efermi, ref.efermi, atol=1e-4):
                        raise ValueError(
                            "Inconsistent Fermi energies between input files: "
                            f"{ref.path} has {ref.efermi:.6f} eV but {p.path} has "
                            f"{p.efermi:.6f} eV (step {step}"
                            + (f", spin {spin}" if spin else "")
                            + "). All files must come from the same calculation; "
                            "process separate calculations separately."
                        )
                    if p.kind == "rspace" or ref.kind == "rspace":
                        continue
                    if p.n_states != ref.n_states or not np.allclose(
                        p.eigenvalues, ref.eigenvalues, atol=1e-4
                    ):
                        raise ValueError(
                            f"{p.path} and {ref.path} list different eigenvalues "
                            f"(step {step}); they are not from the same calculation."
                        )
        # The same kind must expose the same orbital columns in every step.
        cols: Dict[Tuple[str, Optional[str]], List[str]] = {}
        for p in self.projections:
            key = (p.name, p.spin)
            if key in cols and cols[key] != p.labels:
                raise ValueError(
                    f"orbital columns of {p.name} change between files/steps: "
                    f"{cols[key]} vs {p.labels}"
                )
            cols[key] = p.labels
        seen = set()
        for p in self.projections:
            key = (p.name, p.spin, p.step, p.kind)
            if key in seen:
                raise ValueError(
                    f"{p.name} (spin {p.spin}, step {p.step}) was given twice: {p.path}"
                )
            seen.add(key)

    # -- inspection -------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self.projections)

    def __iter__(self) -> Iterator[Projection]:
        return iter(self.projections)

    @property
    def names(self) -> List[str]:
        out: List[str] = []
        for p in self.projections:
            if p.name not in out:
                out.append(p.name)
        return out

    @property
    def spins(self) -> List[Optional[str]]:
        out: List[Optional[str]] = []
        for p in self.projections:
            if p.spin not in out:
                out.append(p.spin)
        return out

    @property
    def steps(self) -> List[int]:
        return sorted({p.step for p in self.projections})

    @property
    def is_spin_polarised(self) -> bool:
        return any(s is not None for s in self.spins)

    @property
    def paths(self) -> List[str]:
        out: List[str] = []
        for p in self.projections:
            if p.path and p.path not in out:
                out.append(p.path)
        return out

    def frames(self) -> Iterator[Tuple[int, List[Projection]]]:
        """Yield ``(step, projections of that step)`` in step order."""
        by_step: Dict[int, List[Projection]] = {}
        for p in self.projections:
            by_step.setdefault(p.step, []).append(p)
        for step in sorted(by_step):
            yield step, by_step[step]

    def select(
        self,
        names: Optional[Sequence[str]] = None,
        elements: Optional[Sequence[str]] = None,
        spin: Optional[str] = "any",
        step: Optional[int] = None,
        kind: Optional[str] = None,
    ) -> "PDOSSet":
        """A subset. ``names`` are kind / list names, ``kind`` is the
        projection type (``"kind"``, ``"list"``, ``"rspace"``)."""
        out = []
        for p in self.projections:
            if names is not None and p.name not in names:
                continue
            if elements is not None and p.element not in elements:
                continue
            if spin != "any" and p.spin != spin:
                continue
            if step is not None and p.step != step:
                continue
            if kind is not None and p.kind != kind:
                continue
            out.append(p)
        if not out:
            raise ValueError("the selection matches no projection")
        return PDOSSet(out, check=False)

    def completeness(self) -> Optional[float]:
        """Mean per-state weight summed over all *kind* projections.

        CP2K's projection is orthonormal, so if the files of every atomic
        kind are loaded each state sums to 1. A value clearly below 1 means
        that some kind files are missing and a "total DOS" built from this
        set would be incomplete. ``None`` if there are no kind files.
        """
        step, frame = next(self.frames())
        vals = []
        for spin in {p.spin for p in frame}:
            kinds = [p for p in frame if p.kind == "kind" and p.spin == spin]
            if kinds:
                vals.append(float(np.mean(sum(p.total for p in kinds))))
        return float(np.mean(vals)) if vals else None

    # -- energies ---------------------------------------------------------- #
    def edges(self, step: Optional[int] = None) -> List[BandEdges]:
        """Band edges per spin channel (first step unless ``step`` given)."""
        for s, frame in self.frames():
            if step is None or s == step:
                return _frame_edges(frame)
        raise ValueError(f"no step {step} in this set")

    def reference_energy(self, reference: Reference = "fermi", step: Optional[int] = None) -> float:
        """Absolute energy (eV) of the requested reference level."""
        return resolve_reference(self.edges(step), reference)[0]

    # -- broadening -------------------------------------------------------- #
    def broaden(
        self,
        sigma: Optional[float] = None,
        de: float = 0.01,
        reference: Reference = "fermi",
        *,
        fwhm: Optional[float] = None,
        lineshape: Union[str, LineShape] = "gaussian",
        eta: float = 0.5,
        gamma: Optional[float] = None,
        emin: Optional[float] = None,
        emax: Optional[float] = None,
        group_by: GroupSpec = "kind",
        groups: Optional[Mapping[str, Sequence[str]]] = None,
        orbitals: str = "l",
        total: bool = False,
        weighting: str = "all",
        natoms: Optional[Mapping[str, int]] = None,
        per_atom: bool = False,
        scale: float = 1.0,
        align: str = "frame",
        method: str = "auto",
    ) -> DOSCurves:
        """Broaden the state-resolved data into curves.

        If the set holds several steps (an MD trajectory), the curves are
        averaged over them.

        Args:
            sigma: Gaussian standard deviation, eV (default 0.1 if neither
                ``sigma`` nor ``fwhm`` is given).
            de: grid spacing, eV.
            reference: zero of the energy axis: ``"fermi"``, ``"vbm"``,
                ``"cbm"``, ``"midgap"``, ``"auto"`` (Fermi level for smeared
                occupations, VBM otherwise -- CP2K's own rule),
                ``"absolute"``, or a number in eV.
            fwhm: width as full width at half maximum instead of ``sigma``.
            lineshape: ``"gaussian"``, ``"lorentzian"``, ``"pseudo-voigt"``,
                ``"voigt"`` or a ready :class:`LineShape`.
            eta: Lorentzian fraction (pseudo-Voigt).
            gamma: Lorentzian HWHM in eV (true Voigt).
            emin, emax: grid limits on the *shifted* axis; default is the
                full range of eigenvalues plus padding.
            group_by, groups: how projections are merged into curves; see
                :func:`make_grouper`.
            orbitals: ``"l"`` (s, p, d; m components summed), ``"m"`` (keep
                the columns of the file) or ``"sum"`` (one curve per group).
            total: also build the total DOS (sum over atomic kinds).
            weighting: ``"all"`` states, or only the ``"occupied"`` /
                ``"unoccupied"`` part (weights times occupation fraction).
            natoms: ``{kind or element: number of atoms}`` for ``per_atom``.
            per_atom: divide each group by its number of atoms.
            scale: constant factor (2 turns a spin-restricted DOS into
                states of both spins per eV).
            align: for several steps, ``"frame"`` shifts every step by its
                own reference before averaging; ``"first"`` uses the first
                step's reference for all.
            method: broadening algorithm, see :func:`broadening.broaden`.
        """
        shape = _as_lineshape(lineshape, sigma, fwhm, eta, gamma)
        if align not in ("frame", "first"):
            raise ValueError("align must be 'frame' or 'first'")
        frames = [frame for _, frame in self.frames()]
        refs: List[float] = []
        label = "absolute"
        for i, frame in enumerate(frames):
            if align == "first" and i > 0:
                refs.append(refs[0])
                continue
            energy, label = resolve_reference(_frame_edges(frame), reference)
            refs.append(energy)

        lo, hi = _energy_window(zip(frames, refs), shape)
        grid = make_grid(lo if emin is None else emin, hi if emax is None else emax, de)
        acc = _Accumulator(
            grid, shape, make_grouper(group_by, groups), orbitals, weighting, total,
            natoms, per_atom, method,
        )
        for frame, ref in zip(frames, refs):
            acc.add(frame, ref)
        curves = acc.result(label, float(np.mean(refs)), self._meta(frames[0], refs))
        return curves.scaled(scale) if scale != 1.0 else curves

    def _meta(self, frame: Sequence[Projection], refs: Sequence[float]) -> Dict[str, object]:
        edges = _frame_edges(frame)
        meta: Dict[str, object] = {"files": ", ".join(os.path.basename(p) for p in self.paths)}
        ref0 = refs[0]
        meta["efermi"] = max(e.efermi for e in edges) - ref0
        vb = [e.vbm for e in edges if e.vbm is not None]
        cb = [e.cbm for e in edges if e.cbm is not None]
        meta["vbm"] = (max(vb) - ref0) if vb else None
        meta["cbm"] = (min(cb) - ref0) if cb else None
        meta["metallic"] = any(e.fractional for e in edges)
        return meta


def _as_lineshape(
    lineshape: Union[str, LineShape],
    sigma: Optional[float],
    fwhm: Optional[float],
    eta: float,
    gamma: Optional[float],
) -> LineShape:
    if isinstance(lineshape, LineShape):
        return lineshape
    if sigma is None and fwhm is None:
        sigma = 0.1
    return LineShape(lineshape, sigma=sigma, fwhm=fwhm, eta=eta, gamma=gamma)


# --------------------------------------------------------------------------- #
# Streaming over long trajectories
# --------------------------------------------------------------------------- #
def _stream_frames(paths: Sequence[str], steps: Optional[slice]) -> Iterator[List[Projection]]:
    """Yield frames from several ``APPEND``-ed files read in lock-step."""
    iterators = [iter_pdos(p) for p in paths]
    start = (steps.start or 0) if steps else 0
    stop = steps.stop if steps else None
    stride = (steps.step or 1) if steps else 1
    if start < 0 or (stop is not None and stop < 0) or stride < 1:
        raise ValueError("streaming needs a forward step selection with non-negative bounds")
    i = 0
    while True:
        frame = []
        for it in iterators:
            block = next(it, None)
            if block is not None:
                frame.append(block)
        if not frame:
            return
        if len(frame) != len(iterators):
            raise ValueError("the PDOS files hold different numbers of steps")
        if stop is not None and i >= stop:
            return
        if i >= start and (i - start) % stride == 0:
            yield frame
        i += 1


def _peek(iterator: Iterator) -> List:
    """First item of an iterator as a list (empty if there is none)."""
    for item in iterator:
        return [item]
    return []


def stream_broaden(
    paths: Iterable[str],
    sigma: Optional[float] = None,
    de: float = 0.01,
    reference: Reference = "fermi",
    *,
    steps: Optional[slice] = None,
    emin: Optional[float] = None,
    emax: Optional[float] = None,
    fwhm: Optional[float] = None,
    lineshape: Union[str, LineShape] = "gaussian",
    eta: float = 0.5,
    gamma: Optional[float] = None,
    group_by: GroupSpec = "kind",
    groups: Optional[Mapping[str, Sequence[str]]] = None,
    orbitals: str = "l",
    total: bool = False,
    weighting: str = "all",
    natoms: Optional[Mapping[str, int]] = None,
    per_atom: bool = False,
    scale: float = 1.0,
    method: str = "auto",
) -> DOSCurves:
    """Trajectory-averaged PDOS with memory bounded by a single step.

    Equivalent to ``PDOSSet.from_files(paths, steps).broaden(...)`` for
    files written with ``APPEND`` during MD, but only one step of each file
    is in memory at any time. Every step is aligned to its own reference
    before averaging.

    If ``emin`` / ``emax`` are not both given the files are read twice (once
    to find the energy range).
    """
    paths = expand_paths(paths)
    shape = _as_lineshape(lineshape, sigma, fwhm, eta, gamma)

    def framed() -> Iterator[Tuple[List[Projection], float, str]]:
        for frame in _stream_frames(paths, steps):
            energy, label = resolve_reference(_frame_edges(frame), reference)
            yield frame, energy, label

    if emin is None or emax is None:
        seen = [(f, r) for f, r, _ in _peek(framed())]
        if not seen:
            raise ValueError("the step selection matches no step")
        lo, hi = _energy_window(((f, r) for f, r, _ in framed()), shape)
        emin = lo if emin is None else emin
        emax = hi if emax is None else emax
    grid = make_grid(emin, emax, de)
    acc = _Accumulator(
        grid, shape, make_grouper(group_by, groups), orbitals, weighting, total,
        natoms, per_atom, method,
    )
    refs: List[float] = []
    label = "absolute"
    meta: Dict[str, object] = {}
    for frame, energy, label in framed():  # noqa: B007 (label is used below)
        if not refs:
            edges = _frame_edges(frame)
            vb = [e.vbm for e in edges if e.vbm is not None]
            cb = [e.cbm for e in edges if e.cbm is not None]
            meta = {
                "files": ", ".join(os.path.basename(p) for p in paths),
                "efermi": max(e.efermi for e in edges) - energy,
                "vbm": (max(vb) - energy) if vb else None,
                "cbm": (min(cb) - energy) if cb else None,
                "metallic": any(e.fractional for e in edges),
            }
        acc.add(frame, energy)
        refs.append(energy)
    if not refs:
        raise ValueError("the step selection matches no step")
    curves = acc.result(label, float(np.mean(refs)), meta)
    return curves.scaled(scale) if scale != 1.0 else curves


# --------------------------------------------------------------------------- #
# Curves that CP2K has already broadened (>= 2026.2)
# --------------------------------------------------------------------------- #
class _NamedCurve:
    """Minimal stand-in for a Projection so that groupers work on curves."""

    def __init__(self, cf: CurveFile) -> None:
        self.name = cf.name or os.path.basename(cf.path)
        self.kind = "list" if self.name.startswith("list") else "kind"
        self.path = cf.path
        self.element = element_of(self.name) if self.kind == "kind" else self.name


def curves_from_cp2k(
    paths: Iterable[str],
    group_by: GroupSpec = "kind",
    groups: Optional[Mapping[str, Sequence[str]]] = None,
    orbitals: str = "l",
    total: bool = False,
    step: int = -1,
) -> DOSCurves:
    """Collect CP2K's own broadened ``*_curve*.pdos`` files into curves.

    This is the route for k-point calculations, for which CP2K writes only
    broadened PDOS. Nothing is re-broadened or re-referenced: the energy
    axis and its zero are taken as CP2K wrote them (converted to eV if the
    file is in Hartree).

    Args:
        step: which step of each file to use (default: the last).
    """
    if orbitals not in ("l", "m", "sum"):
        raise ValueError("orbitals must be 'l', 'm' or 'sum'")
    files = [read_curve(p)[step] for p in expand_paths(paths)]
    first = files[0]
    escale = 1.0 if first.energy_unit == "eV" else HARTREE_TO_EV
    energy = first.energy * escale
    out: "OrderedDict[Key, np.ndarray]" = OrderedDict()
    tot: Dict[Optional[str], np.ndarray] = {}

    grouper = make_grouper(group_by, groups)
    is_kind = [not (cf.name or "").startswith("list") for cf in files]
    for cf, kindfile in zip(files, is_kind):
        scale = 1.0 if cf.energy_unit == "eV" else 1.0 / HARTREE_TO_EV
        e = cf.energy * (1.0 if cf.energy_unit == "eV" else HARTREE_TO_EV)
        same = e.shape == energy.shape and np.allclose(e, energy, atol=1e-6)
        group = grouper(_NamedCurve(cf))  # type: ignore[arg-type]
        ysum = np.zeros_like(energy)
        for label, y in cf.columns.items():
            if label.lower() == "total":
                continue
            y = y * scale
            if not same:
                y = np.interp(energy, e, y, left=0.0, right=0.0)
            ysum += y
            lab = "" if orbitals == "sum" else (l_of(label) if orbitals == "l" else label)
            if group is not None:
                key = (group, lab, cf.spin)
                out[key] = out[key] + y if key in out else y.copy()
        if total and (kindfile or not any(is_kind)):
            tot[cf.spin] = tot.get(cf.spin, 0.0) + ysum
    for spin, y in tot.items():
        out[(TOTAL, "", spin)] = y
    zero = (first.energy_zero or "").split("->")[-1].strip().upper()
    reference = {"HOCO": "vbm", "FERMI": "fermi", "ABSOLUTE": "absolute"}.get(zero, "absolute")
    ref_energy = {"vbm": first.ehoco, "fermi": first.efermi}.get(reference) or 0.0
    meta: Dict[str, object] = {
        "files": ", ".join(os.path.basename(cf.path) for cf in files),
        "lineshape": "as broadened by CP2K"
        + (f" ({first.broadening['type']}, FWHM = {first.broadening['fwhm_eV']:g} eV)"
           if first.broadening else ""),
        "n_frames": 1,
    }
    if first.efermi is not None:
        meta["efermi"] = first.efermi - ref_energy
    if first.ehoco is not None:
        meta["vbm"] = first.ehoco - ref_energy
    return DOSCurves(energy, out, reference, float(ref_energy), "states/eV", meta)
