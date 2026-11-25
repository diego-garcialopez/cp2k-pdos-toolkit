"""Readers for the DOS-related files CP2K writes.

Supported files
---------------
``*.pdos`` (state-resolved, "raw")
    Written by ``&DFT/&PRINT/&PDOS``: one file per atomic kind (``-k<N>``),
    per atom list (``-list<N>``, from ``&LDOS``) or per real-space region
    (``-r_list<N>``, from ``&R_LDOS``), and per spin channel (``ALPHA_`` /
    ``BETA_`` prefix). Both header layouts are understood:

    * the classic one, with the Fermi energy on the first line
      (CP2K up to and including 2026.1), and
    * the layout of CP2K 2026.2 and later, with separate ``E(Fermi)`` /
      ``E(HOCO)`` lines and an extra ``Total`` column.

    Files written with ``APPEND`` (several MD / geometry steps in one file)
    are split into one :class:`Projection` per step.

``*_curve*.pdos`` and ``*-curve*.dos`` (broadened curves, CP2K >= 2026.2)
    Already broadened, directly plottable curves. This is also the only PDOS
    output CP2K offers for k-point calculations.

``*.dos``
    Total DOS histogram / state list from ``&DFT/&PRINT/&DOS``.

Band-structure files (``*.bs``) are handled by :mod:`pdos_toolkit.bands`.

A note on file names: the ``k`` in ``project-k1-1.pdos`` stands for *kind*
(the first atomic kind of the input), **not** for a k-point.
"""

from __future__ import annotations

import glob as _glob
import io as _io
import os
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .constants import HARTREE_TO_EV, element_of, l_of

__all__ = [
    "Projection",
    "CurveFile",
    "DOSFile",
    "PDOSFormatError",
    "iter_pdos",
    "read_pdos",
    "read_pdos_files",
    "read_curve",
    "read_dos",
    "find_pdos_files",
    "parse_pdos_filename",
    "is_curve_file",
]


class PDOSFormatError(ValueError):
    """Raised when a file does not look like the CP2K output it should be."""


# --------------------------------------------------------------------------- #
# Header patterns
# --------------------------------------------------------------------------- #
_FLOAT = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[EeDd][-+]?\d+)?"

_TITLE_RE = re.compile(
    r"^#\s*Projected DOS\s+(?:"
    r"for atomic kind\s+(?P<kind>\S+)"
    r"|for list\s+(?P<list>\d+)\s+of\s+(?P<natoms>\d+)\s+atoms,?"
    r"|in real space, using\s+(?P<npoints>\d+)\s+points of the grid,"
    r"\s*and eval in the range\s*(?P<r1>\S+)\s+(?P<r2>\S+)\s+Hartree,?"
    r")"
    r"(?:\s+at iteration step i =\s*(?P<step>\d+))?"
    r"(?:,?\s*E\(Fermi\)\s*=\s*(?P<efermi>\S+)\s+a\.u\.)?",
)
_EFERMI_RE = re.compile(r"^#\s*E\(Fermi\)\s*=\s*(?P<v>.+?)\s+a\.u\.")
_EHOCO_RE = re.compile(r"^#\s*E\(HOCO\)\s*=\s*(?P<v>.+?)\s+a\.u\.")
_ZERO_RE = re.compile(r"^#\s*Energy zero:\s*(?P<v>.+?)\s*$")
_BROADEN_RE = re.compile(
    r"^#\s*(?P<type>Gaussian|Lorentzian|Pseudo-Voigt) broadening, FWHM =\s*"
    r"(?P<au>\S+)\s+a\.u\.\s*=\s*(?P<ev>\S+)\s+eV"
    r"(?:, Lorentzian fraction =\s*(?P<eta>\S+))?"
)
_FILENAME_RE = re.compile(
    r"(?:(?P<spin>ALPHA|BETA)_)?"
    r"(?P<type>r_list|list|k)(?P<index>\d+)"
    r"(?P<curve>_curve)?"
    r"-(?P<iter>\d+)(?:_(?P<step>\d+))?\.pdos$"
)


def _to_float(text: str) -> float:
    """Parse a Fortran-formatted real (handles ``D`` exponents, ``****``)."""
    text = text.strip()
    if not text or set(text) == {"*"}:
        return float("nan")
    return float(text.replace("D", "E").replace("d", "e"))


def parse_pdos_filename(path: str) -> Dict[str, object]:
    """Extract what CP2K encodes in a PDOS file name.

    Returns a dict with keys ``spin`` (``"alpha"``, ``"beta"`` or ``None``),
    ``type`` (``"kind"``, ``"list"``, ``"rspace"`` or ``None``), ``index``
    (kind / list number), ``curve`` (bool) and ``step`` (MD step encoded in
    the name, or ``None``). Unrecognised names give all-``None`` values, so
    renamed files still load.
    """
    m = _FILENAME_RE.search(os.path.basename(path))
    if not m:
        return {"spin": None, "type": None, "index": None, "curve": False, "step": None}
    kinds = {"k": "kind", "list": "list", "r_list": "rspace"}
    return {
        "spin": m.group("spin").lower() if m.group("spin") else None,
        "type": kinds[m.group("type")],
        "index": int(m.group("index")),
        "curve": bool(m.group("curve")),
        "step": int(m.group("step")) if m.group("step") else None,
    }


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #
@dataclass
class Projection:
    """State-resolved projected DOS of one kind / list, one spin, one step.

    Energies are stored in **eV** (absolute Kohn-Sham eigenvalues, i.e. not
    shifted to any reference). ``weights[i, j]`` is the projection of state
    ``i`` onto orbital column ``labels[j]``.
    """

    name: str  #: kind name (``"Ni1"``), ``"list3"`` or ``"r_list1"``
    kind: str  #: ``"kind"``, ``"list"`` or ``"rspace"``
    eigenvalues: np.ndarray  #: (n_states,) in eV
    occupations: np.ndarray  #: (n_states,)
    weights: np.ndarray  #: (n_states, n_columns)
    labels: List[str]  #: orbital label per weight column
    efermi: float  #: Fermi energy, eV
    ehoco: Optional[float] = None  #: highest occupied state as reported, eV
    spin: Optional[str] = None  #: ``"alpha"``, ``"beta"`` or ``None``
    step: int = 0  #: iteration (MD / GEO_OPT) step
    index: Optional[int] = None  #: kind / list number from the file name
    natoms: Optional[int] = None  #: atoms in the list (LDOS files only)
    mo_index: Optional[np.ndarray] = None  #: CP2K MO numbers
    path: str = ""
    layout: str = "classic"  #: ``"classic"`` or ``"2026"``
    meta: Dict[str, object] = field(default_factory=dict)

    # -- convenience ------------------------------------------------------ #
    @property
    def element(self) -> str:
        """Chemical element the kind name refers to (see ``element_of``)."""
        return element_of(self.name) if self.kind == "kind" else self.name

    @property
    def n_states(self) -> int:
        return int(self.eigenvalues.shape[0])

    @property
    def has_components(self) -> bool:
        """True when columns are m-resolved (``px``, ``d-2``, ...)."""
        return any(len(lab) > 1 and l_of(lab) != lab for lab in self.labels)

    @property
    def total(self) -> np.ndarray:
        """Per-state weight summed over all orbital columns."""
        return self.weights.sum(axis=1)

    def by_l(self) -> Tuple[List[str], np.ndarray]:
        """Collapse m-resolved columns to angular momentum (s, p, d, ...).

        Returns ``(labels, weights)``; a no-op for files that are already
        l-resolved.
        """
        order: List[str] = []
        for lab in self.labels:
            ll = l_of(lab)
            if ll not in order:
                order.append(ll)
        out = np.zeros((self.n_states, len(order)))
        for j, lab in enumerate(self.labels):
            out[:, order.index(l_of(lab))] += self.weights[:, j]
        return order, out

    def column(self, label: str) -> np.ndarray:
        """Weights of one orbital label; an l letter sums its m components."""
        if label in self.labels:
            return self.weights[:, self.labels.index(label)]
        cols = [j for j, lab in enumerate(self.labels) if l_of(lab) == label]
        if not cols:
            raise KeyError(f"{self.name}: no orbital column {label!r} (have {self.labels})")
        return self.weights[:, cols].sum(axis=1)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        spin = f", spin={self.spin}" if self.spin else ""
        return (
            f"Projection({self.name!r}, {self.kind}{spin}, step={self.step}, "
            f"{self.n_states} states, columns={self.labels})"
        )


@dataclass
class CurveFile:
    """A broadened DOS/PDOS curve as written by CP2K >= 2026.2."""

    title: str
    energy: np.ndarray  #: grid, in ``energy_unit``, already shifted to ``energy_zero``
    columns: Dict[str, np.ndarray]
    energy_unit: str = "eV"
    energy_zero: Optional[str] = None
    efermi: Optional[float] = None  #: eV
    ehoco: Optional[float] = None  #: eV
    broadening: Optional[Dict[str, object]] = None
    name: Optional[str] = None
    spin: Optional[str] = None
    step: int = 0
    path: str = ""


@dataclass
class DOSFile:
    """Total DOS from a CP2K ``.dos`` file (one block / iteration step).

    ``density`` and ``occupied`` map a spin label (``"total"``, ``"alpha"``,
    ``"beta"``) to arrays on ``energy`` (same keys in ``energies`` when the
    two spin channels come with separate energy columns).
    """

    energy: np.ndarray  #: eV
    density: Dict[str, np.ndarray]
    occupied: Dict[str, np.ndarray]
    efermi: Optional[Tuple[float, ...]] = None  #: eV, one value per spin
    ehoco: Optional[Tuple[float, ...]] = None  #: eV
    step: int = 0
    energy_zero: Optional[str] = None
    broadening: Optional[Dict[str, object]] = None
    energies: Dict[str, np.ndarray] = field(default_factory=dict)
    file_unit: str = "a.u."
    path: str = ""


# --------------------------------------------------------------------------- #
# Low-level block reader
# --------------------------------------------------------------------------- #
def _iter_blocks(path: str, title_prefix: str) -> Iterator[Tuple[List[str], List[str]]]:
    """Yield ``(header_lines, data_lines)`` for every block of a file.

    A block starts at a comment line beginning with ``title_prefix`` and
    runs to the next such line. Only one block is held in memory at a time,
    which keeps memory bounded for long ``APPEND``-ed trajectories.
    """
    header: List[str] = []
    data: List[str] = []
    seen = False
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            if line.startswith("#") or line.lstrip().startswith("#"):
                stripped = line.strip()
                if stripped.startswith(title_prefix):
                    if seen:
                        yield header, data
                    header, data, seen = [stripped], [], True
                elif seen:
                    header.append(stripped)
            elif seen and "Occupation" in line and not data:
                # A column-header line that lost its leading '#' (hand-edited
                # or tool-generated files).
                header.append("# " + line.strip())
            elif line.strip():
                if not seen:
                    raise PDOSFormatError(
                        f"{path}: data found before any '{title_prefix}...' header line"
                    )
                data.append(line)
    if seen:
        yield header, data


def _load_table(lines: Sequence[str], path: str) -> np.ndarray:
    """Turn data lines into a 2-D float array, tolerating Fortran quirks."""
    if not lines:
        return np.empty((0, 0))
    try:
        return np.loadtxt(_io.StringIO("".join(lines)), ndmin=2)
    except ValueError:
        pass
    rows = [[_to_float(tok) for tok in ln.split()] for ln in lines]
    width = max(len(r) for r in rows)
    if any(len(r) != width for r in rows):
        raise PDOSFormatError(f"{path}: rows with differing numbers of columns")
    return np.asarray(rows, dtype=float)


def _column_labels(header: Sequence[str]) -> Tuple[List[str], int]:
    """Find the column-header line; return (orbital labels, n leading cols).

    Works for both ``#  MO Eigenvalue [a.u.]  Occupation  s p d`` and
    ``#  MO Energy[a.u.]  Occupation  Total  s p d``. The labels are located
    *after the word* ``Occupation`` rather than by counting whitespace
    tokens, because ``Eigenvalue [a.u.]`` contains a space.
    """
    for line in header:
        if "Occupation" in line and "MO" in line.split():
            labels = line.split("Occupation", 1)[1].split()
            return labels, 3
    raise PDOSFormatError("no 'MO ... Occupation ...' column-header line found")


def iter_pdos(path: str) -> Iterator[Projection]:
    """Iterate over the :class:`Projection` blocks of a raw ``.pdos`` file.

    One block per iteration step is yielded; a normal single-point file
    yields exactly one. Use this rather than :func:`read_pdos` to stream a
    long ``APPEND``-ed trajectory without loading it all.

    Raises:
        PDOSFormatError: if the file is not a CP2K state-resolved PDOS file.
    """
    info = parse_pdos_filename(path)
    n_blocks = 0
    for header, data in _iter_blocks(path, "# Projected DOS"):
        n_blocks += 1
        m = _TITLE_RE.match(header[0])
        if not m:
            raise PDOSFormatError(f"{path}: unrecognised PDOS header: {header[0]!r}")
        if any(_ZERO_RE.match(h) or h.startswith("# Energy[") for h in header[1:]):
            raise PDOSFormatError(
                f"{path}: this is a broadened CP2K curve file; read it with read_curve()"
            )

        efermi_ha: Optional[float] = None
        ehoco_ha: Optional[float] = None
        layout = "classic"
        if m.group("efermi"):
            efermi_ha = _to_float(m.group("efermi"))
        for h in header[1:]:
            mf = _EFERMI_RE.match(h)
            if mf:
                efermi_ha = _to_float(mf.group("v").split()[0])
                layout = "2026"
            mh = _EHOCO_RE.match(h)
            if mh:
                ehoco_ha = _to_float(mh.group("v").split()[0])
        if efermi_ha is None:
            raise PDOSFormatError(f"{path}: no Fermi energy found in the header")

        try:
            labels, lead = _column_labels(header)
        except PDOSFormatError as exc:
            raise PDOSFormatError(f"{path}: {exc}") from None

        table = _load_table(data, path)
        if table.size == 0:
            table = np.empty((0, lead + len(labels)))

        meta: Dict[str, object] = {}
        if m.group("kind"):
            ptype, name, natoms = "kind", m.group("kind"), None
        elif m.group("list"):
            ptype, name, natoms = "list", f"list{m.group('list')}", int(m.group("natoms"))
        else:
            ptype, natoms = "rspace", None
            name = f"r_list{info['index']}" if info["index"] else "r_list"
            meta["npoints"] = int(m.group("npoints"))
            meta["eval_range_eV"] = (
                _to_float(m.group("r1")) * HARTREE_TO_EV,
                _to_float(m.group("r2")) * HARTREE_TO_EV,
            )

        weights = table[:, lead:]
        if ptype == "rspace":
            # Two numbers per state: the integrated |psi|^2 in the region and
            # the same scaled by the total number of grid points.
            labels = ["LDOS", "LDOS_scaled"][: weights.shape[1]]
        elif labels and labels[0].lower() == "total":
            # CP2K >= 2026.2 prints the row sum first; keep only the
            # orbital columns so that nothing is counted twice.
            labels = labels[1:]
            weights = weights[:, 1:]
            layout = "2026"

        if weights.shape[1] != len(labels):
            raise PDOSFormatError(
                f"{path}: {len(labels)} orbital labels {labels} but "
                f"{weights.shape[1]} data columns"
            )

        step = int(m.group("step")) if m.group("step") else int(info["step"] or 0)

        yield Projection(
            name=name,
            kind=ptype,
            eigenvalues=table[:, 1] * HARTREE_TO_EV,
            occupations=table[:, 2].copy(),
            weights=np.ascontiguousarray(weights),
            labels=list(labels),
            efermi=efermi_ha * HARTREE_TO_EV,
            ehoco=None if ehoco_ha is None else ehoco_ha * HARTREE_TO_EV,
            spin=info["spin"],
            step=step,
            index=info["index"],
            natoms=natoms,
            mo_index=table[:, 0].astype(int),
            path=str(path),
            layout=layout,
            meta=meta,
        )
    if n_blocks == 0:
        raise PDOSFormatError(
            f"{path}: not a CP2K PDOS file (no '# Projected DOS ...' header line)"
        )


def read_pdos(path: str) -> List[Projection]:
    """Read every step of a raw CP2K ``.pdos`` file into memory."""
    return list(iter_pdos(str(path)))


def _select_steps(n: int, steps: Optional[slice]) -> range:
    return range(n)[steps] if steps is not None else range(n)


def read_pdos_files(
    paths: Iterable[str],
    steps: Optional[slice] = None,
    last_only: bool = False,
) -> List[Projection]:
    """Read several raw PDOS files (kinds, lists, spin channels, steps).

    Args:
        paths: file names; shell-style patterns are expanded.
        steps: optional slice applied to the blocks of *each* file
            (e.g. ``slice(100, None, 10)`` to skip equilibration).
        last_only: keep only the final block of each file.
    """
    out: List[Projection] = []
    for path in expand_paths(paths):
        blocks = read_pdos(path)
        if last_only:
            blocks = blocks[-1:]
        elif steps is not None:
            blocks = [blocks[i] for i in _select_steps(len(blocks), steps)]
        out.extend(blocks)
    return out


def expand_paths(paths: Iterable[str]) -> List[str]:
    """Expand glob patterns, keeping the order given and dropping duplicates."""
    out: List[str] = []
    for p in paths:
        p = str(p)
        matches = sorted(_glob.glob(p), key=_natural_key) if _glob.has_magic(p) else [p]
        if not matches:
            raise FileNotFoundError(f"no file matches {p!r}")
        for q in matches:
            if q not in out:
                out.append(q)
    return out


def _natural_key(text: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", text)]


def find_pdos_files(
    directory: str = ".", project: Optional[str] = None, include_curves: bool = False
) -> List[str]:
    """Locate the PDOS files of a calculation in ``directory``.

    Args:
        directory: where to look (not recursive).
        project: CP2K ``PROJECT`` name; all projects if omitted.
        include_curves: also return broadened ``*_curve*`` files.

    Returns:
        Naturally sorted paths (``k2`` before ``k10``).
    """
    pattern = os.path.join(directory, f"{_glob.escape(project) if project else ''}*.pdos")
    found = []
    for path in sorted(_glob.glob(pattern), key=_natural_key):
        info = parse_pdos_filename(path)
        if info["type"] is None:
            continue
        if info["curve"] and not include_curves:
            continue
        found.append(path)
    return found


def is_curve_file(path: str) -> bool:
    """True when ``path`` is a broadened CP2K (>= 2026.2) curve file."""
    try:
        with open(path, "r", errors="replace") as fh:
            for _ in range(12):
                line = fh.readline()
                if not line or not line.lstrip().startswith("#"):
                    break
                if _ZERO_RE.match(line.strip()) or _BROADEN_RE.match(line.strip()):
                    return True
                if line.lstrip().startswith("# Energy[") and "total" in line.split():
                    return True
    except OSError:
        return False
    return False


# --------------------------------------------------------------------------- #
# Broadened curve files (CP2K >= 2026.2)
# --------------------------------------------------------------------------- #
def _broadening_info(header: Sequence[str]) -> Optional[Dict[str, object]]:
    for h in header:
        m = _BROADEN_RE.match(h)
        if m:
            return {
                "type": m.group("type").lower(),
                "fwhm_eV": _to_float(m.group("ev")),
                "eta": _to_float(m.group("eta")) if m.group("eta") else None,
            }
    return None


def read_curve(path: str) -> List[CurveFile]:
    """Read a broadened ``*_curve*.pdos`` file written by CP2K >= 2026.2.

    These files hold ready-made curves (``Energy total s p d ...``) on an
    energy axis that CP2K has already shifted to the reference named in the
    ``# Energy zero:`` line. One :class:`CurveFile` is returned per step.
    """
    path = str(path)
    info = parse_pdos_filename(path)
    out: List[CurveFile] = []
    for header, data in _iter_blocks(path, "# Projected DOS"):
        m = _TITLE_RE.match(header[0])
        colline = next((h for h in header if h.startswith("# Energy[")), None)
        if colline is None:
            raise PDOSFormatError(f"{path}: not a broadened curve file (no '# Energy[..]' line)")
        tokens = colline[1:].split()
        unit = "eV" if "eV" in tokens[0] else "a.u."
        names = tokens[1:]
        table = _load_table(data, path)
        if table.shape[1] != len(names) + 1:
            raise PDOSFormatError(
                f"{path}: {len(names)} column labels but {table.shape[1] - 1} data columns"
            )
        efermi = ehoco = None
        zero = None
        for h in header[1:]:
            mf, mh, mz = _EFERMI_RE.match(h), _EHOCO_RE.match(h), _ZERO_RE.match(h)
            if mf:
                efermi = _to_float(mf.group("v").split()[0]) * HARTREE_TO_EV
            if mh:
                ehoco = _to_float(mh.group("v").split()[0]) * HARTREE_TO_EV
            if mz:
                zero = mz.group("v")
        name = None
        if m and m.group("kind"):
            name = m.group("kind")
        elif m and m.group("list"):
            name = f"list{m.group('list')}"
        out.append(
            CurveFile(
                title=header[0].lstrip("# "),
                energy=table[:, 0],
                columns={n: table[:, j + 1] for j, n in enumerate(names)},
                energy_unit=unit,
                energy_zero=zero,
                efermi=efermi,
                ehoco=ehoco,
                broadening=_broadening_info(header),
                name=name,
                spin=info["spin"],
                step=int(m.group("step")) if m and m.group("step") else 0,
                path=path,
            )
        )
    if not out:
        raise PDOSFormatError(f"{path}: not a CP2K curve file")
    return out


# --------------------------------------------------------------------------- #
# Total DOS (.dos)
# --------------------------------------------------------------------------- #
_DOS_TITLE_RE = re.compile(
    r"^#\s*DOS at iteration step i =\s*(?P<step>\d+)"
    r"(?:,\s*E_Fermi\[a\.u\.\]\s*=\s*(?P<ef>.+))?$"
)


def read_dos(path: str) -> List[DOSFile]:
    """Read a CP2K total-DOS file (``&DFT/&PRINT/&DOS``).

    Handles restricted and spin-polarised files, the Gamma-point and k-point
    variants, the classic and the 2026.2 header layout, and both the raw
    (``.dos``) and broadened (``-curve*.dos``) flavours. Energies are
    converted to eV; a density that CP2K printed per Hartree is converted to
    per eV so that the two always go together.
    """
    path = str(path)
    out: List[DOSFile] = []
    for header, data in _iter_blocks(path, "# DOS at iteration step"):
        m = _DOS_TITLE_RE.match(header[0])
        if not m:
            raise PDOSFormatError(f"{path}: unrecognised DOS header: {header[0]!r}")
        efermi: Optional[Tuple[float, ...]] = None
        ehoco: Optional[Tuple[float, ...]] = None
        zero = None
        if m.group("ef"):
            efermi = tuple(_to_float(t) * HARTREE_TO_EV for t in m.group("ef").split())
        colline = None
        for h in header[1:]:
            mf, mh, mz = _EFERMI_RE.match(h), _EHOCO_RE.match(h), _ZERO_RE.match(h)
            if mf:
                efermi = tuple(_to_float(t) * HARTREE_TO_EV for t in mf.group("v").split())
            elif mh:
                ehoco = tuple(_to_float(t) * HARTREE_TO_EV for t in mh.group("v").split())
            elif mz:
                zero = mz.group("v")
            elif "Energy[" in h:
                colline = h
        if colline is None:
            raise PDOSFormatError(f"{path}: no column-header line in DOS block")
        in_ev = "Energy[eV]" in colline
        escale = 1.0 if in_ev else HARTREE_TO_EV
        broad = _broadening_info(header)
        # Raw files list states / histogram counts (unit-free); only the
        # broadened curves are true densities per energy unit.
        dscale = 1.0 if (in_ev or broad is None) else 1.0 / HARTREE_TO_EV
        table = _load_table(data, path)
        ncol = table.shape[1] if table.size else 0
        spin = "Alpha_Density" in colline
        density: Dict[str, np.ndarray] = {}
        occupied: Dict[str, np.ndarray] = {}
        energies: Dict[str, np.ndarray] = {}
        if not spin:
            energy = table[:, 0] * escale
            density["total"] = table[:, 1] * dscale
            occupied["total"] = table[:, 2] * dscale
        elif ncol == 6:  # separate energy column per spin channel
            energy = table[:, 0] * escale
            energies = {"alpha": energy, "beta": table[:, 3] * escale}
            density = {"alpha": table[:, 1] * dscale, "beta": table[:, 4] * dscale}
            occupied = {"alpha": table[:, 2] * dscale, "beta": table[:, 5] * dscale}
        elif ncol == 5:
            energy = table[:, 0] * escale
            density = {"alpha": table[:, 1] * dscale, "beta": table[:, 3] * dscale}
            occupied = {"alpha": table[:, 2] * dscale, "beta": table[:, 4] * dscale}
        else:
            raise PDOSFormatError(f"{path}: unexpected number of columns ({ncol}) in DOS file")
        out.append(
            DOSFile(
                energy=energy,
                density=density,
                occupied=occupied,
                efermi=efermi,
                ehoco=ehoco,
                step=int(m.group("step")),
                energy_zero=zero,
                broadening=broad,
                energies=energies,
                file_unit="eV" if in_ev else "a.u.",
                path=path,
            )
        )
    if not out:
        raise PDOSFormatError(f"{path}: not a CP2K DOS file (no '# DOS at iteration step' line)")
    return out
