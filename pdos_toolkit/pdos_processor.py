#!/usr/bin/env python3
"""
CP2K PDOS Post-Processing Tool
===============================
Reads one or more CP2K ``*.pdos`` files (typically one per k-point, all for
the same atomic kind and the same electronic-structure calculation),
applies Gaussian broadening to turn the discrete eigenvalue/PDOS-weight
pairs into a continuous curve, and writes/plots the result.

This module merges and corrects two earlier prototypes (``pdos_plot_5.py``
and ``pdos_plot_6_memory.py``). See ``CHANGELOG.md`` / ``docs/theory.md`` in
this repository for a full account of what was wrong with those and why.

In one sentence, the two bugs that mattered most were:

1. The broadening weight was multiplied by the energy-grid spacing ``de``
   and by the Hartree->eV factor in the wrong direction, which made the
   reported PDOS *magnitude* depend on an arbitrary numerical choice
   (the plotting grid resolution) instead of being a physical quantity.
2. A malformed hex color code (``'#3CCB974'``, 7 hex digits instead of 6)
   crashed the plotting routine as soon as a 5th unique element appeared.

Both are fixed here, along with several smaller robustness issues
documented in the README.
"""

from __future__ import annotations

import sys
import re
import argparse
import itertools
import logging
from typing import Dict, List, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("pdos_toolkit")

# CODATA Hartree -> eV conversion factor.
HARTREE_TO_EV = 27.211386245988

# CP2K PDOS header, e.g.:
# "# Projected DOS for atomic kind Cu at iteration step i = 0, E(Fermi) = -0.123456 a.u."
HEADER_REGEX = re.compile(
    r"# Projected DOS for atomic kind (?P<element>\w+) at iteration step i = \d+, "
    r"E\(Fermi\) = [ \t]*(?P<Efermi>[^\t ]+) a\.u\."
)


def _stream_numeric_rows(filename: str, chunk_size: int = 2000):
    """Yield the numeric body of a CP2K PDOS file in chunks of ``chunk_size``
    rows, without ever holding the full file in memory as text or as one
    giant array.

    This is what a genuine "low-memory" mode looks like: data is read with
    ``itertools.islice`` directly off the file handle and only ``chunk_size``
    rows are materialized into a NumPy array at a time. (For contrast, the
    original "memory-optimized" prototype called ``np.loadtxt`` on the
    *entire* file first and only then sliced chunks out of the resulting
    in-memory array -- which does not reduce peak memory usage at all, and
    additionally re-read every file from disk two or three times.)
    """
    with open(filename, "r") as f:
        f.readline()  # header line 1
        f.readline()  # column-header line 2
        while True:
            lines = list(itertools.islice(f, chunk_size))
            if not lines:
                break
            yield np.loadtxt(lines, ndmin=2)


class PDOSProcessor:
    """Loads CP2K PDOS files and computes a Gaussian-broadened PDOS.

    All internal energy bookkeeping is done in eV. Files are only ever
    converted once, right after being read, which avoids the repeated
    Hartree<->eV round trips that caused the unit bug in the two original
    scripts.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.metadata: Dict[str, object] = {
            "elements": [],   # element symbol per PDOS data column
            "orbitals": [],   # orbital letter per PDOS data column
            "efermi_eV": None,
        }
        # One entry per input file: filename + column bookkeeping.
        # Numeric data itself is NOT kept resident; it is streamed on demand
        # via _stream_numeric_rows so memory use stays roughly constant in
        # the number of MOs times chunk_size, not in the total file size.
        self._files: List[dict] = []

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    def load_files(self) -> None:
        """Parse headers of all input files and validate consistency.

        Raises:
            ValueError: on malformed headers, missing 'Occupation' column,
                or inconsistent Fermi energies across files.
        """
        efermi_values_ha = []

        for filename in self.args.pdosfilenames:
            with open(filename, "r") as f:
                header_line = f.readline().strip()
                match = HEADER_REGEX.match(header_line)
                if not match:
                    raise ValueError(f"Invalid PDOS header in file: {filename}")

                element = match.group("element")
                efermi_ha = float(match.group("Efermi"))
                efermi_values_ha.append(efermi_ha)

                column_headers = f.readline().strip().split()
                try:
                    occupation_idx = column_headers.index("Occupation")
                except ValueError:
                    raise ValueError(
                        f"File {filename} is missing an 'Occupation' column header"
                    )

                orbital_cols = column_headers[occupation_idx + 1 :]
                orbital_types = [col[0].lower() for col in orbital_cols]

                self.metadata["elements"].extend([element] * len(orbital_types))
                self.metadata["orbitals"].extend(orbital_types)

                self._files.append(
                    {
                        "filename": filename,
                        "element": element,
                        "occupation_idx": occupation_idx,
                        "pdos_start_col": occupation_idx + 1,
                        "n_orbitals": len(orbital_types),
                    }
                )

        # Use a tolerance-based comparison: text-parsed floats from
        # independently-written files should not be compared with exact
        # equality (`set()`), since that is fragile to formatting/rounding
        # differences even when physically they are "the same" Fermi level.
        efermi_values_ha = np.asarray(efermi_values_ha)
        if not np.allclose(efermi_values_ha, efermi_values_ha[0], atol=1e-8):
            raise ValueError(
                "Inconsistent Fermi energies between input files: "
                f"{efermi_values_ha.tolist()} (Ha). All files must come "
                "from the same calculation / same electronic step."
            )
        self.metadata["efermi_eV"] = float(efermi_values_ha[0]) * HARTREE_TO_EV

    # ------------------------------------------------------------------ #
    # VBM
    # ------------------------------------------------------------------ #
    def calculate_vbm(self) -> float:
        """Streaming computation of the Valence Band Maximum (VBM), in eV.

        Returns:
            The highest eigenvalue (eV) among occupied states
            (occupation > 0) that lies below the Fermi level.
        """
        best_vbm = None
        efermi_eV = self.metadata["efermi_eV"]

        for finfo in self._files:
            occ_idx = finfo["occupation_idx"]
            for chunk in _stream_numeric_rows(finfo["filename"], self.args.chunk_size):
                eigen_eV = chunk[:, 1] * HARTREE_TO_EV
                occ = chunk[:, occ_idx]
                mask = (eigen_eV < efermi_eV) & (occ > 0)
                if np.any(mask):
                    chunk_max = eigen_eV[mask].max()
                    best_vbm = chunk_max if best_vbm is None else max(best_vbm, chunk_max)

        if best_vbm is None:
            raise ValueError("No occupied states below the Fermi level were found")
        return best_vbm

    # ------------------------------------------------------------------ #
    # Main processing
    # ------------------------------------------------------------------ #
    def process_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the Gaussian-broadened PDOS on a uniform energy grid.

        Returns:
            (energy_grid_eV, pdos_matrix) where ``pdos_matrix`` has shape
            (n_grid_points, n_orbital_columns) and is in states/eV.
        """
        sigma = self.args.sigma   # eV
        de = self.args.de         # eV
        reference_eV = (
            self.metadata["efermi_eV"]
            if self.args.reference == "fermi"
            else self.calculate_vbm()
        )

        # --- Pass 1: stream through every file just to find min/max eigenvalue.
        emin, emax = np.inf, -np.inf
        for finfo in self._files:
            for chunk in _stream_numeric_rows(finfo["filename"], self.args.chunk_size):
                eigen_eV = chunk[:, 1] * HARTREE_TO_EV
                emin = min(emin, eigen_eV.min())
                emax = max(emax, eigen_eV.max())

        emin -= 10 * sigma
        emax += 10 * sigma
        energy_grid = np.arange(emin, emax + de, de)  # eV

        n_cols = len(self.metadata["orbitals"])
        pdos_matrix = np.zeros((len(energy_grid), n_cols))

        def gaussian_eV(x_eV: np.ndarray) -> np.ndarray:
            # Normalized Gaussian, area = 1 when integrated over dE (eV).
            return np.exp(-(x_eV**2) / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))

        # --- Pass 2: accumulate the broadened PDOS, one file / chunk at a time.
        col_offset = 0
        for finfo in self._files:
            K = finfo["n_orbitals"]
            start_col = finfo["pdos_start_col"]
            file_contrib = np.zeros((len(energy_grid), K))

            for chunk in _stream_numeric_rows(finfo["filename"], self.args.chunk_size):
                eigen_eV = chunk[:, 1] * HARTREE_TO_EV
                pdos_vals = chunk[:, start_col:]

                # NOTE (the key fix): no multiplication by `de` and no stray
                # Hartree/eV factor here. Each eigenvalue's Gaussian already
                # integrates to 1 in eV; the discrete sum over eigenvalues,
                # weighted by their PDOS/population values, IS the density.
                # `de` only matters later, when *integrating* the resulting
                # curve (e.g. to check the sum rule) -- it must never rescale
                # the curve itself.
                diff = energy_grid[:, np.newaxis] - eigen_eV[np.newaxis, :]
                weights = gaussian_eV(diff)
                file_contrib += weights @ pdos_vals

            pdos_matrix[:, col_offset : col_offset + K] = file_contrib
            col_offset += K

        energy_grid = energy_grid - reference_eV
        return energy_grid, pdos_matrix

    # ------------------------------------------------------------------ #
    # Output
    # ------------------------------------------------------------------ #
    def save_data(self, energy: np.ndarray, pdos: np.ndarray) -> None:
        """Write energy + per-orbital PDOS (+ optional total) to a text file."""
        headers = ["Energy(eV)"]
        headers += [
            f"{elem}_{orb}"
            for elem, orb in zip(self.metadata["elements"], self.metadata["orbitals"])
        ]

        if self.args.total_dos:
            pdos = np.column_stack((pdos, pdos.sum(axis=1)))
            headers.append("Total_DOS")

        np.savetxt(
            self.args.output,
            np.column_stack((energy, pdos)),
            header=" ".join(f"{h:>16}" for h in headers),
            comments="",
            fmt="%16.8f",
        )
        log.info("Wrote processed PDOS data to %s", self.args.output)

    def generate_plot(self, energy: np.ndarray, pdos: np.ndarray) -> None:
        """Create a PDOS plot with matplotlib."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            log.error("Plotting requires matplotlib. Install with: pip install matplotlib")
            return

        from collections import defaultdict

        element_orbitals: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
        for i, (elem, orb) in enumerate(
            zip(self.metadata["elements"], self.metadata["orbitals"])
        ):
            element_orbitals[elem][orb].append(pdos[:, i])

        # Valid, arbitrarily-extensible color cycle (fixes the malformed
        # 7-digit hex code that crashed plotting once a 5th element appeared).
        n_elements = len(element_orbitals)
        cmap = plt.get_cmap("tab10" if n_elements <= 10 else "tab20")
        colors = [cmap(i % cmap.N) for i in range(n_elements)]

        fig, ax = plt.subplots(figsize=(10, 6))
        xlabel = f"Energy (eV) relative to {self.args.reference.upper()}"

        for color, (element, orbitals) in zip(colors, element_orbitals.items()):
            orbital_sums = {orb: np.sum(vals, axis=0) for orb, vals in orbitals.items()}
            total = sum(orbital_sums.values())

            if self.args.plot_type in ("total", "both"):
                ax.plot(energy, total, label=f"{element} Total", color=color, linewidth=1.3)

            if self.args.plot_type in ("orbital", "both"):
                linestyles = ["--", "-.", ":", (0, (1, 1))]
                for (orb, dos), ls in zip(orbital_sums.items(), itertools.cycle(linestyles)):
                    ax.plot(
                        energy, dos, linestyle=ls, color=color, linewidth=0.7,
                        label=f"{element} {orb}", alpha=0.8,
                    )

        if self.args.total_dos:
            ax.plot(energy, pdos[:, -1], "k--", label="Total DOS", linewidth=1.2)

        ax.axvline(0, color="grey", linestyle=":", alpha=0.7)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("PDOS (states/eV)")
        ax.set_title("Projected Density of States")

        if self.args.xlim is not None:
            ax.set_xlim(self.args.xlim)
        # y-limits are left to matplotlib's autoscale by default, since a
        # hardcoded [-0.3, 14.3] range (as in the original scripts) silently
        # clips or misrepresents any system whose PDOS doesn't happen to
        # fall in that range. Use --ylim to override explicitly.
        if self.args.ylim is not None:
            ax.set_ylim(self.args.ylim)

        ax.legend(bbox_to_anchor=(1.05, 0.5), loc="center left", borderaxespad=0.1)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.args.plot_output, dpi=300)
        plt.close(fig)
        log.info("Wrote plot to %s", self.args.plot_output)


# ---------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Process and visualize CP2K PDOS data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Example:\n"
            "  python -m pdos_toolkit.pdos_processor "
            "pdos_system-k1-1.pdos pdos_system-k2-1.pdos pdos_system-k3-1.pdos "
            "-s 0.05 -d 0.01 --reference fermi --total-dos --plot "
            "--plot-output dos.png --plot-type both"
        ),
    )
    parser.add_argument("pdosfilenames", nargs="+", help="CP2K PDOS output files (e.g. *-k*.pdos)")
    parser.add_argument("-o", "--output", default="pdos.dat", help="Output filename for processed data")
    parser.add_argument("-s", "--sigma", type=float, default=0.1, help="Gaussian broadening width, in eV")
    parser.add_argument("-d", "--de", type=float, default=0.01, help="Energy grid spacing, in eV")
    parser.add_argument("--reference", choices=["fermi", "vbm"], default="fermi", help="Energy reference for alignment")
    parser.add_argument("--total-dos", action="store_true", help="Include a summed Total_DOS column/curve")
    parser.add_argument("--chunk-size", type=int, default=2000, help="Rows read per chunk (controls peak memory use)")

    parser.add_argument("--plot", action="store_true", help="Generate a PDOS plot")
    parser.add_argument("--plot-output", default="pdos.png", help="Output filename for the plot")
    parser.add_argument("--plot-type", choices=["total", "orbital", "both"], default="total", help="Visualization detail level")
    parser.add_argument("--xlim", type=float, nargs=2, default=None, metavar=("XMIN", "XMAX"), help="Optional x-axis limits, in eV")
    parser.add_argument("--ylim", type=float, nargs=2, default=None, metavar=("YMIN", "YMAX"), help="Optional y-axis limits, in states/eV")
    return parser


def main(argv: List[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)

    try:
        processor = PDOSProcessor(args)
        processor.load_files()
        energy, pdos = processor.process_data()
        processor.save_data(energy, pdos)
        if args.plot:
            processor.generate_plot(energy, pdos)
    except Exception as exc:  # noqa: BLE001 - CLI top-level error boundary
        log.error(str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
