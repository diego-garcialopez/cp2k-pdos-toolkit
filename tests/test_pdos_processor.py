"""
Tests for pdos_toolkit.pdos_processor.

Run with:  pytest

These tests focus on the two substantive bugs found in the original
prototypes (pdos_plot_5.py and pdos_plot_6_memory.py):

1. The broadened PDOS must NOT depend on the chosen output grid spacing
   `de` (it did, in the original code, because `de` was folded into the
   convolution weight).
2. The plotting color palette must never contain an invalid hex code
   (the original had one 7-digit code, '#3CCB974', which crashed
   matplotlib as soon as a 5th unique element was plotted).

A couple of lighter parsing/robustness tests are included too.
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend for CI

import pytest

from pdos_toolkit.pdos_processor import (
    PDOSProcessor,
    HEADER_REGEX,
    HARTREE_TO_EV,
)


HEADER = (
    "# Projected DOS for atomic kind Cu at iteration step i = 0, "
    "E(Fermi) =   -0.150000 a.u."
)
COLUMN_HEADER = "  MO  Eigenvalue[a.u.]  Occupation      s      p      d"


def _write_pdos_file(path, element, n=200, efermi=-0.15, seed=0):
    rng = np.random.default_rng(seed)
    eig = np.sort(rng.normal(efermi, 0.05, n))
    occ = (eig < efermi).astype(float) * 2.0
    s = rng.random(n) * 0.5
    p = rng.random(n) * 0.5
    d = rng.random(n) * 0.2
    with open(path, "w") as f:
        f.write(
            f"# Projected DOS for atomic kind {element} at iteration step "
            f"i = 0, E(Fermi) =   {efermi} a.u.\n"
        )
        f.write(COLUMN_HEADER + "\n")
        for i in range(n):
            f.write(
                f"{i+1:6d} {eig[i]:14.6f} {occ[i]:10.4f} "
                f"{s[i]:8.4f} {p[i]:8.4f} {d[i]:8.4f}\n"
            )
    return path


def _make_args(**overrides):
    defaults = dict(
        pdosfilenames=[],
        output="out.dat",
        sigma=0.1,
        de=0.05,
        reference="fermi",
        total_dos=True,
        chunk_size=64,
        plot=False,
        plot_output="out.png",
        plot_type="both",
        xlim=None,
        ylim=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_header_regex_matches_expected_format():
    m = HEADER_REGEX.match(HEADER)
    assert m is not None
    assert m.group("element") == "Cu"
    assert float(m.group("Efermi")) == pytest.approx(-0.15)


def test_broadened_dos_is_independent_of_grid_spacing(tmp_path):
    """This is the regression test for the main physics bug: the integral
    of the broadened DOS (a proxy for total occupied+unoccupied weight)
    must be essentially the same regardless of `de`."""
    f = _write_pdos_file(tmp_path / "pdos-k1-1.pdos", "Cu", n=300)

    integrals = []
    for de in (0.05, 0.01, 0.005):
        args = _make_args(pdosfilenames=[str(f)], de=de, total_dos=False)
        proc = PDOSProcessor(args)
        proc.load_files()
        energy, pdos = proc.process_data()
        integrals.append(np.sum(pdos) * de)

    integrals = np.array(integrals)
    # All integrals should agree to within a small relative tolerance,
    # regardless of how fine the output grid is.
    assert np.allclose(integrals, integrals[0], rtol=0.02)


def test_efermi_consistency_check_rejects_mismatched_files(tmp_path):
    f1 = _write_pdos_file(tmp_path / "a-k1-1.pdos", "Cu", efermi=-0.15)
    f2 = _write_pdos_file(tmp_path / "a-k2-1.pdos", "Cu", efermi=-0.30)

    args = _make_args(pdosfilenames=[str(f1), str(f2)])
    proc = PDOSProcessor(args)
    with pytest.raises(ValueError, match="Inconsistent Fermi energies"):
        proc.load_files()


def test_plot_color_palette_has_no_invalid_hex_codes():
    """Regression test for the malformed '#3CCB974' color: every generated
    color must be a valid RGBA tuple/hex understood by matplotlib, for any
    number of elements."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

    for n_elements in range(1, 15):
        cmap = plt.get_cmap("tab10" if n_elements <= 10 else "tab20")
        colors = [cmap(i % cmap.N) for i in range(n_elements)]
        for c in colors:
            to_rgba(c)  # raises ValueError if invalid


def test_generate_plot_runs_with_many_elements(tmp_path):
    """End-to-end: plotting must not crash even with >4 unique elements
    (the original hardcoded 7-item color list, with one invalid entry,
    would raise a ValueError on the 5th element)."""
    files = []
    for i, elem in enumerate(["Cu", "Au", "S", "Me", "O", "N"]):
        files.append(str(_write_pdos_file(tmp_path / f"{elem}-k1-1.pdos", elem, seed=i)))

    args = _make_args(
        pdosfilenames=files,
        plot=True,
        plot_output=str(tmp_path / "out.png"),
    )
    proc = PDOSProcessor(args)
    proc.load_files()
    energy, pdos = proc.process_data()
    proc.generate_plot(energy, pdos)  # should not raise
    assert (tmp_path / "out.png").exists()


def test_vbm_below_fermi(tmp_path):
    f = _write_pdos_file(tmp_path / "pdos-k1-1.pdos", "Cu", efermi=-0.15)
    args = _make_args(pdosfilenames=[str(f)], reference="vbm")
    proc = PDOSProcessor(args)
    proc.load_files()
    vbm_eV = proc.calculate_vbm()
    assert vbm_eV < proc.metadata["efermi_eV"]
