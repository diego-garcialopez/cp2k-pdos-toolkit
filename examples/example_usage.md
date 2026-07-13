# Example usage

## Basic run: one element, multiple k-points

Suppose CP2K produced, for a copper atomic kind, one PDOS file per k-point:

```
pdos_system-k1-1.pdos
pdos_system-k2-1.pdos
pdos_system-k3-1.pdos
pdos_system-k4-1.pdos
```

Process and plot them, referenced to the Fermi level, with a 0.05 eV
Gaussian broadening and a 0.01 eV output grid:

```bash
python -m pdos_toolkit.pdos_processor \
    pdos_system-k1-1.pdos pdos_system-k2-1.pdos pdos_system-k3-1.pdos pdos_system-k4-1.pdos \
    -s 0.05 -d 0.01 \
    --reference fermi \
    --total-dos \
    --plot --plot-output cu_pdos.png --plot-type both \
    -o cu_pdos.dat
```

This writes:
- `cu_pdos.dat` — a text table: `Energy(eV)`, one column per
  element/orbital, plus `Total_DOS` (states/eV).
- `cu_pdos.png` — a plot with both the per-element totals and the
  individual s/p/d contributions.

## Comparing two elements (e.g. a transmetallation Cu/Au system)

If you have PDOS files for both a copper and a gold atomic kind from the
*same* calculation (same Fermi energy), just pass all files together — the
tool groups columns by the element name parsed from each file's header:

```bash
python -m pdos_toolkit.pdos_processor \
    pdos_system-Cu-k*.pdos pdos_system-Au-k*.pdos \
    -s 0.05 -d 0.01 \
    --reference vbm \
    --plot --plot-type total \
    --xlim -6 6 --ylim 0 20
```

`--reference vbm` aligns the energy axis to the Valence Band Maximum
instead of the Fermi level, which is often more convenient when comparing
across sligtly different Fermi-level conventions.

## Using it as a library instead of a CLI

```python
import argparse
from pdos_toolkit import PDOSProcessor

args = argparse.Namespace(
    pdosfilenames=["pdos_system-k1-1.pdos", "pdos_system-k2-1.pdos"],
    output="out.dat",
    sigma=0.05,       # eV
    de=0.01,          # eV
    reference="fermi",
    total_dos=True,
    chunk_size=2000,
    plot=False,
    plot_output="out.png",
    plot_type="both",
    xlim=None,
    ylim=None,
)

proc = PDOSProcessor(args)
proc.load_files()
energy, pdos = proc.process_data()   # numpy arrays, ready for your own analysis
proc.save_data(energy, pdos)
```

## Memory footprint on large systems

For very large calculations (many MOs / many k-point files), tune
`--chunk-size` (rows read into memory at once per file, default `2000`).
Lowering it reduces peak memory at a small cost in speed; the default is a
reasonable balance for typical CP2K PDOS files.
