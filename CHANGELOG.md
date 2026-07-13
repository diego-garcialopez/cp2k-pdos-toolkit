# Changelog

## [1.0.0] - Initial public release

This release consolidates two earlier internal prototype scripts,
into a single, tested package.
See `docs/theory.md` for the full derivation behind fix #1 below.

### Fixed

- **Broadening magnitude bug (both prototypes).** The Gaussian broadening
  weight was multiplied by the output grid spacing `de` and by the
  Hartree→eV factor in the wrong direction, making the reported PDOS
  magnitude depend on an arbitrary plotting-resolution choice instead of
  being a physical quantity. Fixed by doing all broadening math directly
  in eV with no `de` term in the per-eigenvalue weight.
- **Invalid hex color crash (both prototypes).** The hardcoded plot color
  list contained `'#3CCB974'`, a 7-hex-digit string that matplotlib
  rejects. Plotting any system with a 5th unique element raised a
  `ValueError`. Replaced with a `tab10`/`tab20` colormap that scales to
  any number of elements.
- **Fragile Fermi-energy consistency check.** Comparing parsed floats with
  `set()` (exact equality) is brittle. Replaced with `np.allclose`.
- **Hardcoded, non-general y-axis limits** silently
  clipped/misrepresented PDOS for any system whose magnitude fell outside
  that range. Plots now autoscale by default, with `--xlim`/`--ylim` to
  override explicitly.
- **Fake "memory optimization".** A previous script "chunked"
  processing called `np.loadtxt()` on an entire file first and only
  sliced chunks out of the resulting in-memory array afterward — this
  does not reduce peak memory at all, and additionally re-read every input
  file from disk two to three times (once for `calculate_vbm`, once to
  collect `all_eigenvalues`, once for the main accumulation loop). Replaced
  with genuine chunked streaming (`_stream_numeric_rows`) that never holds
  more than `--chunk-size` rows of a file in memory and reads each file
  exactly twice (once to find the energy range, once to accumulate).

### Changed

- Column indices for `Occupation` are now consistently detected from the
  file header everywhere (the original `pdos_plot_5.py` hardcoded
  `Occupation` as column index 2, which happened to work but was brittle
  if CP2K's column order ever changed).
- CLI `--sigma`/`--de` are now taken in eV and used in eV throughout,
  removing the Hartree round-trip that made the two prototypes' unit
  handling hard to audit.
- Unused `matplotlib.ticker` import removed.

### Added

- Automated test suite (`tests/test_pdos_processor.py`), including a
  regression test that would have caught the broadening-magnitude bug.
- `docs/theory.md` explaining the underlying math and the bug in detail.
- `--chunk-size`, `--xlim`, `--ylim` CLI options.
- Packaging (`pyproject.toml`) with a `pdos-process` console-script entry
  point.
- GitHub Actions CI running the test suite on each push/PR.
