# Theory: from CP2K eigenvalues to a broadened PDOS curve

## 1. What CP2K gives you

For each atomic kind, CP2K's `PDOS` output writes one row per molecular
orbital (MO): its index, its eigenvalue (in Hartree), its occupation number,
and its projected weight onto each atomic orbital type (s, p, d, ...) of
that atomic kind (a Mulliken-like population). Each k-point (or each replica
in a multi-walker/multi-image run) is written to its own file, and all such
files for one calculation share the same Fermi energy.

This is fundamentally a **list of delta functions**: each MO contributes a
spike of height equal to its orbital weight, sitting exactly at its
eigenvalue. That's not plottable as-is — you broaden each spike into a
Gaussian (or Lorentzian) of width `sigma`, and sum them:

```
DOS(E) = Σ_i  w_i · G(E - E_i, σ)
```

where `w_i` is the orbital weight of MO *i*, `E_i` its eigenvalue, and `G`
a Gaussian normalized so that `∫ G(x, σ) dx = 1`.

Two properties fall directly out of this definition, and both are useful as
sanity checks on any implementation:

- **`DOS(E)` is a density.** Its units are *states per unit energy*
  (e.g. states/eV). It is **not** supposed to depend on how finely you
  choose to *sample* `E` for plotting — only on `σ` and on the underlying
  eigenvalues/weights.
- **The sum rule.** `∫ DOS(E) dE = Σ_i w_i`, the total weight summed over
  all MOs. Numerically, `Σ_grid DOS(E_grid) · Δe ≈ Σ_i w_i`, and this
  approximation should converge to the same value as `Δe → 0`, not drift.

## 2. The bug in the original scripts

Both `pdos_plot_5.py` and `pdos_plot_6_memory.py` computed the broadening
weight as:

```python
gaussian(energy_diff) * (self.args.de * 27.211384)
```

i.e. they multiplied the Gaussian by the *output grid spacing* `de` and by
the Hartree→eV conversion factor `27.211384`. This is wrong on two counts:

1. **`de` should never appear in the convolution weight at all.** `de` is
   a plotting/output-grid choice, not a physical quantity. Folding it into
   the weight makes `DOS(E)` scale linearly with `de`: halve the grid
   spacing (i.e. plot on a finer grid, which should only improve
   resolution) and every peak height is silently cut in half. This was
   confirmed numerically (see the repository's test suite and the analysis
   log): with a single unit-weight eigenvalue, the integral of the
   resulting curve came out to `2.721`, `0.272`, and `0.027` for
   `de = 0.1, 0.01, 0.001 eV` respectively — i.e. it scaled linearly with
   `de` instead of staying at the physically-correct value of `1.0`.
2. **The Hartree→eV factor was applied in the wrong direction.** Converting
   a normalized density from "per Hartree" to "per eV" requires *dividing*
   by 27.211386 (a Gaussian normalized in Hartree units is `27.211386`
   times taller, per unit eV, than the same distribution normalized
   directly in eV) — the code *multiplied* by it instead.

The practical consequence: the absolute PDOS magnitude reported and
plotted by both scripts was off by a factor that depended on an arbitrary
numerical choice (`de`), not on physics. The original code even documents
the symptom without diagnosing the cause, in a comment: *"A too-fine grid
(small de) can lead to underestimation"* — that sentence is describing the
bug, not a real physical effect. On a genuinely correct implementation, DOS
magnitude should be practically insensitive to `de` (with a tiny, expected
discretization error that shrinks — not grows — as `de` gets smaller).

## 3. The fix used in this repository

`pdos_toolkit/pdos_processor.py` converts every eigenvalue to eV
immediately upon reading, and does the entire broadening calculation in eV
throughout, with no back-and-forth unit conversion inside the hot loop:

```python
def gaussian_eV(x_eV):
    return np.exp(-(x_eV**2) / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))

diff = energy_grid[:, None] - eigen_eV[None, :]
weights = gaussian_eV(diff)          # <-- no `de`, no stray unit factor
contribution = weights @ pdos_vals
```

`de` is used only for two things, both legitimate: (a) choosing the
spacing of the *output* energy grid, and (b) as the trapezoid/Riemann
weight *if and when you want to integrate the final curve*
(`np.sum(dos) * de`) — never inside the per-eigenvalue weight itself.

This was verified directly: with the fix, the same single-eigenvalue test
described above gives `1.000000` at `de = 0.1, 0.01, 0.001 eV` alike (see
`tests/test_pdos_processor.py::test_broadened_dos_is_independent_of_grid_spacing`).

## 4. Choosing `sigma` and `de` in practice

- `sigma` (broadening width) should be comparable to or a bit larger than
  the typical spacing between neighboring eigenvalues in the region of
  interest — too small and the curve looks like a comb of narrow spikes;
  too large and you wash out real features (e.g. splitting a d-manifold).
  `0.05–0.2 eV` is a reasonable starting range for solid-state/molecular
  systems; tune per system.
- `de` (grid spacing) should be several times smaller than `sigma`
  (`de ≈ sigma / 10` is a safe rule of thumb) purely for plotting
  smoothness — with the fix above, making `de` smaller no longer changes
  the DOS magnitude, only the plotting resolution and runtime.
