# cp2k-pdos-toolkit

A small, tested Python tool to post-process and plot **Projected Density of
States (PDOS)** output from [CP2K](https://www.cp2k.org/): it applies
Gaussian broadening to CP2K's per-MO eigenvalue/orbital-weight data,
optionally aligns the energy axis to the Fermi level or the Valence Band
Maximum, and writes both a text table and a plot.

This repository consolidates and fixes two earlier prototype scripts. See
[`CHANGELOG.md`](CHANGELOG.md) for the itemized list of what changed and
[`docs/theory.md`](docs/theory.md) for the physics/math behind the main fix.

## Quick start

```bash
git clone https://github.com/<your-username>/cp2k-pdos-toolkit.git
cd cp2k-pdos-toolkit
pip install -e .

python -m pdos_toolkit.pdos_processor \
    pdos_system-k1-1.pdos pdos_system-k2-1.pdos \
    -s 0.05 -d 0.01 --reference fermi --total-dos \
    --plot --plot-output pdos.png -o pdos.dat
```

See [`examples/example_usage.md`](examples/example_usage.md) for more
scenarios (multi-element systems, VBM referencing, library usage).

## Repository layout

```
cp2k-pdos-toolkit/
├── README.md                     <- you are here
├── LICENSE                       <- MIT
├── CHANGELOG.md                  <- what changed vs. the original prototypes, and why
├── CONTRIBUTING.md               <- how to set up a dev environment and submit changes
├── pyproject.toml                <- packaging + pytest config
├── requirements.txt              <- runtime dependencies (numpy, matplotlib)
├── .gitignore
├── .github/
│   └── workflows/tests.yml       <- CI: runs the test suite on every push/PR
├── pdos_toolkit/                 <- the actual importable package
│   ├── __init__.py
│   └── pdos_processor.py
├── tests/
│   └── test_pdos_processor.py    <- pytest suite, including regression tests for the two bugs below
├── docs/
│   └── theory.md                 <- Gaussian broadening math + full derivation of the unit bug
└── examples/
    └── example_usage.md
```

This is a fairly standard "small scientific Python package" layout: code
lives in an importable package directory (`pdos_toolkit/`) rather than as a
loose top-level script, so it can be both run as a CLI (`python -m
pdos_toolkit.pdos_processor ...`) and imported as a library
(`from pdos_toolkit import PDOSProcessor`). Tests, docs, and examples each
get their own directory so the project scales cleanly if it grows.

---

## Which of the two original scripts was better, and what was wrong with both?

You supplied two versions, `pdos_plot_5.py` and `pdos_plot_6_memory.py`,
the latter billed as a memory-usage optimization. Here is the honest
comparison.

### Neither is "better" in the way its name implies

`pdos_plot_6_memory.py`'s central claim — that it processes data "in
chunks to avoid large matrix allocations" — **does not actually hold up**:

- Its `calculate_vbm()` calls `np.loadtxt(filename, skiprows=2)`, loading
  the *entire* file into memory, just to compute a VBM.
- Its `process_data()` then calls `np.loadtxt(filename, skiprows=2)`
  **again**, on the same file, just to find `emin`/`emax` for the energy
  grid.
- It then calls `np.loadtxt(filename, skiprows=2)` a **third** time in the
  main accumulation loop — and only *after* that full array is sitting in
  RAM does it slice it into `chunk_size`-row pieces for the convolution.

So relative to `pdos_plot_5.py` (which reads each file exactly once), the
"memory-optimized" version reads every input file from disk two to three
times, while its "chunking" happens entirely on data that's already fully
loaded — meaning **peak memory usage is not reduced at all**, and runtime
is worse due to redundant I/O. If your files are large enough that memory
is a real concern, this version would be strictly worse than the first.

Both scripts otherwise share essentially identical logic (and identical
bugs — see below), copy-pasted between `save_data()`/`generate_plot()`
verbatim. `pdos_plot_6_memory.py`'s one genuine improvement is that it
detects the `Occupation` column index dynamically from the file header
(`pdos_plot_5.py` hardcodes it as column `2`), which is more robust if
CP2K's column order ever changes.

**Bottom line:** rather than pick a "winner," this repository merges the
two into one implementation that keeps `pdos_plot_6_memory.py`'s dynamic
column detection, fixes the shared bugs below, and adds a *genuine*
low-memory streaming mode (`--chunk-size`) that never loads a full file
into memory at once — something neither original script actually achieved.

### Bug 1 (serious): PDOS magnitude silently depends on plot resolution

Both scripts compute the broadening weight as:

```python
gaussian(energy_diff) * (self.args.de * 27.211384)
```

This folds the output-grid spacing `de` and the Hartree→eV conversion
factor into the per-eigenvalue Gaussian weight — which is wrong on both
counts (`de` shouldn't appear there at all, and the unit factor is applied
in the wrong direction). The practical effect: **the reported PDOS peak
heights scale with an arbitrary plotting-resolution choice.** We verified
this numerically — with a single unit-weight eigenvalue and the original
formula, the integral of the resulting curve (which should equal `1.0`,
the state's weight) came out as `2.721`, `0.272`, and `0.027` for
`de = 0.1, 0.01, 0.001 eV` respectively. That's not noise — it's a
100x swing driven entirely by a cosmetic parameter. The code's own comment
("a too-fine grid can lead to underestimation") is describing this bug's
symptom without identifying its cause.

The fix (implemented in `pdos_toolkit/pdos_processor.py`) does all
broadening math directly in eV, with no `de` term or stray conversion
factor inside the convolution weight. See [`docs/theory.md`](docs/theory.md)
for the full derivation, and
`tests/test_pdos_processor.py::test_broadened_dos_is_independent_of_grid_spacing`
for the regression test.

### Bug 2 (crashes on real data): invalid hex color

```python
colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B3', '#3CCB974', '#64B5CD', '#4C72B0']
```

`'#3CCB974'` has 7 hex digits (matplotlib requires 6 or 8). This is
silently fine right up until a **5th unique element** appears in the plot,
at which point `matplotlib.colors.to_rgba` raises `ValueError: Invalid
RGBA argument`. Confirmed directly:

```
>>> matplotlib.colors.to_rgba('#3CCB974')
ValueError: Invalid RGBA argument: '#3CCB974'
```

For a project studying a Cu/Au transmetallation with an SMe₂ ligand (Cu,
Au, S, C, N, O, H, ...), this is not a corner case — it's very plausibly
going to be hit. Fixed by generating colors from a `tab10`/`tab20`
matplotlib colormap, which scales to any number of elements without
hand-maintained hex strings.

### Smaller issues fixed along the way

| Issue | Where | Fix |
|---|---|---|
| Fermi-energy consistency check used `set()` (exact float equality) on independently-parsed text values | both | `np.allclose(..., atol=1e-8)` |
| Hardcoded `plt.ylim([-0.3, 14.3])` silently clips/misrepresents any system whose PDOS falls outside that range | both | autoscale by default; `--ylim`/`--xlim` to override |
| `Occupation`/eigenvalue column index hardcoded (`d[:, 2]`) rather than derived from the header | `pdos_plot_5.py` | dynamic column detection everywhere |
| Unused `import matplotlib.ticker as ticker` | both | removed |
| `sigma`/`de` converted Hartree↔eV in `main()` in one script but inside the class in the other — inconsistent and easy to get wrong if the two files are ever merged carelessly | both | single convention: CLI takes eV, all internal computation in eV |
| No automated tests | both | `tests/test_pdos_processor.py`, run in CI |

### Known limitations / possible future work

- The tool assumes all input files belong to the *same* calculation (same
  Fermi energy) — it does not attempt to align PDOS from separate,
  unrelated calculations.
- It does not currently validate that the set of orbital types is
  consistent across files for the same element (e.g. if one k-point's file
  were malformed and had a different orbital-column layout, results would
  silently be misaligned). Worth adding if you routinely deal with
  heterogeneous CP2K output.
- Broadening is Gaussian only; a Lorentzian or Voigt option would be a
  natural extension for materials with more lifetime broadening.

---

## Creating a GitHub repository for this code, step by step

This section assumes you're starting from zero — you have this folder
locally and want it on GitHub, properly documented, with a clean history.

### 1. Decide on repository contents and structure (already done above)

Before touching `git`, it's worth deciding what belongs in version control
and what doesn't:

- **Do commit:** source code, tests, docs, packaging files, a `.gitignore`,
  a license, a README.
- **Don't commit:** raw CP2K output (`*.pdos` files are often large and are
  regeneratable simulation output, not source code — see the discussion of
  Git LFS below if you *do* want to version large example data), generated
  plots/`.dat` files, virtual environments, `__pycache__/`.

The `.gitignore` in this repository already encodes those choices.

### 2. Initialize the local Git repository

From inside the `cp2k-pdos-toolkit/` folder:

```bash
git init
git add .
git status                     # sanity check: nothing from .gitignore should appear
git commit -m "Initial commit: consolidated, fixed PDOS post-processing toolkit"
```

`git status` before committing is worth the habit — it's the easiest place
to catch "oops, I'm about to commit a 200 MB `.pdos` file" mistakes.

### 3. Choose a license (already included: MIT)

If you want something different (e.g. GPLv3, Apache-2.0, or a "no license,
all rights reserved" stance for now), GitHub's own picker
(https://choosealicense.com/) is a good reference. MIT was chosen here as a
permissive default suitable for a small scientific utility; change
`LICENSE` and the license line in `pyproject.toml` if you'd prefer
something else, e.g. `GPL-3.0` to match common conventions in some
computational chemistry software ecosystems (CP2K itself is GPL-licensed).

### 4. Create the remote repository on GitHub

**Option A — GitHub CLI (`gh`), recommended if you have it installed:**

```bash
gh repo create cp2k-pdos-toolkit --public --source=. --remote=origin --push
```

This single command creates the remote repo, sets it as `origin`, and
pushes your existing commit(s) in one step. Use `--private` instead of
`--public` if you'd rather keep it internal for now (e.g. while it's tied
to unpublished research) and open it up later — flipping a repo from
private to public is a one-click setting change whenever you're ready.

**Option B — GitHub website:**

1. Go to https://github.com/new
2. Repository name: `cp2k-pdos-toolkit`
3. Add a short description, e.g. "Gaussian-broadened PDOS post-processing
   and plotting for CP2K"
4. Choose Public or Private
5. **Do not** initialize with a README, `.gitignore`, or license on
   GitHub's side — you already have all three locally, and letting GitHub
   create its own would immediately conflict with your local commit.
6. Click "Create repository." GitHub will show you the remote URL,
   something like `https://github.com/<your-username>/cp2k-pdos-toolkit.git`
   (HTTPS) or `git@github.com:<your-username>/cp2k-pdos-toolkit.git` (SSH).

Then, back in your terminal:

```bash
git remote add origin git@github.com:<your-username>/cp2k-pdos-toolkit.git
git branch -M main
git push -u origin main
```

(`-u` sets `origin/main` as the upstream for your local `main`, so future
pushes/pulls can just be `git push` / `git pull`.)

### 5. Authentication note (SSH vs. HTTPS)

- **SSH** (`git@github.com:...`) requires an SSH key added to your GitHub
  account (Settings → SSH and GPG keys). Once set up, no password prompts.
- **HTTPS** (`https://github.com/...`) will prompt for a username and a
  **Personal Access Token** (not your account password — GitHub disabled
  password auth for git operations years ago). Create one under
  Settings → Developer settings → Personal access tokens.

If you're not sure which you have configured, `git remote -v` shows which
URL scheme your `origin` is using, and you can switch with
`git remote set-url origin <new-url>`.

### 6. Verify the CI workflow runs

This repo includes `.github/workflows/tests.yml`, which runs `pytest`
automatically on every push and pull request, across Python 3.9/3.11/3.12.
After your first push:

1. Go to your repository page on GitHub → **Actions** tab.
2. You should see a "Tests" workflow run triggered by your push. Click into
   it to see per-Python-version results.
3. If it's green, add the status badge to the top of your README (GitHub
   shows you the exact Markdown snippet under Actions → the workflow run →
   "..." menu → "Create status badge").

### 7. Set up branch protection (optional but recommended once collaborators are involved)

Settings → Branches → Add branch protection rule for `main`:
- Require pull request reviews before merging.
- Require status checks to pass before merging (select the "Tests" CI
  workflow once it has run at least once).

This prevents anyone (including future-you, in a hurry) from pushing
straight to `main` without the test suite passing.

### 8. Tag a release

Once you're happy with the state of the code:

```bash
git tag -a v1.0.0 -m "v1.0.0: consolidated and fixed PDOS toolkit"
git push origin v1.0.0
```

Then on GitHub: Releases → Draft a new release → choose the `v1.0.0` tag →
paste in the relevant section of `CHANGELOG.md` as release notes → Publish.
This gives collaborators (and your future self) a stable, citable
reference point — useful if this code ends up referenced in a
methods/SI section of a paper.

### 9. (Optional) Publishing example data with Git LFS

If you want to commit real (not synthetic) example `.pdos` files for
`examples/`, and they're more than a few MB, plain Git handles large
binary-ish files poorly (every version is stored in full, bloating clone
size forever). Git LFS (Large File Storage) is the standard fix:

```bash
git lfs install
git lfs track "examples/*.pdos"
git add .gitattributes
git commit -m "Track example PDOS files with Git LFS"
```

Note this requires collaborators to have `git-lfs` installed too, and
GitHub's free tier has LFS storage/bandwidth limits — for a handful of
example files, it's often simpler to just keep examples small/synthetic
(as this repo's tests already do) and skip LFS entirely.

### 10. Everyday workflow going forward

```bash
git checkout -b feature/my-change     # new branch per change
# ... edit files ...
pytest -v                             # make sure tests pass locally
git add -A
git commit -m "Concise description of the change"
git push -u origin feature/my-change
gh pr create --fill                   # or open a PR via the GitHub website
```

Once CI is green and (if applicable) a review is approved, merge via the
GitHub website ("Squash and merge" keeps `main`'s history to one commit
per feature, which is usually the most readable option for a small
project like this).

## Citing / referencing

If this tool ends up supporting a publication's methods section, tagging a
release (step 8 above) and linking to that specific tagged version (e.g.
`https://github.com/<your-username>/cp2k-pdos-toolkit/releases/tag/v1.0.0`)
is more reproducible than linking to `main`, since `main` can change after
the paper is published.
