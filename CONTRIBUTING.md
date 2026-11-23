# Contributing

Thanks for considering a contribution. This is a small, focused scientific
tool, so the bar is mostly "does it stay correct and readable."

## Setting up a development environment

```bash
git clone https://github.com/diego-garcialopez/cp2k-pdos-toolkit.git
cd cp2k-pdos-toolkit
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Running the tests

```bash
pytest -v
```

Please add or update a test in `tests/test_pdos_processor.py` for any
bug fix or new feature — especially anything touching the broadening math
in `process_data()`, since that code has a history of silent unit bugs
(see `docs/theory.md`).

## Style

- Keep functions small and docstringed (Google-style docstrings, as used
  throughout `pdos_processor.py`).
- Prefer explicit unit suffixes in variable names (`_eV`, `_ha`) whenever a
  quantity's unit isn't otherwise obvious from context — this codebase has
  already been bitten once by an implicit Hartree/eV mismatch.
- Run `python -m pyflakes pdos_toolkit/` (or your linter of choice) before
  opening a PR to catch unused imports, etc.

## Submitting changes

1. Fork the repository and create a feature branch:
   `git checkout -b fix/short-description`
2. Make your change, with tests.
3. Make sure `pytest` passes.
4. Open a pull request describing *what* changed and *why* — for numerical
   code, a before/after sanity check (like the sum-rule check in
   `docs/theory.md`) is extremely helpful for reviewers.

## Reporting bugs

Please open an issue with:
- CP2K version and a minimal PDOS file (or a snippet of one) that
  reproduces the problem, if possible.
- The exact command you ran.
- What you expected vs. what you got.
