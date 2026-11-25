"""Command-line interface: ``cp2k-pdos <command> ...``.

Commands
--------
``pdos``         broaden raw PDOS/LDOS files, write a table, plot (default)
``info``         band edges, gap, populations, band-edge character
``moments``      band centre / width / filling (d-band centre and friends)
``bands``        band structure: gap, effective masses, plot, CSV
``dos``          convert / plot a CP2K total-DOS (.dos) file
``compare``      overlay the same curve from several calculations
``input``        print the CP2K input section that produces these files

``cp2k-pdos FILES... [options]`` without a command name runs ``pdos``, and
the old entry point ``pdos-process`` does the same, so command lines written
for version 1 keep working.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import analysis
from .bands import cell_from_parameters, read_bandstructure
from .broadening import LINESHAPES
from .dos import TOTAL, DOSCurves, PDOSSet, curves_from_cp2k, stream_broaden
from .io import expand_paths, is_curve_file, read_dos

log = logging.getLogger("pdos_toolkit")

COMMANDS = ("pdos", "info", "moments", "bands", "dos", "compare", "input")


# --------------------------------------------------------------------------- #
# Small argument parsers
# --------------------------------------------------------------------------- #
def _reference(text: str):
    from .dos import _normalise_reference

    try:
        return _normalise_reference(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def _steps(text: str) -> slice:
    """``"10:200:5"`` -> slice(10, 200, 5); ``"-50:"`` -> last 50 steps."""
    parts = text.split(":")
    if not 1 <= len(parts) <= 3:
        raise argparse.ArgumentTypeError("steps must look like START:STOP:STRIDE")
    try:
        nums = [int(p) if p.strip() else None for p in parts]
    except ValueError:
        raise argparse.ArgumentTypeError("steps must be integers, e.g. 100::10") from None
    if len(nums) == 1:
        return slice(nums[0], None if nums[0] is None else nums[0] + 1)
    return slice(*nums)


def _mapping(items: Optional[Sequence[str]], what: str, cast=str) -> Dict[str, object]:
    """``["Ni=2", "O=12"]`` or ``["Ni=2,O=12"]`` -> dict."""
    out: Dict[str, object] = OrderedDict()
    for item in items or []:
        for part in item.split(";") if cast is not int else item.split(","):
            if not part.strip():
                continue
            if "=" not in part:
                raise SystemExit(f"error: {what} must be given as NAME=VALUE, got {part!r}")
            key, val = part.split("=", 1)
            try:
                out[key.strip()] = cast(val.strip())
            except ValueError:
                raise SystemExit(f"error: bad value in {what}: {part!r}") from None
    return out


def _groups(items: Optional[Sequence[str]]) -> Optional[Dict[str, List[str]]]:
    raw = _mapping(items, "--group")
    if not raw:
        return None
    return {k: [m.strip() for m in str(v).split(",") if m.strip()] for k, v in raw.items()}


def _cell(values: Optional[Sequence[float]]) -> Optional[np.ndarray]:
    if values is None:
        return None
    if len(values) == 1:
        return np.eye(3) * values[0]
    if len(values) == 3:
        return cell_from_parameters(*values)
    if len(values) == 6:
        return cell_from_parameters(*values)
    if len(values) == 9:
        return np.asarray(values, dtype=float).reshape(3, 3)
    raise SystemExit(
        "error: --cell takes 1 (cubic a), 3 (a b c), 6 (a b c alpha beta gamma) "
        "or 9 (three lattice vectors) numbers"
    )


# --------------------------------------------------------------------------- #
# Shared option groups
# --------------------------------------------------------------------------- #
def _add_broadening(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("broadening")
    g.add_argument("-s", "--sigma", type=float, default=None,
                   help="Gaussian standard deviation in eV (default 0.1)")
    g.add_argument("--fwhm", type=float, default=None,
                   help="width as full width at half maximum in eV, instead of --sigma "
                        "(this is the convention of CP2K's own &BROADEN/WIDTH)")
    g.add_argument("--lineshape", choices=LINESHAPES, default="gaussian",
                   help="line shape (default: gaussian)")
    g.add_argument("--eta", type=float, default=0.5,
                   help="Lorentzian fraction of the pseudo-Voigt (default 0.5)")
    g.add_argument("--gamma", type=float, default=None,
                   help="Lorentzian half width in eV, for --lineshape voigt")
    g.add_argument("-d", "--de", type=float, default=0.01,
                   help="energy grid spacing in eV (default 0.01)")
    g.add_argument("--erange", type=float, nargs=2, default=None, metavar=("EMIN", "EMAX"),
                   help="limits of the output grid, relative to the reference "
                        "(default: all states)")
    g.add_argument("--method", choices=("auto", "exact", "fft"), default="auto",
                   help="broadening algorithm; 'fft' is much faster for long trajectories")


def _add_selection(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("what to include and how to group it")
    g.add_argument("--reference", type=_reference, default="fermi", metavar="REF",
                   help="zero of the energy axis: fermi, vbm (=homo), cbm (=lumo), midgap, "
                        "auto (CP2K's rule: fermi if occupations are smeared, else vbm), "
                        "absolute, or a number in eV (default: fermi)")
    g.add_argument("--kinds", nargs="+", default=None, metavar="NAME",
                   help="only these kinds / lists / elements")
    g.add_argument("--group-by", choices=("kind", "element", "file"), default="kind",
                   help="merge projections per CP2K kind (default), per element "
                        "(Ni1+Ni2 -> Ni) or not at all")
    g.add_argument("--group", action="append", default=None, metavar="NAME=A,B,..",
                   help="define a named group of kinds/lists/elements, e.g. "
                        "--group framework=Al,O,C,H --group CO2=list1 (repeatable)")
    g.add_argument("--orbitals", choices=("l", "m", "sum"), default="l",
                   help="resolve by angular momentum (l, default), keep m components "
                        "(m; needs COMPONENTS in CP2K) or sum over orbitals (sum)")
    g.add_argument("--weighting", choices=("all", "occupied", "unoccupied"), default="all",
                   help="plot all states, or only the occupied / unoccupied part")
    g.add_argument("--per-atom", action="store_true",
                   help="divide each group by its number of atoms (see --natoms)")
    g.add_argument("--natoms", action="append", default=None, metavar="KIND=N,..",
                   help="atoms per kind or element for --per-atom, e.g. --natoms Ni=4,O=8")
    g.add_argument("--scale", type=float, default=1.0,
                   help="multiply all curves by this factor (2 = count both spins of a "
                        "spin-restricted calculation)")
    g.add_argument("--steps", type=_steps, default=None, metavar="START:STOP:STRIDE",
                   help="MD steps to average over, as a Python-style slice over the "
                        "steps stored in each file (e.g. 100::10)")
    g.add_argument("--last-step", action="store_true",
                   help="use only the last step stored in each file")
    g.add_argument("--stream", action="store_true",
                   help="average a long trajectory one step at a time (low memory)")


def _add_plot(p: argparse.ArgumentParser, default_output: str) -> None:
    g = p.add_argument_group("plot")
    g.add_argument("--plot", action="store_true", help="make a figure")
    g.add_argument("--plot-output", default=None, metavar="FILE",
                   help=f"figure file; the extension picks the format: png, pdf, svg "
                        f"(default: {default_output}); implies --plot")
    g.add_argument("--xlim", type=float, nargs=2, default=None, metavar=("XMIN", "XMAX"),
                   help="energy range shown, in eV")
    g.add_argument("--ylim", type=float, nargs=2, default=None, metavar=("YMIN", "YMAX"),
                   help="DOS range shown (default: fitted to what is visible)")
    g.add_argument("--title", default=None, help="figure title")
    g.add_argument("--dpi", type=int, default=300, help="resolution of raster output")
    g.add_argument("--figsize", type=float, nargs=2, default=None, metavar=("W", "H"),
                   help="figure size in inches")


def _broaden_kwargs(args: argparse.Namespace) -> Dict[str, object]:
    return dict(
        sigma=args.sigma, fwhm=args.fwhm, lineshape=args.lineshape, eta=args.eta,
        gamma=args.gamma, de=args.de, reference=args.reference,
        emin=None if args.erange is None else args.erange[0],
        emax=None if args.erange is None else args.erange[1],
        group_by=args.group_by, groups=_groups(args.group), orbitals=args.orbitals,
        weighting=args.weighting, per_atom=args.per_atom,
        natoms=_mapping(args.natoms, "--natoms", int) or None,
        scale=args.scale, method=args.method,
    )


def _load_set(args: argparse.Namespace) -> PDOSSet:
    pset = PDOSSet.from_files(args.files, steps=getattr(args, "steps", None),
                              last_only=getattr(args, "last_step", False))
    kinds = getattr(args, "kinds", None)
    if kinds:
        keep = [p for p in pset if p.name in kinds or p.element in kinds]
        if not keep:
            raise ValueError(f"--kinds {kinds} matches nothing; available: {pset.names}")
        pset = PDOSSet(keep, check=False)
    return pset


def _make_curves(args: argparse.Namespace, total: bool) -> DOSCurves:
    """Raw files -> broadened curves; CP2K curve files -> taken as they are."""
    files = expand_paths(args.files)
    flags = [is_curve_file(f) for f in files]
    if all(flags):
        log.info("Input is already broadened by CP2K; using the curves as written.")
        return curves_from_cp2k(files, group_by=args.group_by, groups=_groups(args.group),
                                orbitals=args.orbitals, total=total)
    if any(flags):
        raise ValueError(
            "mixing raw .pdos files and CP2K-broadened '_curve' files; give one sort only"
        )
    kw = _broaden_kwargs(args)
    if args.stream:
        if args.kinds:
            raise ValueError("--kinds cannot be combined with --stream; pass only those files")
        return stream_broaden(files, steps=args.steps, total=total, **kw)
    pset = _load_set(args)
    if total:
        comp = pset.completeness()
        if comp is not None and comp < 0.98:
            log.warning(
                "The loaded kinds account for only %.1f%% of each state: the 'Total' "
                "curve is the sum of the files given, not the full DOS. Pass the PDOS "
                "files of all kinds to get the true total.", 100 * comp,
            )
    return pset.broaden(total=total, **kw)


def _report_levels(curves: DOSCurves) -> None:
    m = curves.meta
    bits = [f"reference = {curves.reference} ({curves.reference_energy:.4f} eV absolute)"]
    if m.get("vbm") is not None and m.get("cbm") is not None:
        gap = float(m["cbm"]) - float(m["vbm"])  # type: ignore[arg-type]
        metallic = m.get("metallic")
        bits.append("metallic / smeared occupations" if metallic else f"gap = {gap:.3f} eV")
    if int(m.get("n_frames", 1)) > 1:  # type: ignore[arg-type]
        bits.append(f"averaged over {m['n_frames']} steps")
    log.info("; ".join(bits))


# --------------------------------------------------------------------------- #
# pdos
# --------------------------------------------------------------------------- #
def _cmd_pdos(args: argparse.Namespace) -> int:
    curves = _make_curves(args, total=args.total_dos)
    _report_levels(curves)
    if not args.no_output:
        curves.save(args.output, beta_negative=args.beta_negative)
        log.info("Wrote %d curves to %s", len(curves.curves), args.output)
    if args.plot or args.plot_output:
        from .plotting import plot_dos, save_figure

        out = args.plot_output or "pdos.png"
        fig, _ = plot_dos(
            curves, kind=args.plot_type, select=args.curves, xlim=args.xlim, ylim=args.ylim,
            fill=args.fill, stack=args.stack, title=args.title, mark_edges=args.mark_edges,
            **({"figsize": tuple(args.figsize)} if args.figsize else {}),
        )
        save_figure(fig, out, args.dpi)
        log.info("Wrote plot to %s", out)
    return 0


# --------------------------------------------------------------------------- #
# info
# --------------------------------------------------------------------------- #
def _fmt(value: Optional[float], fmt: str = "{:.4f}") -> str:
    return "-" if value is None else fmt.format(value)


def _cmd_info(args: argparse.Namespace) -> int:
    pset = _load_set(args)
    summary = analysis.summarise(pset)
    pops = analysis.populations(pset, group_by=args.group_by, groups=_groups(args.group))
    char = analysis.edge_character(pset, width=args.edge_width, group_by=args.group_by,
                                   groups=_groups(args.group))
    if args.json:
        payload = {"summary": summary, "populations": pops, "edge_character": char}
        json.dump(payload, sys.stdout, indent=2, default=float)
        sys.stdout.write("\n")
        return 0
    w = sys.stdout.write
    w(f"Files: {summary['n_files']}   steps: {summary['steps']}   "
      f"spin-polarised: {'yes' if summary['spin_polarised'] else 'no'}\n")
    if summary["steps"] > 1:  # type: ignore[operator]
        w("(levels below are for the first step)\n")
    w("\nProjections\n")
    w(f"  {'name':<12}{'type':<8}{'spin':<7}{'atoms':>6}{'states':>8}  columns\n")
    for p in summary["projections"]:  # type: ignore[union-attr]
        w(f"  {p['name']:<12}{p['type']:<8}{(p['spin'] or '-'):<7}"
          f"{(p['natoms'] if p['natoms'] else '-'):>6}{p['n_states']:>8}  "
          f"{' '.join(p['columns'])}\n")
    w("\nEnergy levels (eV, absolute)\n")
    w(f"  {'spin':<7}{'E_Fermi':>11}{'VBM':>11}{'CBM':>11}{'gap':>9}{'electrons':>11}"
      "  occupations\n")
    for c in summary["spin_channels"]:  # type: ignore[union-attr]
        w(f"  {(c['spin'] or '-'):<7}{c['efermi_eV']:>11.4f}{_fmt(c['vbm_eV']):>11}"
          f"{_fmt(c['cbm_eV']):>11}{_fmt(c['gap_eV'], '{:.3f}'):>9}{c['n_electrons']:>11.3f}  "
          f"{'fractional' if c['fractional_occupations'] else 'integer'}\n")
    if summary["gap_eV"] is None:
        w("  No unoccupied states in the files: no gap can be given "
          "(request them with NLUMO / ADDED_MOS).\n")
    elif summary["metallic"]:
        w("  Smeared occupations or no gap: treat as metallic; the 'gap' is just the "
          "spacing of the levels around E_F.\n")
    else:
        w(f"  Fundamental gap (all spins): {summary['gap_eV']:.3f} eV\n")
    comp = summary["completeness"]
    if comp is not None:
        note = "complete set of kinds" if comp > 0.98 else "some kinds are missing"
        w(f"  Sum of kind projections per state: {comp:.4f} ({note})\n")
    w("\nPopulations (sum over states of weight x occupation)\n")
    w(f"  {'group':<14}{'orbital':<9}{'spin':<8}{'states':>10}{'electrons':>11}\n")
    for r in pops:
        states = "-" if np.isnan(r["states"]) else f"{r['states']:.3f}"  # type: ignore[arg-type]
        w(f"  {r['group']:<14}{r['orbital']:<9}{(r['spin'] or '-'):<8}"
          f"{states:>10}{r['electrons']:>11.3f}\n")
    for edge, comp_ in char.items():
        w(f"\nCharacter of the states within {args.edge_width:g} eV of the {edge.upper()}\n")
        for key, frac in list(comp_.items())[:8]:
            if frac >= 0.005:
                w(f"  {key:<16}{100 * frac:6.1f} %\n")
    return 0


# --------------------------------------------------------------------------- #
# moments
# --------------------------------------------------------------------------- #
def _cmd_moments(args: argparse.Namespace) -> int:
    pset = _load_set(args)
    emin, emax = (args.window if args.window else (None, None))
    res = analysis.band_moments(pset, args.of, args.orbital, args.reference, emin, emax)
    edge = None
    if args.upper_edge:
        names = args.of
        sel = PDOSSet(
            [p for p in pset if names is None or p.name in names or p.element in names],
            check=False,
        )
        cur = sel.broaden(sigma=args.sigma, de=args.de, reference=args.reference,
                          orbitals="sum" if args.orbital == "all" else "l",
                          group_by=lambda p: "band", emin=emin, emax=emax)
        y = cur.get("band", None if args.orbital == "all" else args.orbital, "sum")
        edge = analysis.upper_band_edge(cur.energy, y)
    if args.json:
        payload = {k: v.as_dict() for k, v in res.items()}
        if edge is not None:
            payload["upper_band_edge_eV"] = edge  # type: ignore[assignment]
        json.dump(payload, sys.stdout, indent=2, default=float)
        sys.stdout.write("\n")
        return 0
    w = sys.stdout.write
    label = "+".join(args.of) if args.of else "all kinds"
    ref = args.reference if isinstance(args.reference, str) else f"{args.reference:g} eV"
    win = "all states" if not args.window else f"{emin:g} .. {emax:g} eV"
    w(f"{args.orbital}-band of {label}; energies relative to {ref}; window: {win}\n\n")
    w(f"  {'spin':<7}{'centre':>10}{'width':>9}{'skew':>8}{'kurt':>8}"
      f"{'filling':>9}{'states':>9}{'occ. centre':>13}\n")
    for spin, m in res.items():
        w(f"  {spin:<7}{m.centre:>10.4f}{m.width:>9.4f}{m.skewness:>8.3f}{m.kurtosis:>8.3f}"
          f"{m.filling:>9.3f}{m.weight:>9.3f}{_fmt(m.centre_occupied):>13}\n")
    if edge is not None:
        w(f"\n  Upper band edge (Hilbert-transform peak, sigma = {args.sigma:g} eV): "
          f"{edge:.4f} eV\n")
    return 0


# --------------------------------------------------------------------------- #
# bands
# --------------------------------------------------------------------------- #
def _cmd_bands(args: argparse.Namespace) -> int:
    bs = read_bandstructure(args.file)
    cell = _cell(args.cell)
    w = sys.stdout.write
    nk = bs.energies.shape[1]
    w(f"{args.file}: {len(bs.sets)} k-point set(s), {nk} k-points, {bs.nbands} bands, "
      f"{bs.nspin} spin channel(s)\n")
    gap = None
    try:
        gap = bs.band_gap(efermi=args.efermi, n_occupied=args.nocc)
    except ValueError as exc:
        w(f"Band gap not determined: {exc}\n")
    if gap is not None:
        def kfmt(k):
            return "(" + ", ".join(f"{x:.4f}" for x in k) + ")"

        if gap.is_metal:
            w("No gap along this path (metal, or bands cross the chosen Fermi level).\n")
        else:
            w(f"VBM {gap.vbm:.4f} eV at k = {kfmt(gap.vbm_kpoint)}, band {gap.vbm_band}\n")
            w(f"CBM {gap.cbm:.4f} eV at k = {kfmt(gap.cbm_kpoint)}, band {gap.cbm_band}\n")
            w(f"Gap {gap.gap:.4f} eV ({'direct' if gap.is_direct else 'indirect'}); "
              f"smallest direct gap {gap.direct_gap:.4f} eV at k = {kfmt(gap.direct_kpoint)}\n")
            w("(Gap along the sampled path only; extrema off the path are not seen.)\n")
    shift = 0.0
    ref = args.reference
    if isinstance(ref, float):
        shift = ref
    elif ref == "fermi":
        if args.efermi is None:
            raise ValueError("--reference fermi needs --efermi (the .bs file does not store it)")
        shift = args.efermi
    elif ref in ("vbm", "cbm", "midgap", "auto"):
        if gap is None:
            raise ValueError(f"--reference {ref} needs occupations, --efermi or --nocc")
        shift = {"vbm": gap.vbm, "auto": gap.vbm, "cbm": gap.cbm,
                 "midgap": 0.5 * (gap.vbm + gap.cbm)}[ref]
    if args.mass:
        if cell is None:
            raise ValueError("--mass needs --cell to convert k-points to 1/Angstrom")
        if gap is None or gap.is_metal:
            raise ValueError("--mass needs a gapped band structure")
        k = bs.kpoints
        carriers = (
            ("hole (VBM)", gap.vbm_band, gap.vbm_kpoint, gap.spin_vbm),
            ("electron (CBM)", gap.cbm_band, gap.cbm_kpoint, gap.spin_cbm),
        )
        for name, band, kpt, spin in carriers:
            for idx in np.where(np.all(np.isclose(k, kpt, atol=1e-8), axis=1))[0]:
                try:
                    res = bs.effective_mass(band, int(idx), cell, args.mass_points, spin)
                except ValueError:
                    continue
                d = ", ".join(f"{x:+.3f}" for x in res["direction"])  # type: ignore[union-attr]
                w(f"m*/m_e {name}: {res['mass']:+.3f} along ({d}), {res['npoints']}-point "
                  f"{'one-sided ' if res['one_sided'] else ''}fit, rms {res['residual']:.1e} eV\n")
    if args.csv:
        bs.to_csv(args.csv, cell=cell, shift=shift)
        log.info("Wrote bands to %s", args.csv)
    if args.plot or args.plot_output:
        from .plotting import plot_bands, plot_bands_dos, reference_label, save_figure

        out = args.plot_output or "bands.png"
        label = ref if isinstance(ref, str) else f"{ref:g} eV"
        kw = dict(shift=shift, cell=cell, ylim=args.ylim, title=args.title,
                  mark_gap=args.mark_gap, efermi=args.efermi, n_occupied=args.nocc)
        if args.pdos:
            pfiles = expand_paths(args.pdos)
            if all(is_curve_file(f) for f in pfiles):
                # k-point calculations: CP2K only writes broadened curves,
                # already on its own energy zero.
                curves = curves_from_cp2k(pfiles, group_by=args.group_by,
                                          groups=_groups(args.group), total=True)
                log.info("PDOS panel uses CP2K's curves and their energy zero (%s); "
                         "make sure --reference matches it.", curves.reference)
            else:
                pset = PDOSSet.from_files(pfiles, last_only=True)
                pref = shift if ref != "absolute" else "absolute"
                if args.pdos_reference is not None:
                    pref = args.pdos_reference
                curves = pset.broaden(sigma=args.sigma, de=args.de, reference=pref,
                                      group_by=args.group_by, groups=_groups(args.group),
                                      total=True)
                if isinstance(ref, str) and ref != "auto":
                    curves.reference = ref
            fig, _ = plot_bands_dos(bs, curves, dos_kind=args.plot_type, **kw)
        else:
            fig, _ = plot_bands(bs, ylabel=reference_label(label) if isinstance(ref, str)
                                and ref != "auto" else None, **kw)
        save_figure(fig, out, args.dpi)
        log.info("Wrote plot to %s", out)
    return 0


# --------------------------------------------------------------------------- #
# dos (total DOS files)
# --------------------------------------------------------------------------- #
def _cmd_dos(args: argparse.Namespace) -> int:
    blocks = read_dos(args.file)
    d = blocks[-1]
    w = sys.stdout.write
    flavour = "broadened curve" if d.broadening else "raw histogram / state list"
    w(f"{args.file}: step {d.step}, {d.energy.size} points, channels: "
      f"{', '.join(d.density)}; {flavour}\n")
    if d.efermi:
        w("E_Fermi = " + ", ".join(f"{x:.4f}" for x in d.efermi) + " eV\n")
    if d.ehoco:
        w("E_HOCO  = " + ", ".join(f"{x:.4f}" for x in d.ehoco) + " eV\n")
    shift = 0.0
    reference = "absolute"
    already = d.energy_zero and "ABSOLUTE" not in d.energy_zero.upper()
    if already:
        reference = {"HOCO": "vbm", "FERMI": "fermi"}.get(
            d.energy_zero.split("->")[-1].strip().upper(), "absolute")  # type: ignore[union-attr]
        w(f"Energy axis already referenced by CP2K ({d.energy_zero}).\n")
    elif args.reference == "fermi" and d.efermi:
        shift, reference = max(d.efermi), "fermi"
    elif args.reference == "vbm" and d.ehoco:
        shift, reference = max(d.ehoco), "vbm"
    elif isinstance(args.reference, float):
        shift, reference = args.reference, f"{args.reference:g} eV"
    curves: "OrderedDict" = OrderedDict()
    for spin, y in d.density.items():
        s = None if spin == "total" else spin
        e = d.energies.get(spin, d.energy)
        curves[(TOTAL, "", s)] = np.interp(d.energy, e, y) if e is not d.energy else y
        if args.occupied:
            curves[("Occupied", "", s)] = d.occupied[spin]
    unit = "states/eV" if d.broadening else "as written by CP2K"
    out = DOSCurves(d.energy - shift, curves, reference, shift, unit,
                    {"files": args.file})
    if args.output:
        out.save(args.output)
        log.info("Wrote %s", args.output)
    if args.plot or args.plot_output:
        from .plotting import plot_dos, save_figure

        fig, _ = plot_dos(out, select=["Occupied"] if args.occupied else None,
                          show_total=True, xlim=args.xlim, ylim=args.ylim, title=args.title)
        save_figure(fig, args.plot_output or "dos.png", args.dpi)
        log.info("Wrote plot to %s", args.plot_output or "dos.png")
    return 0


# --------------------------------------------------------------------------- #
# compare
# --------------------------------------------------------------------------- #
def _cmd_compare(args: argparse.Namespace) -> int:
    from .plotting import plot_comparison, save_figure

    sets = _mapping(args.set, "--set")
    if len(sets) < 2:
        raise ValueError("give at least two datasets: --set LABEL=FILES --set LABEL=FILES")
    group, orbital = (args.curve.split(":") + [None])[:2]
    datasets: "OrderedDict[str, DOSCurves]" = OrderedDict()
    kw = _broaden_kwargs(args)
    for label, pattern in sets.items():
        files = [f for f in str(pattern).split(",") if f]
        pset = PDOSSet.from_files(files, steps=args.steps, last_only=args.last_step)
        datasets[label] = pset.broaden(total=True, **kw)
    fig, _ = plot_comparison(datasets, args.curve, xlim=args.xlim, ylim=args.ylim,
                             title=args.title)
    out = args.plot_output or "compare.png"
    save_figure(fig, out, args.dpi)
    log.info("Wrote plot to %s", out)
    if args.output:
        first = next(iter(datasets.values()))
        merged: "OrderedDict" = OrderedDict()
        for label, cur in datasets.items():
            y = cur.get(group, orbital, "sum")
            merged[(label, "", None)] = np.interp(first.energy, cur.energy, y, left=0, right=0)
        DOSCurves(first.energy, merged, first.reference, 0.0, first.unit,
                  {"files": f"curve {args.curve} of " + ", ".join(datasets)}).save(args.output)
        log.info("Wrote %s", args.output)
    return 0


# --------------------------------------------------------------------------- #
# input
# --------------------------------------------------------------------------- #
_INPUT_CLASSIC = """\
! CP2K up to 2026.1 -- goes inside &FORCE_EVAL / &DFT
&PRINT
  &PDOS
    NLUMO {nlumo}              ! unoccupied states to include (-1 = all)
{components}{append}{ldos}{each}  &END PDOS
&END PRINT
"""

_INPUT_2026 = """\
! CP2K 2026.2 and later -- goes inside &FORCE_EVAL / &DFT
! (check the &DFT/&PRINT/&DOS page of the input reference of your version)
&PRINT
  &DOS
    NLUMO {nlumo}              ! unoccupied states to include (-1 = all)
{append}    &PDOS
{components2}    &END PDOS
{ldos}{curve}{each}  &END DOS
&END PRINT
"""


def _cmd_input(args: argparse.Namespace) -> int:
    ldos = ""
    for item in args.ldos or []:
        ldos += f"    &LDOS\n      LIST {item}\n"
        if args.components:
            ldos += "      COMPONENTS\n"
        ldos += "    &END LDOS\n"
    each = ""
    if args.md_every:
        each = f"    &EACH\n      MD {args.md_every}\n    &END EACH\n"
    append = "    APPEND                 ! all MD steps in one file\n" if args.md_every else ""
    if args.style == "classic":
        text = _INPUT_CLASSIC.format(
            nlumo=args.nlumo,
            components="    COMPONENTS             ! px, py, pz, d-2 ... instead of p, d\n"
            if args.components else "",
            append=append, ldos=ldos, each=each,
        )
    else:
        curve = ""
        if args.curve:
            curve = (
                "    &CURVE                 ! CP2K's own broadened curves (needed with k-points)\n"
                "      ENERGY_UNIT EV\n      ENERGY_ZERO AUTO\n"
                "      &BROADEN\n        TYPE GAUSSIAN\n        WIDTH [eV] 0.1   ! FWHM\n"
                "      &END BROADEN\n    &END CURVE\n"
            )
        text = _INPUT_2026.format(
            nlumo=args.nlumo,
            components2="      COMPONENTS\n" if args.components else "",
            append=append, ldos=ldos, curve=curve, each=each,
        )
    sys.stdout.write(text)
    sys.stdout.write(
        "! With OT, add unoccupied states via NLUMO as above; with diagonalisation\n"
        "! also set &SCF / ADDED_MOS. A gap or a conduction band needs empty states.\n"
    )
    return 0


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    from . import __version__

    parser = argparse.ArgumentParser(
        prog="cp2k-pdos",
        description="Post-process the DOS-related output of CP2K: projected DOS, "
                    "local DOS, total DOS and band structures.",
        epilog="Run 'cp2k-pdos COMMAND -h' for the options of a command. "
               "'cp2k-pdos FILES...' is short for 'cp2k-pdos pdos FILES...'.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    quiet = argparse.ArgumentParser(add_help=False)
    quiet.add_argument("-q", "--quiet", action="store_true", help="only print warnings")

    # pdos ---------------------------------------------------------------- #
    p = sub.add_parser(
        "pdos", parents=[quiet],
        help="broaden raw PDOS / LDOS files into curves; write a table and a plot",
        description="Broaden CP2K's state-resolved PDOS files (*-k<N>-1.pdos per kind, "
                    "*-list<N>-1.pdos per atom list, ALPHA_/BETA_ per spin) into curves. "
                    "Files holding several MD steps are averaged. Broadened '*_curve*' "
                    "files written by CP2K >= 2026.2 are accepted too and used as they are.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="examples:\n"
               "  cp2k-pdos pdos proj-k*-1.pdos -s 0.1 --total-dos --plot\n"
               "  cp2k-pdos pdos proj-ALPHA_k*.pdos proj-BETA_k*.pdos --reference vbm \\\n"
               "      --plot-type both --xlim -6 4 --plot-output pdos.pdf\n"
               "  cp2k-pdos pdos md-k*-1.pdos --steps 200::10 --stream --group-by element --plot",
    )
    p.add_argument("files", nargs="+", metavar="FILE", help="CP2K .pdos files (globs allowed)")
    p.add_argument("-o", "--output", default="pdos.dat",
                   help="table of the curves; .dat, .csv, .npz or .json (default: pdos.dat)")
    p.add_argument("--no-output", action="store_true", help="do not write the table")
    p.add_argument("--total-dos", action="store_true",
                   help="add the total DOS (sum over all kinds given)")
    p.add_argument("--beta-negative", action="store_true",
                   help="write beta-spin columns with a minus sign (ready for mirrored plots)")
    p.add_argument("--chunk-size", type=int, default=None, help=argparse.SUPPRESS)
    _add_broadening(p)
    _add_selection(p)
    _add_plot(p, "pdos.png")
    g = p.add_argument_group("plot content")
    g.add_argument("--plot-type", choices=("total", "orbital", "both"), default="total",
                   help="one line per group, per group and orbital, or both")
    g.add_argument("--curves", nargs="+", default=None, metavar="GROUP[:ORB[:SPIN]]",
                   help="plot exactly these curves, e.g. --curves Ni:d O:p Total")
    g.add_argument("--fill", action="store_true", help="shade the area under each group")
    g.add_argument("--stack", action="store_true",
                   help="stack the groups (composition of the total DOS)")
    g.add_argument("--mark-edges", action="store_true", help="mark VBM and CBM")
    p.set_defaults(func=_cmd_pdos)

    # info ---------------------------------------------------------------- #
    p = sub.add_parser(
        "info", parents=[quiet],
        help="what is in the files: levels, gap, populations, band-edge character",
        description="Summarise raw PDOS files without broadening anything: projections "
                    "found, Fermi level, VBM/CBM and gap per spin, electron populations "
                    "per kind and orbital, and which atoms/orbitals the band edges are made of.",
    )
    p.add_argument("files", nargs="+", metavar="FILE")
    p.add_argument("--kinds", nargs="+", default=None, metavar="NAME")
    p.add_argument("--group-by", choices=("kind", "element", "file"), default="kind")
    p.add_argument("--group", action="append", default=None, metavar="NAME=A,B,..")
    p.add_argument("--edge-width", type=float, default=0.3,
                   help="energy window (eV) below the VBM / above the CBM used for the "
                        "band-edge character (default 0.3)")
    p.add_argument("--steps", type=_steps, default=None)
    p.add_argument("--last-step", action="store_true")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.set_defaults(func=_cmd_info)

    # moments ------------------------------------------------------------- #
    p = sub.add_parser(
        "moments", parents=[quiet],
        help="band centre, width, filling of one orbital (e.g. the d-band centre)",
        description="First to fourth moments of a projected band, computed from the "
                    "discrete states (independent of any broadening).",
        epilog="example: cp2k-pdos moments slab-k*-1.pdos --of Pt --orbital d --window -10 5",
    )
    p.add_argument("files", nargs="+", metavar="FILE")
    p.add_argument("--of", nargs="+", default=None, metavar="NAME",
                   help="kinds / lists / elements to pool (default: all)")
    p.add_argument("--orbital", default="d", help="s, p, d, f, an m label, or 'all' (default d)")
    p.add_argument("--reference", type=_reference, default="fermi", metavar="REF")
    p.add_argument("--window", type=float, nargs=2, default=None, metavar=("EMIN", "EMAX"),
                   help="energy window relative to the reference")
    p.add_argument("--upper-edge", action="store_true",
                   help="also report the upper band edge (peak of the Hilbert transform)")
    p.add_argument("-s", "--sigma", type=float, default=0.1,
                   help="broadening used for --upper-edge only (eV)")
    p.add_argument("-d", "--de", type=float, default=0.01)
    p.add_argument("--kinds", nargs="+", default=None, help=argparse.SUPPRESS)
    p.add_argument("--steps", type=_steps, default=None)
    p.add_argument("--last-step", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_cmd_moments)

    # bands --------------------------------------------------------------- #
    p = sub.add_parser(
        "bands", parents=[quiet],
        help="band structure (.bs): gap, effective masses, plot, CSV",
        description="Read a CP2K band-structure file (current or CP2K <= 7 layout), "
                    "report the gap along the path, and plot it -- optionally next to a PDOS.",
        epilog="example: cp2k-pdos bands WO3.bs --reference vbm --ylim -6 8 --mark-gap "
               "--cell 3.81 --mass --pdos WO3-k*-1.pdos --plot",
    )
    p.add_argument("file", metavar="FILE.bs")
    p.add_argument("--reference", type=_reference, default="vbm", metavar="REF",
                   help="zero of the energy axis: vbm (default), cbm, midgap, fermi "
                        "(needs --efermi), absolute or a number in eV")
    p.add_argument("--efermi", type=float, default=None,
                   help="Fermi energy in eV (decides occupation for old files without it)")
    p.add_argument("--nocc", type=int, default=None,
                   help="number of occupied bands per spin channel")
    p.add_argument("--cell", type=float, nargs="+", default=None,
                   help="cell in Angstrom: a | a b c | a b c alpha beta gamma | 9 numbers; "
                        "gives true path lengths and is needed for --mass")
    p.add_argument("--mass", action="store_true", help="effective masses at VBM and CBM")
    p.add_argument("--mass-points", type=int, default=5, help="points in the parabolic fit")
    p.add_argument("--csv", default=None, metavar="FILE", help="write the bands as CSV")
    p.add_argument("--mark-gap", action="store_true", help="mark VBM and CBM in the plot")
    p.add_argument("--pdos", nargs="+", default=None, metavar="FILE",
                   help="raw PDOS files to draw next to the bands")
    p.add_argument("--pdos-reference", type=_reference, default=None, metavar="REF",
                   help="reference of the PDOS panel if it should differ from the absolute "
                        "energy used for the bands")
    p.add_argument("-s", "--sigma", type=float, default=0.1)
    p.add_argument("-d", "--de", type=float, default=0.01)
    p.add_argument("--group-by", choices=("kind", "element", "file"), default="kind")
    p.add_argument("--group", action="append", default=None, metavar="NAME=A,B,..")
    p.add_argument("--plot-type", choices=("total", "orbital", "both"), default="total")
    _add_plot(p, "bands.png")
    p.set_defaults(func=_cmd_bands)

    # dos ----------------------------------------------------------------- #
    p = sub.add_parser(
        "dos", parents=[quiet], help="convert / plot a CP2K total-DOS (.dos) file",
        description="Read the total DOS written by &DFT/&PRINT/&DOS (any CP2K version, "
                    "restricted or spin-polarised), convert to eV, save and plot.",
    )
    p.add_argument("file", metavar="FILE.dos")
    p.add_argument("-o", "--output", default=None, help="table (.dat, .csv, .npz, .json)")
    p.add_argument("--reference", type=_reference, default="fermi", metavar="REF",
                   help="fermi (default), vbm, absolute or a number in eV")
    p.add_argument("--occupied", action="store_true", help="also show the occupied DOS")
    _add_plot(p, "dos.png")
    p.set_defaults(func=_cmd_dos)

    # compare ------------------------------------------------------------- #
    p = sub.add_parser(
        "compare", parents=[quiet],
        help="overlay the same curve from several calculations",
        description="Broaden several calculations identically and overlay one curve of "
                    "each, e.g. the framework with and without an adsorbate.",
        epilog="example: cp2k-pdos compare --set 'bare=bare-k*-1.pdos' "
               "--set 'CO2 loaded=co2-k*-1.pdos' --curve O:p --reference vbm --xlim -8 4",
    )
    p.add_argument("--set", action="append", required=True, metavar="LABEL=FILES",
                   help="a dataset: label and comma-separated files / globs (repeatable)")
    p.add_argument("--curve", default=TOTAL, metavar="GROUP[:ORB[:SPIN]]",
                   help="which curve to compare (default: Total)")
    p.add_argument("-o", "--output", default=None, help="table with one column per dataset")
    _add_broadening(p)
    _add_selection(p)
    _add_plot(p, "compare.png")
    p.set_defaults(func=_cmd_compare)

    # input --------------------------------------------------------------- #
    p = sub.add_parser(
        "input", help="print the CP2K input section that produces PDOS files",
        description="Print a ready-to-paste &PRINT section for CP2K.",
    )
    p.add_argument("--style", choices=("classic", "2026"), default="classic",
                   help="classic: &PRINT/&PDOS (CP2K <= 2026.1); 2026: &PRINT/&DOS (>= 2026.2)")
    p.add_argument("--nlumo", type=int, default=-1, help="unoccupied states (-1 = all)")
    p.add_argument("--components", action="store_true", help="m-resolved columns")
    p.add_argument("--ldos", action="append", metavar="'I J K..'",
                   help="add an &LDOS section for this atom list, e.g. --ldos '1..12 40'")
    p.add_argument("--md-every", type=int, default=None, metavar="N",
                   help="print every N MD steps and APPEND them to one file")
    p.add_argument("--curve", action="store_true",
                   help="(2026 style) also request CP2K's own broadened curves")
    p.set_defaults(func=_cmd_input, quiet=False)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point. Returns the process exit code."""
    argv = list(sys.argv[1:] if argv is None else argv)
    # 'cp2k-pdos FILES...' (and the legacy 'pdos-process FILES...') -> 'pdos'.
    if argv and argv[0] not in COMMANDS and argv[0] not in ("-h", "--help", "--version"):
        argv = ["pdos"] + argv
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 2
    logging.basicConfig(
        level=logging.WARNING if getattr(args, "quiet", False) else logging.INFO,
        format="%(levelname)s: %(message)s", force=True,
    )
    if getattr(args, "chunk_size", None) is not None:
        log.warning("--chunk-size is no longer needed and is ignored "
                    "(use --stream for low-memory trajectory averaging).")
    try:
        return int(args.func(args) or 0)
    except (ValueError, KeyError, OSError, ImportError) as exc:
        msg = exc.args[0] if isinstance(exc, KeyError) and exc.args else exc
        log.error("%s", msg)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
