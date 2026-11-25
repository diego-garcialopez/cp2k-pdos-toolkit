"""Physical constants and small shared lookup tables."""

from __future__ import annotations

#: CODATA 2018 Hartree -> eV conversion factor.
HARTREE_TO_EV = 27.211386245988

#: hbar^2 / m_e in eV * Angstrom^2 (used for effective masses).
HBAR2_OVER_ME_EV_A2 = 7.619964231070

#: Angular-momentum letters in the order CP2K uses them.
L_LABELS = ("s", "p", "d", "f", "g", "h", "i")

#: Chemical symbols, used to map CP2K kind names (``Ni1``, ``O_w``, ``Fe_up``)
#: onto elements.
ELEMENTS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni "
    "Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I "
    "Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt "
    "Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr "
    "Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og"
).split()

_ELEMENT_SET = {e.lower() for e in ELEMENTS}


def element_of(kind: str) -> str:
    """Return the chemical element a CP2K kind name refers to.

    CP2K kinds are free-form labels (``Ni1``, ``O_w``, ``Fe_up``, ``C2a``).
    The element is the leading one- or two-letter chemical symbol of the
    name. A two-letter symbol is preferred when its second letter is lower
    case (``Ni1`` -> Ni, ``Co`` -> Co); with an upper-case second letter the
    one-letter reading wins when it is itself an element (``CO2`` -> C,
    ``OW`` -> O, and therefore also the ambiguous ``NI`` -> N) and the
    two-letter reading is used otherwise (``ZN1`` -> Zn). Use explicit groups
    when your kind names are ambiguous. Names that do not start with an element symbol
    (``list1``, ``X``) are returned unchanged.
    """
    letters = ""
    for ch in kind:
        if not ch.isalpha():
            break
        letters += ch
    one = letters[:1].lower()
    two = letters[:2].lower()
    two_ok = len(two) == 2 and two in _ELEMENT_SET
    one_ok = one in _ELEMENT_SET and letters[:1].isupper()
    if two_ok and letters[0].isupper() and letters[1].islower():
        return two.capitalize()
    if one_ok:
        return one.upper()
    if two_ok and letters[:2].isupper():
        return two.capitalize()
    return kind


def l_of(label: str) -> str:
    """Angular-momentum letter of an orbital column label.

    ``"s" -> "s"``, ``"py" -> "p"``, ``"d-2" -> "d"``, ``"f+3" -> "f"``.
    Labels that are not orbital labels (``"LDOS"``) are returned unchanged.
    """
    if label and label[0].lower() in L_LABELS:
        return label[0].lower()
    return label


def trapezoid(y, x):
    """Trapezoidal rule (``numpy.trapz`` was renamed in NumPy 2)."""
    import numpy as np

    func = getattr(np, "trapezoid", None) or np.trapz  # type: ignore[attr-defined]
    return func(y, x)
