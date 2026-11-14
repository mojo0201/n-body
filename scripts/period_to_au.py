#!/usr/bin/env python3
"""Convert an orbital period (yr) to semi-major axis (AU) using Kepler's 3rd law.

Usage:
    python3 period_to_au.py 79.762±0.019 --mass 1.0788±0.0029 0.9092±0.0025
    python3 period_to_au.py 547000+6600-4000 --mass 1.0788 0.9092 0.1221
    python3 period_to_au.py 547000+6600−4000 --mass 1.0788 0.9092 0.1221
"""

import argparse
import re
import sys

M_SUN_IN_EARTH = 332_946.0487
DEFAULT_MASS_SUN = (368_072.0 + 312_070.0) / M_SUN_IN_EARTH

# Numbers: allow sign + decimal + exponent.
_NUM  = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
# Error magnitudes are positive; forbid an embedded sign so "+6600-4000"
# parses cleanly as (upper=6600, lower=4000) rather than as (upper=6600) - (-4000).
_POS  = r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

# (value, upper_error, lower_error) or plain (value, None, None).
_ASYM = re.compile(rf"^\s*({_NUM})\s*\+\s*({_POS})\s*[-\u2212]\s*({_POS})\s*$")
_SYM  = re.compile(rf"^\s*({_NUM})\s*(?:±|\+/-|\+-)\s*({_POS})\s*$")
_PLAIN = re.compile(rf"^\s*({_NUM})\s*$")


def parse_quantity(text: str):
    """Return (value, upper_err_or_None, lower_err_or_None)."""
    m = _ASYM.match(text)
    if m:
        return float(m.group(1)), float(m.group(2)), float(m.group(3))
    m = _SYM.match(text)
    if m:
        e = float(m.group(2))
        return float(m.group(1)), e, e
    m = _PLAIN.match(text)
    if m:
        return float(m.group(1)), None, None
    raise ValueError(f"cannot parse quantity: {text!r}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Convert orbital period (yr) to semi-major axis (AU).")
    ap.add_argument("period", help="e.g. 79.762±0.019, 547000+6600-4000, or 547000.")
    ap.add_argument("--mass", nargs="+", default=[f"{DEFAULT_MASS_SUN:.6f}"],
                    help="One or more masses in M_sun; summed, uncertainties in quadrature.")
    ap.add_argument("-p", "--precision", type=int, default=6)
    args = ap.parse_args(argv)

    try:
        P, P_up, P_lo = parse_quantity(args.period)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr); return 1
    if P <= 0:
        print("error: period must be positive", file=sys.stderr); return 1

    M = 0.0
    M_up_sq = M_lo_sq = 0.0
    try:
        for t in args.mass:
            v, uu, ul = parse_quantity(t)
            if v <= 0:
                print(f"error: mass {t!r} must be positive", file=sys.stderr); return 1
            M += v
            if uu is not None: M_up_sq += uu * uu
            if ul is not None: M_lo_sq += ul * ul
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr); return 1

    a = (P * P * M) ** (1.0 / 3.0)

    # a = (P^2 M)^(1/3):  da/a = (2/3) dP/P + (1/3) dM/M
    # Combine period and mass terms in quadrature per side.
    up_terms, lo_terms = [], []
    if P_up is not None: up_terms.append((2.0 / 3.0) * P_up / P)
    if P_lo is not None: lo_terms.append((2.0 / 3.0) * P_lo / P)
    if M_up_sq:          up_terms.append((1.0 / 3.0) * (M_up_sq ** 0.5) / M)
    if M_lo_sq:          lo_terms.append((1.0 / 3.0) * (M_lo_sq ** 0.5) / M)
    a_up = a * (sum(t * t for t in up_terms) ** 0.5) if up_terms else None
    a_lo = a * (sum(t * t for t in lo_terms) ** 0.5) if lo_terms else None

    if len(args.mass) > 1:
        print("M_total = " + " + ".join(f"{parse_quantity(t)[0]:g}" for t in args.mass) + f" = {M:g} M_sun")
    else:
        print(f"M_total = {M:g} M_sun")
    print(f"P = {P:g} yr  =>  a = {a:.{args.precision}f} AU")

    if a_up is None and a_lo is None:
        return 0
    if a_up is not None and a_lo is not None:
        if abs(a_up - a_lo) < 1e-9 * max(a, 1.0):
            print(f"a = {a:.{args.precision}f} ± {(a_up + a_lo) / 2:.{args.precision}f} AU")
        else:
            print(f"a = {a:.{args.precision}f} +{a_up:.{args.precision}f} / -{a_lo:.{args.precision}f} AU")
    elif a_up is not None:
        print(f"a = {a:.{args.precision}f} +{a_up:.{args.precision}f} / -? AU")
    else:
        print(f"a = {a:.{args.precision}f} +? / -{a_lo:.{args.precision}f} AU")
    return 0


if __name__ == "__main__":
    sys.exit(main())