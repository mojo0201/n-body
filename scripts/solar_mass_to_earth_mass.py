#!/usr/bin/env python3
"""Convert solar masses to Earth masses.

Accepts plain values ("120", "0.9092") and ± uncertainties
("0.9092±0.0025", "0.9092 +/- 0.0025", "0.9092+-0.0025").
"""

import argparse
import re
import sys

SOLAR_MASS_IN_EARTH_MASSES = 332_946.0487

_NUM = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

_PATTERN = re.compile(
    rf"^\s*({_NUM})\s*(?:(?:±|\+/-|\+-)\s*({_NUM}))?\s*$"
)


def parse_quantity(text: str):
    """Return (value, uncertainty_or_None). Raises ValueError on bad input."""
    m = _PATTERN.match(text)
    if not m:
        raise ValueError(f"cannot parse quantity: {text!r}")
    value = float(m.group(1))
    unc = float(m.group(2)) if m.group(2) is not None else None
    return value, unc


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert solar masses (M_sun) to Earth masses (M_earth)."
    )
    parser.add_argument("solar_masses", nargs="+",
                        help="Mass(es) in M_sun, e.g. 120 or 0.9092±0.0025.")
    parser.add_argument("-p", "--precision", type=int, default=6,
                        help="Decimal places (default: 6).")
    args = parser.parse_args(argv)

    for text in args.solar_masses:
        try:
            value, unc = parse_quantity(text)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        if value < 0 or (unc is not None and unc < 0):
            print(f"error: negative mass/uncertainty in {text!r}", file=sys.stderr)
            return 1

        earth = value * SOLAR_MASS_IN_EARTH_MASSES
        if unc is None:
            print(f"{value:g} M_sun = {earth:.{args.precision}f} M_earth")
        else:
            earth_unc = unc * SOLAR_MASS_IN_EARTH_MASSES
            print(f"{value:g}±{unc:g} M_sun = "
                  f"{earth:.{args.precision}f}±{earth_unc:.{args.precision}f} M_earth")
    return 0


if __name__ == "__main__":
    sys.exit(main())