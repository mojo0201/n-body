#!/usr/bin/env python3
"""Add real inclinations (J2000 ecliptic) to assets/presets/solar_system.json.

Usage: python3 tools/add_solar_inclinations.py assets/presets/solar_system.json
Matches planets by name (case-insensitive), sets inc_deg / node_deg, and prints
what it changed. Existing omega_deg is left alone (it is read as the longitude
of periapsis whenever node_deg is non-zero)."""
import json, sys

PLANETS = {  # name: (inc, node)
    "mercury": (7.00, 48.33), "venus": (3.39, 76.68), "mars": (1.85, 49.56),
    "jupiter": (1.30, 100.46), "saturn": (2.49, 113.67), "uranus": (0.77, 74.01),
    "neptune": (1.77, 131.78), "ceres": (10.59, 80.31), "pluto": (17.16, 110.30),
    "eris": (43.87, 35.95), "haumea": (28.21, 122.17), "makemake": (29.03, 79.27),
    "halley": (162.26, 59.07), "pallas": (34.84, 173.1), "vesta": (7.14, 103.8),
}
MOONS = {"moon": (5.14, 0.0)}  # inclination to the planet's orbital plane

path = sys.argv[1]
d = json.load(open(path))

changed = []
for p in d.get("planets", []):
    n = str(p.get("name", "")).lower()
    for key in PLANETS:
        if n == key or n.startswith(key + " "):
            p["inc_deg"], p["node_deg"] = PLANETS[key]
            changed.append(p["name"])
    for m in p.get("moons", []):
        if str(m.get("name", "")).lower() in MOONS:
            m["inc_deg"], m["node_deg"] = MOONS[str(m["name"]).lower()]
            changed.append(m["name"] + " (moon)")
json.dump(d, open(path, "w"), indent=2)
print("updated:", ", ".join(changed) or "nothing (check the file's structure)")
