#!/usr/bin/env python3
"""
Paper1_PTMC_Fillers - Stage 2.2 Equilibration Checkpoint (v2)

Parses a Desmond .ene file directly (no manual CSV export needed) and
checks whether Volume (proxy for density, since mass is fixed) and
Temperature have plateaued over the last fraction of the run.

USAGE (run once per system, from inside that system's own folder):
    python checkpoint_ene.py Paper1_Baseline_Equil.ene
    python checkpoint_ene.py Paper1_Baseline_Equil.ene --tail 0.2 --vol-slope-tol 1 --temp-slope-tol 1

.ene column layout (Desmond 8.3.131, confirmed from this project's file):
    0:time(ps) 1:E 2:E_p 3:E_k 4:E_c 5:E_x 6:E_f 7:P(bar) 8:V(A^3) 9:T(K)

GOTCHAS:
- Lines starting with '#' are header/metadata - skipped automatically.
- Volume is used as the density proxy (mass is constant for a given system,
  so %-range in V == %-range in density). This is only valid for checking
  ONE system's own plateau, not for comparing absolute density across
  different systems (different systems have different total mass).
- FAIL means: per roadmap Stage 2.2, extend equilibration - don't start
  production on a FAIL.
- Single-system script by design (cd-into-folder safeguard) - run
  separately inside each of the 10 system folders.

METHODOLOGY NOTE (for Methods section later):
The %-range check is a practical/descriptive heuristic, not a literature-
standard cutoff - no citable paper defines equilibration by a fixed %
range for amorphous polymer systems on ns timescales (chain relaxation is
too slow for that to be a fair bar; published protocols instead require
"no systematic drift", and stricter frameworks that DO use a fixed
numeric cutoff need 50-200ns runs to reach it). The SLOPE check added
below (linear regression of the tail window) is the primary, reviewer-
defensible criterion: it directly tests for "no systematic drift", which
is what published protocols actually rely on. Report both slope and %
range in the Methods table, but justify PASS/FAIL primarily on slope.
"""
import argparse
import sys
import numpy as np


def load_ene(path):
    time, V, T = [], [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 10:
                continue
            try:
                t = float(parts[0])
                v = float(parts[8])
                temp = float(parts[9])
            except ValueError:
                continue
            time.append(t)
            V.append(v)
            T.append(temp)
    return np.array(time), np.array(V), np.array(T)


def plateau_check(label, t, arr, tail_frac, slope_tol_pct_per_ns):
    """Two criteria reported:
    1. SLOPE (primary, reviewer-defensible): linear regression of the tail
       window. Normalized to %/ns of the tail mean, so it's comparable
       across systems/quantities. |slope| below slope_tol_pct_per_ns -> no
       systematic drift -> equilibrated, regardless of absolute noise.
    2. %-RANGE (secondary/descriptive only): reported for transparency,
       NOT used alone to fail a system - noisy quantities (e.g. raw
       instantaneous Temperature) can have a large range with zero drift.
    """
    n_tail = max(1, int(len(arr) * tail_frac))
    t_tail = t[-n_tail:]
    tail = arr[-n_tail:]
    mean_tail = tail.mean()
    pct_range = (tail.max() - tail.min()) / abs(mean_tail) * 100 if mean_tail != 0 else float("inf")

    # linear regression: value vs time(ns)
    t_ns = (t_tail - t_tail[0]) / 1000.0
    slope, intercept = np.polyfit(t_ns, tail, 1)
    slope_pct_per_ns = (slope / abs(mean_tail)) * 100 if mean_tail != 0 else float("inf")

    status = "PASS" if abs(slope_pct_per_ns) <= slope_tol_pct_per_ns else "FAIL - systematic drift detected"
    print(f"{label}: mean(last {tail_frac*100:.0f}%) = {mean_tail:.3f}")
    print(f"  slope = {slope_pct_per_ns:+.3f} %/ns (tol +/-{slope_tol_pct_per_ns:.2f} %/ns)  [PRIMARY]")
    print(f"  range = {pct_range:.2f}% over tail window  [descriptive only]")
    print(f"  -> {status}")
    return status.startswith("PASS")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("ene_file")
    p.add_argument("--tail", type=float, default=0.2,
                    help="fraction of frames (from the end) used as the plateau window")
    p.add_argument("--vol-slope-tol", type=float, default=1.0,
                    help="allowed |slope| for Volume/density, in %%/ns (primary criterion)")
    p.add_argument("--temp-slope-tol", type=float, default=1.0,
                    help="allowed |slope| for Temperature, in %%/ns (primary criterion)")
    args = p.parse_args()

    t, V, T = load_ene(args.ene_file)
    if len(V) == 0:
        print(f"ERROR: no numeric data rows parsed from {args.ene_file}")
        sys.exit(1)

    print(f"Loaded {len(V)} frames spanning {t[0]:.1f} to {t[-1]:.1f} ps\n")
    ok_v = plateau_check("Volume (density proxy)", t, V, args.tail, args.vol_slope_tol)
    ok_t = plateau_check("Temperature", t, T, args.tail, args.temp_slope_tol)

    print()
    if ok_v and ok_t:
        print("OVERALL: PASS - equilibration checkpoint cleared, safe to proceed to production.")
    else:
        print("OVERALL: FAIL - do not start production yet. Extend the NPT stage and re-check.")
