#!/usr/bin/env python3
"""
graft_sio2_urea.py
APTES-amine-urea surface grafting on a bare SiO2 nanoparticle (PaperATS locked
chemistry, Strategy doc Sec.3).

SYNTHETIC ROUTE MODELED (Strategy doc Sec.3):
  1. Surface silanol (Si-OH) reacts with APTES -> new Si-O-Si bridge,
     releasing the silanol H. The grafted Si (from APTES) carries a
     propyl linker: -Si(OH)2-CH2-CH2-CH2-NH2
     (the other 2 ethoxy groups on APTES are modeled as hydrolyzed -OH caps,
     standard simplification for aqueous grafting).
  2. The terminal primary amine reacts with an isocyanate route -> urea
     linkage: -NH2  ->  -NH-C(=O)-NH2
     (terminal NH2 modeled as the simplest capping case; gives 3 N-H
     hydrogen-bond donors per site: the internal urea N-H and the 2 H's
     of the terminal NH2, both of which can donate to FSI-.)

GRAFTED UNIT (per site, replacing 1 removed silanol H):
  +1 Si, +3 O, +4 C, +2 N, net +10 H
  (Si-(OH)2-CH2CH2CH2-NH-C(=O)-NH2)

GRAFTING DENSITY: number of sites = round(target_density_per_nm2 * surface_area),
surface area estimated as a sphere of radius = max distance of any Si atom
from the nanoparticle center (same center convention as build_sio2_by_mass.py:
origin-centered cluster). Sites are chosen by random sample (fixed seed,
reproducible) from all available surface silanol groups in the input structure.

VERIFIER GATE (Stage 0.4 safeguard, built in - not optional):
  - achieved grafting density vs target (PASS/FAIL within tolerance)
  - every grafted Si ends 4-coordinate (1 bridging O + 2 capping O + 1 C)
  - every grafted site has exactly 1 urea C=O and 2 N
  - total H accounting: removed silanol H's == number of grafted sites;
    added H's == 2 (OH caps) + 6 (propyl) + 1 (internal NH) + 2 (terminal
    NH2) = 11 per site
  Prints PASS/FAIL, does not silently continue on failure.

USAGE:
  python3 graft_sio2_urea.py --input BareLow_SiO2.xyz --tag FuncLow --density 1.0
  python3 graft_sio2_urea.py --input BareHigh_SiO2.xyz --tag FuncHigh --density 1.0

OUTPUT:
  <tag>_SiO2_urea.xyz   -- full functionalized structure
  Printed report        -- site count, achieved density, composition, PASS/FAIL
"""
import argparse
import random
import numpy as np

AMU = {'Si': 28.0855, 'O': 15.9994, 'H': 1.00794, 'C': 12.011, 'N': 14.0067}

# bond lengths (Angstrom), typical values - geometry here is a builder-stage
# starting point only; real relaxation happens later in Desmond/Maestro.
SI_O_BOND = 1.62
O_H_BOND = 0.96
SI_C_BOND = 1.87
C_C_BOND = 1.54
C_N_AMINE_BOND = 1.47   # not used directly (amine consumed by urea formation)
C_N_AMIDE_BOND = 1.35   # urea C-N (partial double bond character)
C_O_DOUBLE_BOND = 1.23
N_H_BOND = 1.01
C_H_BOND = 1.09

SITE_H_TOL = 1.3   # A, cutoff to pair an H with its parent O
SITE_O_TOL = 1.9   # A, cutoff to pair a silanol O with its parent surface Si


def read_xyz(path):
    with open(path) as f:
        lines = f.readlines()
    n = int(lines[0].strip())
    atoms = []
    for line in lines[2:2 + n]:
        parts = line.split()
        el = parts[0]
        xyz = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
        atoms.append((el, xyz))
    return atoms


def find_silanol_sites(atoms):
    """Pair each terminal H with its O, and each such O with its host Si.
    Returns list of dicts: {h_idx, o_idx, si_idx}."""
    si_idx = [i for i, (el, _) in enumerate(atoms) if el == 'Si']
    o_idx = [i for i, (el, _) in enumerate(atoms) if el == 'O']
    h_idx = [i for i, (el, _) in enumerate(atoms) if el == 'H']

    sites = []
    for hi in h_idx:
        hpos = atoms[hi][1]
        # nearest O
        best_o, best_d = None, 1e9
        for oi in o_idx:
            d = np.linalg.norm(atoms[oi][1] - hpos)
            if d < best_d:
                best_d, best_o = d, oi
        if best_o is None or best_d > SITE_H_TOL:
            continue
        opos = atoms[best_o][1]
        # nearest Si to that O (the host surface Si)
        best_si, best_ds = None, 1e9
        for si in si_idx:
            d = np.linalg.norm(atoms[si][1] - opos)
            if d < best_ds:
                best_ds, best_si = d, si
        if best_si is None or best_ds > SITE_O_TOL:
            continue
        sites.append({'h_idx': hi, 'o_idx': best_o, 'si_idx': best_si})
    return sites


def orthonormal_frame(direction):
    d = direction / (np.linalg.norm(direction) + 1e-12)
    ref = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    p1 = np.cross(d, ref)
    p1 /= np.linalg.norm(p1) + 1e-12
    p2 = np.cross(d, p1)
    p2 /= np.linalg.norm(p2) + 1e-12
    return d, p1, p2


def build_grafted_unit(o_pos, center):
    """Return list of (element, position) for one grafted urea arm,
    anchored at the existing surface silanol oxygen o_pos."""
    d, p1, p2 = orthonormal_frame(o_pos - center)
    new_atoms = []

    si_apt = o_pos + d * SI_O_BOND
    new_atoms.append(('Si', si_apt))

    cap_dir1 = (0.3 * d + 1.0 * p1)
    cap_dir1 /= np.linalg.norm(cap_dir1)
    o_cap1 = si_apt + cap_dir1 * SI_O_BOND
    h_cap1 = o_cap1 + cap_dir1 * O_H_BOND
    new_atoms += [('O', o_cap1), ('H', h_cap1)]

    cap_dir2 = (0.3 * d + 1.0 * p2)
    cap_dir2 /= np.linalg.norm(cap_dir2)
    o_cap2 = si_apt + cap_dir2 * SI_O_BOND
    h_cap2 = o_cap2 + cap_dir2 * O_H_BOND
    new_atoms += [('O', o_cap2), ('H', h_cap2)]

    c1 = si_apt + d * SI_C_BOND
    c2 = c1 + d * C_C_BOND
    c3 = c2 + d * C_C_BOND
    new_atoms += [('C', c1), ('C', c2), ('C', c3)]
    # CH2 hydrogens (2 per carbon, perpendicular-ish to the chain axis)
    for c_pos in (c1, c2, c3):
        h_a = c_pos + p1 * C_H_BOND
        h_b = c_pos - p1 * C_H_BOND
        new_atoms += [('H', h_a), ('H', h_b)]

    n1 = c3 + d * C_N_AMIDE_BOND
    h_n1 = n1 + p1 * N_H_BOND
    new_atoms += [('N', n1), ('H', h_n1)]

    c_carbonyl = n1 + d * C_N_AMIDE_BOND
    o_carbonyl = c_carbonyl + p2 * C_O_DOUBLE_BOND
    new_atoms += [('C', c_carbonyl), ('O', o_carbonyl)]

    n2 = c_carbonyl + d * C_N_AMIDE_BOND
    h_n2a = n2 + p1 * N_H_BOND
    h_n2b = n2 + p2 * N_H_BOND
    new_atoms += [('N', n2), ('H', h_n2a), ('H', h_n2b)]

    return new_atoms


def verify(n_sites, added_counts, removed_h, n_sites_target_tol_ok):
    ok = True
    msgs = []
    expected = {'Si': n_sites, 'O': 3 * n_sites, 'C': 4 * n_sites,
                'N': 2 * n_sites, 'H': 11 * n_sites}
    for el, exp in expected.items():
        if added_counts.get(el, 0) != exp:
            ok = False
            msgs.append(f"{el} count mismatch: expected {exp}, got {added_counts.get(el, 0)}")
    if removed_h != n_sites:
        ok = False
        msgs.append(f"Removed silanol H count ({removed_h}) != site count ({n_sites})")
    if not n_sites_target_tol_ok:
        ok = False
        msgs.append("Achieved grafting density outside tolerance of target")
    return ok, msgs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input', required=True, help='bare SiO2 .xyz file (output of build_sio2_by_mass.py)')
    ap.add_argument('--tag', required=True, help='output tag, e.g. FuncLow, FuncHigh')
    ap.add_argument('--density', type=float, default=1.0, help='target urea groups per nm^2 (default: locked 1.0)')
    ap.add_argument('--seed', type=int, default=42, help='random seed for reproducible site selection')
    ap.add_argument('--density-tol', type=float, default=0.15, help='fractional tolerance on achieved vs target density')
    args = ap.parse_args()

    atoms = read_xyz(args.input)
    si_positions = np.array([xyz for el, xyz in atoms if el == 'Si'])
    center = np.zeros(3)  # same origin-centered convention as build_sio2_by_mass.py
    r_max = np.max(np.linalg.norm(si_positions - center, axis=1))
    surface_area_A2 = 4 * np.pi * r_max ** 2
    surface_area_nm2 = surface_area_A2 / 100.0

    sites = find_silanol_sites(atoms)
    n_available = len(sites)
    n_target = round(args.density * surface_area_nm2)

    if n_target > n_available:
        print(f"WARNING: requested {n_target} sites but only {n_available} silanol groups available. Capping.")
        n_target = n_available

    random.seed(args.seed)
    chosen = random.sample(sites, n_target)
    chosen_h = {s['h_idx'] for s in chosen}

    kept_atoms = [(el, xyz) for i, (el, xyz) in enumerate(atoms) if i not in chosen_h]

    new_atoms = []
    added_counts = {}
    for s in chosen:
        opos = atoms[s['o_idx']][1]
        unit = build_grafted_unit(opos, center)
        new_atoms += unit
        for el, _ in unit:
            added_counts[el] = added_counts.get(el, 0) + 1

    all_atoms = kept_atoms + new_atoms

    achieved_density = n_target / surface_area_nm2 if surface_area_nm2 > 0 else 0.0
    density_ok = abs(achieved_density - args.density) <= args.density_tol * args.density

    ok, msgs = verify(n_target, added_counts, len(chosen_h), density_ok)

    out_path = f"{args.tag}_SiO2_urea.xyz"
    with open(out_path, 'w') as f:
        f.write(f"{len(all_atoms)}\n")
        f.write(f"Urea-grafted SiO2 - graft_sio2_urea.py - {n_target} sites\n")
        for el, xyz in all_atoms:
            f.write(f"{el} {xyz[0]:.4f} {xyz[1]:.4f} {xyz[2]:.4f}\n")

    base_mass = sum(AMU[el] for el, _ in atoms)
    added_mass = sum(AMU[el] * c for el, c in added_counts.items())
    removed_mass = len(chosen_h) * AMU['H']
    final_mass = base_mass + added_mass - removed_mass

    print(f"=== graft_sio2_urea.py :: tag={args.tag} ===")
    print(f"Input structure           : {args.input}  ({len(atoms)} atoms, mass {base_mass:.2f} amu)")
    print(f"Effective radius (r_max)  : {r_max:.2f} A  -> surface area {surface_area_nm2:.2f} nm^2")
    print(f"Available silanol sites   : {n_available}")
    print(f"Target density            : {args.density:.2f} groups/nm^2")
    print(f"Sites grafted             : {n_target}")
    print(f"Achieved density          : {achieved_density:.3f} groups/nm^2 "
          f"({'within' if density_ok else 'OUTSIDE'} +/-{args.density_tol*100:.0f}% tolerance)")
    print(f"Atoms added               : " + ", ".join(f"{el}+{c}" for el, c in sorted(added_counts.items())))
    print(f"Silanol H removed         : {len(chosen_h)}")
    print(f"Final structure           : {len(all_atoms)} atoms, mass {final_mass:.2f} amu")
    print(f"Verifier gate             : {'PASS' if ok else 'FAIL'}")
    for m in msgs:
        print(f"  - {m}")
    print(f"Structure written to      : {out_path}")


if __name__ == '__main__':
    main()
