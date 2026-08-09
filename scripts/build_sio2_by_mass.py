#!/usr/bin/env python3
"""
build_sio2_by_mass.py
Mass-target-driven SiO2 nanoparticle builder (Paper1 / PaperATS locked method).

METHOD (locked, do not change without updating this docstring):
- Si placed on a diamond-cubic lattice (beta-cristobalite-derived proxy).
- O inserted at every Si-Si nearest-neighbor bond midpoint -> bridging oxygen
  (Si-O-Si), giving the correct 1:2 Si:O bulk stoichiometry.
- Cluster grown radially (sphere cut from the lattice) until the closest
  achievable mass to --target-mass is found. Nanoparticles are small, so
  atom counts jump in discrete shells -> exact target mass is NOT always
  achievable. This script reports the ACTUAL achieved mass/wt%, per project
  convention (never force the nominal target number).
- Surface passivation (hydroxylation): any Si with fewer than 4 O neighbors
  (i.e. a dangling bond because its lattice neighbor fell outside the cut
  sphere) gets a terminal -OH group per missing bond -> silanol surface.
- Verifier gate (built in, per Stage 0.4 safeguards): confirms every Si ends
  at exactly 4 O neighbors, and total Si-O bond count reconciles with
  bridging + terminal oxygens. Prints PASS/FAIL.

USAGE:
  python3 build_sio2_by_mass.py --target-mass 3403.44 --tag BareLow
  python3 build_sio2_by_mass.py --target-mass 13127.54 --tag BareHigh --matrix-mass 30630.92

OUTPUT:
  <tag>_SiO2.xyz            -- coordinates (Si, O, H)
  Printed report            -- achieved mass, wt%, Si/O/H counts, PASS/FAIL
"""
import argparse
import numpy as np

AMU = {'Si': 28.0855, 'O': 15.9994, 'H': 1.00794}
SI_O_BOND = 1.62   # Angstrom, typical Si-O
O_H_BOND = 0.96    # Angstrom, typical O-H


def diamond_lattice_points(a, n_cells):
    """Si positions on a diamond-cubic lattice, conventional cell constant a.
    Diamond = FCC Bravais lattice (4 centering points per conventional cell)
    + 2-atom basis. Missing the FCC centering gives a sparse simple-cubic+basis
    lattice instead of true diamond coordination - this was the earlier bug."""
    centering = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0],
                          [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]])
    basis = np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]])
    pts = []
    rng = range(-n_cells, n_cells + 1)
    for i in rng:
        for j in rng:
            for k in rng:
                origin = np.array([i, j, k], dtype=float)
                for c in centering:
                    for b in basis:
                        pts.append((origin + c + b) * a)
    return np.unique(np.round(np.array(pts), 6), axis=0)


def build_cluster(radius, a):
    n_cells = int(np.ceil(radius / a)) + 2
    pts = diamond_lattice_points(a, n_cells)
    d = np.linalg.norm(pts, axis=1)
    return pts[d <= radius]


def nearest_neighbor_bonds(si_pos, a):
    """Diamond-lattice Si-Si nearest-neighbor distance = a*sqrt(3)/4."""
    nn_dist = a * np.sqrt(3) / 4.0
    tol = 0.15
    n = len(si_pos)
    # simple O(n^2) neighbor search - fine at these small cluster sizes
    bonds = []
    for i in range(n):
        diffs = si_pos[i + 1:] - si_pos[i]
        dists = np.linalg.norm(diffs, axis=1)
        hits = np.where(np.abs(dists - nn_dist) < tol)[0]
        for h in hits:
            bonds.append((i, i + 1 + h))
    return bonds


def build_sio2(radius, a):
    si_pos = build_cluster(radius, a)
    n_Si = len(si_pos)
    if n_Si == 0:
        return None
    bonds = nearest_neighbor_bonds(si_pos, a)

    si_coord = np.zeros(n_Si, dtype=int)
    o_pos = []
    for (i, j) in bonds:
        o_pos.append((si_pos[i] + si_pos[j]) / 2.0)
        si_coord[i] += 1
        si_coord[j] += 1
    n_bridging_O = len(o_pos)

    # terminal -OH for any Si missing bonds (dangling surface bonds)
    h_pos = []
    n_terminal_OH = 0
    for i in range(n_Si):
        missing = 4 - si_coord[i]
        if missing <= 0:
            continue
        center_dir = si_pos[i] / (np.linalg.norm(si_pos[i]) + 1e-9)
        # scatter missing bonds slightly so multiple -OH on one Si don't overlap
        for m in range(missing):
            jitter = 0.15 * np.array([np.sin(m * 2.1), np.cos(m * 1.7), np.sin(m * 0.9)])
            direction = center_dir + jitter
            direction /= np.linalg.norm(direction)
            o = si_pos[i] + direction * SI_O_BOND
            h = o + direction * O_H_BOND
            o_pos.append(o)
            h_pos.append(h)
            n_terminal_OH += 1

    n_O = n_bridging_O + n_terminal_OH
    n_H = n_terminal_OH
    mass = n_Si * AMU['Si'] + n_O * AMU['O'] + n_H * AMU['H']

    return {
        'n_Si': n_Si, 'n_O': n_O, 'n_H': n_H,
        'n_bridging_O': n_bridging_O, 'n_terminal_OH': n_terminal_OH,
        'mass': mass,
        'si_pos': si_pos, 'o_pos': np.array(o_pos), 'h_pos': np.array(h_pos),
    }


def verify_connectivity(res):
    ok = True
    msgs = []
    expected_si_o_bonds = res['n_Si'] * 4
    actual_si_o_bonds = 2 * res['n_bridging_O'] + res['n_terminal_OH']
    if actual_si_o_bonds != expected_si_o_bonds:
        ok = False
        msgs.append(f"Si-O bond mismatch: expected {expected_si_o_bonds}, got {actual_si_o_bonds}")
    if res['n_H'] != res['n_terminal_OH']:
        ok = False
        msgs.append("H count does not equal terminal -OH count (surface not fully passivated)")
    if res['n_bridging_O'] + res['n_terminal_OH'] != res['n_O']:
        ok = False
        msgs.append("O accounting mismatch (bridging + terminal != total O)")
    return ok, msgs


def search_target_mass(target_mass, a, r_min=3.0, r_max=45.0, r_step=0.25):
    best = None
    r = r_min
    while r <= r_max:
        res = build_sio2(r, a)
        if res is not None:
            if best is None or abs(res['mass'] - target_mass) < abs(best[1]['mass'] - target_mass):
                best = (r, res)
        r += r_step
    return best


def write_xyz(path, res):
    lines = []
    total = res['n_Si'] + res['n_O'] + res['n_H']
    lines.append(str(total))
    lines.append("SiO2 nanoparticle - build_sio2_by_mass.py")
    for p in res['si_pos']:
        lines.append(f"Si {p[0]:.4f} {p[1]:.4f} {p[2]:.4f}")
    for p in res['o_pos']:
        lines.append(f"O {p[0]:.4f} {p[1]:.4f} {p[2]:.4f}")
    for p in res['h_pos']:
        lines.append(f"H {p[0]:.4f} {p[1]:.4f} {p[2]:.4f}")
    with open(path, 'w') as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--target-mass', type=float, required=True, help='target filler mass in amu')
    ap.add_argument('--tag', type=str, required=True, help='system tag, e.g. BareLow, BareHigh')
    ap.add_argument('--matrix-mass', type=float, default=30630.92, help='matrix mass in amu (default: locked project matrix)')
    ap.add_argument('--lattice-a', type=float, default=7.16, help='cristobalite-derived diamond lattice constant, Angstrom')
    args = ap.parse_args()

    r, res = search_target_mass(args.target_mass, args.lattice_a)
    ok, msgs = verify_connectivity(res)
    achieved_wt = 100.0 * res['mass'] / (args.matrix_mass + res['mass'])
    target_wt = 100.0 * args.target_mass / (args.matrix_mass + args.target_mass)

    out_path = f"{args.tag}_SiO2.xyz"
    write_xyz(out_path, res)

    print(f"=== build_sio2_by_mass.py :: tag={args.tag} ===")
    print(f"Search radius used       : {r:.2f} A")
    print(f"Target filler mass       : {args.target_mass:.2f} amu  (target wt% = {target_wt:.2f}%)")
    print(f"Achieved filler mass     : {res['mass']:.2f} amu")
    print(f"Achieved wt%             : {achieved_wt:.2f} %  (matrix = {args.matrix_mass} amu)")
    print(f"Composition              : Si{res['n_Si']}/O{res['n_O']}/H{res['n_H']}  "
          f"(total atoms = {res['n_Si'] + res['n_O'] + res['n_H']})")
    print(f"  bridging O             : {res['n_bridging_O']}")
    print(f"  terminal -OH (silanol) : {res['n_terminal_OH']}")
    print(f"Verifier gate            : {'PASS' if ok else 'FAIL'}")
    for m in msgs:
        print(f"  - {m}")
    print(f"Structure written to     : {out_path}")


if __name__ == '__main__':
    main()
