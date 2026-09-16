#!/usr/bin/env python3
"""
hbond_angle_occupancy_lifetime.py
PaperATS R2 revision - R1-C0 (general rigor comment): "adopting consistent
distance definitions and including angular, occupancy, and lifetime criteria."

Applies a standard geometric H-bond definition to the urea N-H...FSI-O
interaction (the paper's central claim):
  - Distance criterion: H...O < --dist-cutoff (default 2.5 A)
  - Angle criterion: N-H...O angle > --angle-cutoff (default 130 deg)
  - A frame counts as "bonded" for a given donor H if AT LEAST ONE FSI
    oxygen simultaneously satisfies BOTH criteria.

Reports, per donor H (i.e. per urea N-H group):
  - Occupancy: % of frames where the donor is H-bonded (by the combined
    criterion above) to any FSI oxygen.
  - Lifetime: mean continuous-run length (ps) of the bonded state,
    i.e. how long a bond persists once formed before breaking, averaged
    over all bonded runs and all donors. This tracks the BONDED STATE
    (donor bonded to any acceptor), not continuity to one specific
    acceptor - a standard, literature-common simplification.

USAGE:
  $SCHRODINGER/run python3 hbond_angle_occupancy_lifetime.py \\
      <system-out.cms> <system_trj> --tag FuncLow_MQ \\
      --dist-cutoff 2.5 --angle-cutoff 130

OUTPUT: printed summary (occupancy %, mean lifetime ps, n donors, n frames).
No new MD required - reuses existing trajectories.
"""
import argparse
import numpy as np
from schrodinger.application.desmond.packages import topo, traj


def _get(obj, name, *a):
    x = getattr(obj, name)
    return x(*a) if callable(x) else x


def classify_molecules(cms):
    label = {}
    for m in cms.molecule:
        atoms = list(m.atom)
        has_si = any(a.element == 'Si' for a in atoms)
        has_s = any(a.element == 'S' for a in atoms)
        tag = 'filler' if has_si else ('fsi' if has_s else 'ptmc')
        for a in atoms:
            label[a.index] = tag
    return label


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cms_file')
    ap.add_argument('trj_dir')
    ap.add_argument('--tag', required=True)
    ap.add_argument('--dist-cutoff', type=float, default=2.5, help='H...O distance cutoff, Angstrom')
    ap.add_argument('--angle-cutoff', type=float, default=130.0, help='N-H...O angle cutoff, degrees (linear=180)')
    ap.add_argument('--max-frames', type=int, default=3000)
    args = ap.parse_args()

    _msys, cms = topo.read_cms(args.cms_file)
    label = classify_molecules(cms)

    # donor H atoms: H in filler molecule bonded to N (urea N-H, both
    # internal N-H and terminal NH2), paired with their bonded N.
    donor_pairs = []  # list of (h_aid, n_aid)
    for a in cms.atom:
        if a.element == 'H' and label[a.index] == 'filler':
            for nbr in a.bonded_atoms:
                if nbr.element == 'N':
                    donor_pairs.append((a.index, nbr.index))
                    break

    acceptor_aids = [a.index for a in cms.atom if a.element == 'O' and label[a.index] == 'fsi']

    print(f"Donor N-H pairs found: {len(donor_pairs)}")
    print(f"Acceptor O atoms found: {len(acceptor_aids)}")
    if not donor_pairs or not acceptor_aids:
        print("ERROR: empty donor or acceptor selection - check this system has urea groups / FSI.")
        return

    h_aids = [p[0] for p in donor_pairs]
    n_aids = [p[1] for p in donor_pairs]
    h_gids = np.array(topo.aids2gids(cms, h_aids, include_pseudoatoms=False))
    n_gids = np.array(topo.aids2gids(cms, n_aids, include_pseudoatoms=False))
    o_gids = np.array(topo.aids2gids(cms, acceptor_aids, include_pseudoatoms=False))

    frames = traj.read_traj(args.trj_dir)
    nf = len(frames)
    step = max(1, nf // args.max_frames)
    use = list(range(0, nf, step))
    print(f"Using {len(use)} of {nf} frames (stride {step})")

    n_donors = len(h_gids)
    bonded_state = np.zeros((len(use), n_donors), dtype=bool)
    times = []

    for fi, idx in enumerate(use):
        fr = frames[idx]
        pos = np.asarray(_get(fr, 'pos'))
        box = np.diag(np.asarray(_get(fr, 'box')))
        t = _get(fr, 'time')
        times.append(t)

        h_pos = pos[h_gids]      # (n_donors, 3)
        n_pos = pos[n_gids]      # (n_donors, 3)
        o_pos = pos[o_gids]      # (n_acceptors, 3)

        # H...O displacement with minimum image
        diff_ho = h_pos[:, None, :] - o_pos[None, :, :]
        diff_ho -= box * np.round(diff_ho / box)
        dist_ho = np.linalg.norm(diff_ho, axis=2)  # (n_donors, n_acceptors)

        within_dist = dist_ho < args.dist_cutoff

        # N-H vector and H...O vector for angle N-H...O
        diff_nh = n_pos - h_pos  # (n_donors, 3), vector H->N (we need H->N and H->O for angle at H)
        # angle at H between H->N and H->O
        v1 = diff_nh  # H->N
        v1_norm = v1 / (np.linalg.norm(v1, axis=1, keepdims=True) + 1e-9)

        any_bonded = np.zeros(n_donors, dtype=bool)
        donor_idx, acc_idx = np.where(within_dist)
        if len(donor_idx) > 0:
            v2 = -diff_ho[donor_idx, acc_idx]  # H->O vector (positive direction)
            v2_norm = v2 / (np.linalg.norm(v2, axis=1, keepdims=True) + 1e-9)
            cos_angle = np.einsum('ij,ij->i', v1_norm[donor_idx], v2_norm)
            angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
            # N-H...O angle: angle at H between H-N and H-O bonds; a "linear"
            # H-bond has H-N and H-O roughly opposite (angle near 180 deg
            # between vector H->N and vector H->O means N-H...O is linear)
            valid = angle_deg > args.angle_cutoff
            for d in donor_idx[valid]:
                any_bonded[d] = True

        bonded_state[fi] = any_bonded

    times = np.array(times)
    occupancy_per_donor = bonded_state.mean(axis=0) * 100.0
    print(f"\n=== {args.tag}: H-bond occupancy (distance<{args.dist_cutoff}A AND angle>{args.angle_cutoff}deg) ===")
    print(f"Mean occupancy across {n_donors} donors: {occupancy_per_donor.mean():.2f}% "
          f"(range {occupancy_per_donor.min():.2f}-{occupancy_per_donor.max():.2f}%)")

    # lifetime: mean continuous-run length (in frames -> ps) of bonded=True runs, across all donors
    dt = times[1] - times[0] if len(times) > 1 else 1.0
    all_run_lengths = []
    for d in range(n_donors):
        seq = bonded_state[:, d]
        run = 0
        for v in seq:
            if v:
                run += 1
            else:
                if run > 0:
                    all_run_lengths.append(run)
                run = 0
        if run > 0:
            all_run_lengths.append(run)

    if all_run_lengths:
        mean_lifetime_ps = np.mean(all_run_lengths) * dt
        print(f"Mean H-bond lifetime (continuous bonded run): {mean_lifetime_ps:.2f} ps "
              f"(n={len(all_run_lengths)} bonding events, frame spacing={dt:.2f}ps)")
    else:
        print("No bonding events found at this distance/angle criterion.")

    print("===============================================================")


if __name__ == '__main__':
    main()
