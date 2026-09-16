#!/usr/bin/env python3
"""
hbond_blocks.py - block-averaged version of hbond_angle_occupancy_lifetime.py
Same geometric criterion (H...O < cutoff AND N-H...O angle > cutoff), but
splits the trajectory into N contiguous blocks and reports mean +/- std
across blocks, matching the 5-block protocol used for CN and D_FSI.
"""
import argparse
import numpy as np
from schrodinger.application.desmond.packages import topo, traj


def _get(obj, name, *a):
    x = getattr(obj, name)
    return x(*a) if callable(x) else x


def classify(cms):
    lab = {}
    for m in cms.molecule:
        atoms = list(m.atom)
        has_si = any(a.element == 'Si' for a in atoms)
        has_s = any(a.element == 'S' for a in atoms)
        t = 'filler' if has_si else ('fsi' if has_s else 'ptmc')
        for a in atoms:
            lab[a.index] = t
    return lab


def runs_of_true(seq):
    out = []
    run = 0
    for v in seq:
        if v:
            run += 1
        else:
            if run:
                out.append(run)
            run = 0
    if run:
        out.append(run)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cms_file')
    ap.add_argument('trj_dir')
    ap.add_argument('--tag', required=True)
    ap.add_argument('--dist-cutoff', type=float, default=2.5)
    ap.add_argument('--angle-cutoff', type=float, default=130.0)
    ap.add_argument('--max-frames', type=int, default=3000)
    ap.add_argument('--n-blocks', type=int, default=5)
    args = ap.parse_args()

    _msys, cms = topo.read_cms(args.cms_file)
    lab = classify(cms)

    donor = []
    for a in cms.atom:
        if a.element == 'H' and lab[a.index] == 'filler':
            for nbr in a.bonded_atoms:
                if nbr.element == 'N':
                    donor.append((a.index, nbr.index))
                    break
    acc = [a.index for a in cms.atom if a.element == 'O' and lab[a.index] == 'fsi']
    if not donor or not acc:
        print("ERROR: empty donor/acceptor selection")
        return

    h_g = np.array(topo.aids2gids(cms, [p[0] for p in donor], include_pseudoatoms=False))
    n_g = np.array(topo.aids2gids(cms, [p[1] for p in donor], include_pseudoatoms=False))
    o_g = np.array(topo.aids2gids(cms, acc, include_pseudoatoms=False))

    frames = traj.read_traj(args.trj_dir)
    nf = len(frames)
    step = max(1, nf // args.max_frames)
    use = list(range(0, nf, step))
    nd = len(h_g)
    state = np.zeros((len(use), nd), dtype=bool)
    times = []

    for fi, idx in enumerate(use):
        fr = frames[idx]
        pos = np.asarray(_get(fr, 'pos'))
        box = np.diag(np.asarray(_get(fr, 'box')))
        times.append(_get(fr, 'time'))
        hp, np_, op = pos[h_g], pos[n_g], pos[o_g]
        d = hp[:, None, :] - op[None, :, :]
        d -= box * np.round(d / box)
        dist = np.linalg.norm(d, axis=2)
        v1 = np_ - hp
        v1 /= (np.linalg.norm(v1, axis=1, keepdims=True) + 1e-9)
        di, ai = np.where(dist < args.dist_cutoff)
        if len(di):
            v2 = -d[di, ai]
            v2 /= (np.linalg.norm(v2, axis=1, keepdims=True) + 1e-9)
            ang = np.degrees(np.arccos(np.clip(np.einsum('ij,ij->i', v1[di], v2), -1, 1)))
            for dd in di[ang > args.angle_cutoff]:
                state[fi, dd] = True

    times = np.array(times)
    dt = times[1] - times[0] if len(times) > 1 else 1.0
    nfr = len(use)
    bsz = nfr // args.n_blocks

    occ_blocks, life_blocks = [], []
    for b in range(args.n_blocks):
        s = state[b * bsz:(b + 1) * bsz]
        occ_blocks.append(s.mean() * 100.0)
        rl = []
        for dd in range(nd):
            rl.extend(runs_of_true(s[:, dd]))
        life_blocks.append(np.mean(rl) * dt if rl else 0.0)

    occ_blocks = np.array(occ_blocks)
    life_blocks = np.array(life_blocks)

    print("=" * 62)
    print("%s  |  %d donors, %d frames, %d blocks of %d frames (%.1f ns each)"
          % (args.tag, nd, nfr, args.n_blocks, bsz, bsz * dt / 1000.0))
    print("  criterion: H...O < %.2f A AND N-H...O angle > %.0f deg"
          % (args.dist_cutoff, args.angle_cutoff))
    print("  per-block occupancy (%%): " + ", ".join("%.2f" % v for v in occ_blocks))
    print("  per-block lifetime (ps): " + ", ".join("%.2f" % v for v in life_blocks))
    print("  >> OCCUPANCY = %.2f +/- %.2f %%" % (occ_blocks.mean(), occ_blocks.std(ddof=1)))
    print("  >> LIFETIME  = %.2f +/- %.2f ps" % (life_blocks.mean(), life_blocks.std(ddof=1)))
    print("=" * 62)


if __name__ == '__main__':
    main()
