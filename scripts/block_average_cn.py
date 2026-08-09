#!/usr/bin/env python3
"""
block_average_cn.py  (v2 - multi-chunk capable)

Block-averaged Li-anion(FSI) coordination number, for error bars without
new MD. This is the ORIGINAL 17-Jul script (classify_molecules,
gids_from_aids, compute_cn_at_cutoff, block-splitting logic, output format
- all byte-for-byte unchanged) EXTENDED with ONE capability: it now accepts
multiple trajectory CHUNKS (--chunks cms trj cms trj ...) and stitches them
in chronological order before block-splitting, instead of a single fixed
cms/trj pair.

WHY THIS CHANGE (re-verification, not a rewrite): the original single-
chunk-only version was invoked (17 Jul) with MISMATCHED windows across
systems:
  - Reference used ONLY Prod_ext2 (Baseline's 3rd 50ns chunk, ns~100-150)
  - FuncLow    used ONLY Prod_ext1 (90ns, missing its own first 10ns)
  - BareLow/BareHigh/FuncHigh used their full single 100ns Prod (fine)
This was never caught for the CN comparison itself (this script's output -
the actual winner-selection statistic) even though the same-class bug was
caught and fixed for B.1/B.2 on 18 Jul. This version restores exact parity
with the original CN computation logic while fixing ONLY the windowing
input, so Reference and FuncLow can be run on their confirmed-consistent
0-100ns window (Prod+ext1 stitched) like everyone else.

USAGE:
  Single chunk (BareLow / BareHigh / FuncHigh - same result as original):
    $SCHRODINGER/run python3 block_average_cn.py \
        --chunks out.cms trj_dir \
        --tag BareLow --li-cutoff 2.55 --n-blocks 5

  Multi-chunk (Reference, FuncLow - chronological order, Prod then ext1):
    $SCHRODINGER/run python3 block_average_cn.py \
        --chunks Prod-out.cms Prod_trj Prod_ext1-out.cms Prod_ext1_trj \
        --tag Reference --li-cutoff 2.55 --n-blocks 5
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


def gids_from_aids(cms, aids):
    if not aids:
        return np.array([], dtype=int)
    return np.array(topo.aids2gids(cms, aids, include_pseudoatoms=False))


def compute_cn_at_cutoff(pos_li_frames, pos_anion_frames, boxes, cutoff, dr=0.1):
    """RDF-based CN via shell integration up to cutoff - UNCHANGED from
    the original block_average_cn.py (17 Jul)."""
    rmax = cutoff + 2.0
    nb = int(rmax / dr)
    edges = np.linspace(0, rmax, nb + 1)
    shell_vol = 4 * np.pi / 3 * (edges[1:] ** 3 - edges[:-1] ** 3)
    hist = np.zeros(nb)
    n_li_total = n_an_total = 0
    vol_total = 0.0
    nframes = len(pos_li_frames)

    for pli, pan, box in zip(pos_li_frames, pos_anion_frames, boxes):
        if len(pli) == 0 or len(pan) == 0:
            continue
        diff = pli[:, None, :] - pan[None, :, :]
        diff -= box * np.round(diff / box)
        d = np.linalg.norm(diff, axis=2)
        h, _ = np.histogram(d.ravel(), bins=edges)
        hist += h
        n_li_total += len(pli)
        n_an_total += len(pan)
        vol_total += np.prod(box)

    if nframes == 0 or n_li_total == 0:
        return np.nan

    mean_li = n_li_total / nframes
    mean_an = n_an_total / nframes
    mean_vol = vol_total / nframes
    rho_an = mean_an / mean_vol
    norm = mean_li * nframes * shell_vol * rho_an
    norm[norm == 0] = np.nan
    gr = np.nan_to_num(hist / norm)
    cn_cum = np.cumsum(gr * shell_vol * rho_an)

    rmid = 0.5 * (edges[1:] + edges[:-1])
    idx = np.searchsorted(rmid, cutoff)
    idx = min(idx, len(cn_cum) - 1)
    return cn_cum[idx]


def load_chunk(cms_file, trj_dir, li_gid_ref, fsi_gid_ref):
    """Load one chunk's frames. If a reference (first-chunk) atom-count is
    given, verify this chunk matches it - guards against accidentally
    pairing a chunk from the wrong system."""
    _msys, cms = topo.read_cms(cms_file)
    frames = traj.read_traj(trj_dir)
    label = classify_molecules(cms)
    li_aids = [a.index for a in cms.atom if a.element == 'Li']
    fsi_f = [a.index for a in cms.atom if a.element == 'F' and label[a.index] == 'fsi']
    fsi_o = [a.index for a in cms.atom if a.element == 'O' and label[a.index] == 'fsi']
    fsi_n = [a.index for a in cms.atom if a.element == 'N' and label[a.index] == 'fsi']
    li_gid = gids_from_aids(cms, li_aids)
    fsi_gid = gids_from_aids(cms, fsi_f + fsi_o + fsi_n)

    if li_gid_ref is not None and (len(li_gid) != len(li_gid_ref) or len(fsi_gid) != len(fsi_gid_ref)):
        raise SystemExit(f"ERROR: atom-count mismatch in {cms_file} vs first chunk "
                          f"(Li {len(li_gid)} vs {len(li_gid_ref)}, "
                          f"FSI {len(fsi_gid)} vs {len(fsi_gid_ref)}) - wrong chunk/system paired?")

    pos_li, pos_an, boxes, times = [], [], [], []
    for fr in frames:
        p = np.asarray(_get(fr, 'pos'))
        pos_li.append(p[li_gid])
        pos_an.append(p[fsi_gid])
        boxes.append(np.diag(np.asarray(_get(fr, 'box'))))
        times.append(_get(fr, 'time'))
    return times, pos_li, pos_an, boxes, li_gid, fsi_gid


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--chunks', nargs='+', required=True,
                     help='alternating cms trj cms trj ... in chronological order')
    ap.add_argument('--tag', required=True)
    ap.add_argument('--li-cutoff', type=float, required=True,
                     help="this system's OWN first-minimum r (A) - REUSE the exact value "
                          "originally locked for this system (see PaperATS_Phase1_CN_comparison_v1_17Jul.csv), "
                          "do not re-detect - we are isolating the window as the only changed variable")
    ap.add_argument('--n-blocks', type=int, default=5)
    args = ap.parse_args()

    if len(args.chunks) % 2 != 0:
        raise SystemExit("ERROR: --chunks must be pairs of <cms> <trj_dir>")
    pairs = list(zip(args.chunks[0::2], args.chunks[1::2]))

    all_times, all_pos_li, all_pos_an, all_boxes = [], [], [], []
    li_gid_ref = fsi_gid_ref = None
    t_offset = 0.0
    for idx, (cms_f, trj_d) in enumerate(pairs):
        t, pli, pan, box, li_gid, fsi_gid = load_chunk(cms_f, trj_d, li_gid_ref, fsi_gid_ref)
        if li_gid_ref is None:
            li_gid_ref, fsi_gid_ref = li_gid, fsi_gid
        if idx > 0:
            # drop duplicate boundary frame (chunk N+1 frame 0 == chunk N's
            # last frame - same restart convention as li_msd_multichunk.py)
            t, pli, pan, box = t[1:], pli[1:], pan[1:], box[1:]
        t_glob = [ti + t_offset for ti in t]
        all_times.extend(t_glob)
        all_pos_li.extend(pli)
        all_pos_an.extend(pan)
        all_boxes.extend(box)
        t_offset = t_glob[-1]

    F = len(all_times)
    print(f"{args.tag}: stitched {len(pairs)} chunk(s) -> {F} frames, "
          f"0 to {all_times[-1]/1000:.1f} ns")

    block_edges = np.linspace(0, F, args.n_blocks + 1).astype(int)
    cn_values = []
    for b in range(args.n_blocks):
        lo, hi = block_edges[b], block_edges[b + 1]
        cn = compute_cn_at_cutoff(all_pos_li[lo:hi], all_pos_an[lo:hi], all_boxes[lo:hi], args.li_cutoff)
        cn_values.append(cn)
        print(f"  block {b+1}/{args.n_blocks} (frames {lo}-{hi}): CN = {cn:.3f}")

    cn_arr = np.array(cn_values)
    mean = cn_arr.mean()
    std = cn_arr.std(ddof=1)
    sem = std / np.sqrt(len(cn_arr))
    print(f"=== {args.tag}: CN(Li-anion, r<={args.li_cutoff}A) = {mean:.3f} +/- {std:.3f} (std), SEM={sem:.3f}, n_blocks={args.n_blocks} ===")


if __name__ == '__main__':
    main()
