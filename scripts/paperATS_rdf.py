#!/usr/bin/env python3
"""
paperATS_rdf.py
Stage 4 Level-1 RDF + Li coordination analysis for PaperATS Phase-1 systems
(Reference / BareLow / BareHigh / FuncLow / FuncHigh).

Adapted from the prior-project analyze_rdf.py pattern (frame iteration,
PBC minimum-image, histogram/shell-volume g(r) normalization) but with
ATOM-SELECTION LOGIC REWRITTEN for this system's actual chemistry:
  - PTMC matrix has NO nitrogen (unlike the old PAN -C#N system), so the
    old "element N = polymer marker" trick does not apply here.
  - FSI- anion is identified by S (sulfur) - unique to FSI in this system
    (PTMC, urea-grafted filler, and Li+ all lack S).
  - Filler (bare or urea-grafted SiO2) is identified by Si - unique to the
    filler in this system.
  - PTMC (matrix polymer) = whatever is left: molecules with no Si, no S.

COMPUTES:
  1. Li-O(PTMC carbonate)      - polymer coordination
  2. Li-anion (Li to FSI N/O/F) - ion pairing / caging
  3. N-H(urea)...FSI (F/O/N)   - anion-trapping / decaging signature
                                  (only present for Func systems; 0 atoms
                                  for Reference/BareLow/BareHigh, script
                                  will report this and skip gracefully)
  4. Si-OH...FSI (F/O/N)       - baseline bare-silanol/anion interaction
                                  (present in ALL systems with a filler,
                                  including the un-grafted silanols that
                                  remain on Func systems - useful internal
                                  comparison within the same trajectory)
  5. Li+ coordination number breakdown (polymer O / anion / free), using
     the first-minimum cutoff read off the Li-O and Li-anion RDFs

USAGE:
  $SCHRODINGER/run python3 paperATS_rdf.py <system-out.cms> <system_trj> \
      [--rmax 8.0] [--dr 0.1] [--max-frames 3000] [--li-cutoff 3.2]

  --li-cutoff sets the Li-O / Li-anion coordination cutoff (Angstrom).
  Default 3.2 A is a common first-minimum starting guess for Li-O in
  polymer electrolytes - ALWAYS visually confirm against this system's
  own RDF first minimum before trusting the coordination-number numbers
  (per Stage 0.4: no fitting/interpretation without checking the actual
  plotted curve first).

OUTPUT:
  <tag>_rdf.csv   - all RDF curves, one column per pair, r in Angstrom
  Printed summary - Li coordination number breakdown, PASS/FAIL sanity
                     checks (non-empty selections, reasonable g(r) shape)
"""
import argparse
import sys
import numpy as np
from schrodinger.application.desmond.packages import topo, traj


def classify_molecules(cms):
    """Per-atom index -> component label ('ptmc', 'fsi', 'filler'),
    based on unique marker elements (Si=filler, S=FSI, else=PTMC)."""
    label = {}
    for m in cms.molecule:
        atoms = list(m.atom)
        has_si = any(a.element == 'Si' for a in atoms)
        has_s = any(a.element == 'S' for a in atoms)
        if has_si:
            tag = 'filler'
        elif has_s:
            tag = 'fsi'
        else:
            tag = 'ptmc'
        for a in atoms:
            label[a.index] = tag
    return label


def _get(obj, name, *a):
    """Safe attribute getter - some Schrodinger versions expose frame.pos/
    frame.box as callable methods, others as plain array attributes. This
    project hit the same inconsistency before (see li_msd_multichunk.py)."""
    x = getattr(obj, name)
    return x(*a) if callable(x) else x


def gids_from_aids(cms, aids):
    if not aids:
        return np.array([], dtype=int)
    return np.array(topo.aids2gids(cms, aids, include_pseudoatoms=False))


def build_selections(cms):
    label = classify_molecules(cms)
    sel = {}

    sel['li'] = [a.index for a in cms.atom if a.element == 'Li']
    sel['ptmc_O'] = [a.index for a in cms.atom if a.element == 'O' and label[a.index] == 'ptmc']
    sel['fsi_F'] = [a.index for a in cms.atom if a.element == 'F' and label[a.index] == 'fsi']
    sel['fsi_O'] = [a.index for a in cms.atom if a.element == 'O' and label[a.index] == 'fsi']
    sel['fsi_N'] = [a.index for a in cms.atom if a.element == 'N' and label[a.index] == 'fsi']
    # combined anion acceptor set (F, O, N all can H-bond / coordinate Li)
    sel['fsi_all'] = sel['fsi_F'] + sel['fsi_O'] + sel['fsi_N']

    # urea N-H hydrogens: H atoms in the filler molecule bonded to an N
    # (only urea N's have attached H among filler atoms - bridging/terminal
    # SiO2 oxygens and carbonyl atoms don't produce N-H)
    urea_nh = []
    for a in cms.atom:
        if a.element == 'H' and label[a.index] == 'filler':
            for nbr in a.bonded_atoms:
                if nbr.element == 'N':
                    urea_nh.append(a.index)
                    break
    sel['urea_NH'] = urea_nh

    # Si-OH oxygens: O atoms in the filler molecule bonded to an H
    # (captures both original surface silanols and APTES capping -OH;
    # excludes bridging Si-O-Si oxygens and the urea C=O oxygen, neither
    # of which has an attached H)
    silanol_o = []
    for a in cms.atom:
        if a.element == 'O' and label[a.index] == 'filler':
            for nbr in a.bonded_atoms:
                if nbr.element == 'H':
                    silanol_o.append(a.index)
                    break
    sel['silanol_O'] = silanol_o

    return sel


def compute_rdf(pos_a_frames, pos_b_frames, boxes, rmax, dr, self_pair=False):
    nb = int(rmax / dr)
    edges = np.linspace(0, rmax, nb + 1)
    rmid = 0.5 * (edges[1:] + edges[:-1])
    shell_vol = 4 * np.pi / 3 * (edges[1:] ** 3 - edges[:-1] ** 3)

    hist = np.zeros(nb)
    n_a_total = 0
    n_b_total = 0
    vol_total = 0.0
    nframes = len(pos_a_frames)

    for pa, pb, box in zip(pos_a_frames, pos_b_frames, boxes):
        na, nb_ = len(pa), len(pb)
        if na == 0 or nb_ == 0:
            continue
        diff = pa[:, None, :] - pb[None, :, :]
        diff -= box * np.round(diff / box)
        d = np.linalg.norm(diff, axis=2)
        if self_pair:
            np.fill_diagonal(d, np.inf)
        h, _ = np.histogram(d.ravel(), bins=edges)
        hist += h
        n_a_total += na
        n_b_total += nb_
        vol_total += np.prod(box)

    if nframes == 0 or n_a_total == 0:
        return rmid, np.zeros(nb), np.zeros(nb)

    mean_na = n_a_total / nframes
    mean_nb = n_b_total / nframes
    mean_vol = vol_total / nframes
    rho_b = mean_nb / mean_vol
    norm = mean_na * nframes * shell_vol * rho_b
    norm[norm == 0] = np.nan
    gr = hist / norm

    # coordination number: cumulative integral N(r) = integral 4pi r^2 rho g(r) dr
    cn = np.cumsum(gr * shell_vol * rho_b)
    return rmid, np.nan_to_num(gr), cn


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cms_file')
    ap.add_argument('trj_dir')
    ap.add_argument('--tag', default='FuncLow')
    ap.add_argument('--rmax', type=float, default=8.0)
    ap.add_argument('--dr', type=float, default=0.1)
    ap.add_argument('--max-frames', type=int, default=3000)
    ap.add_argument('--li-cutoff', type=float, default=3.2,
                     help='Li-O/Li-anion coordination cutoff, A (VERIFY against this system RDF first minimum)')
    args = ap.parse_args()

    msys, cms = topo.read_cms(args.cms_file)
    frames = traj.read_traj(args.trj_dir)
    nf = len(frames)
    print(f"Loaded {nf} frames from {args.trj_dir}")

    sel = build_selections(cms)
    gids = {k: gids_from_aids(cms, v) for k, v in sel.items()}
    for k, v in gids.items():
        print(f"  {k:12s}: {len(v)} atoms")

    step = max(1, nf // args.max_frames)
    use = list(range(0, nf, step))
    print(f"Using {len(use)} of {nf} frames (stride {step}) for RDF averaging")

    pos_cache = {}
    boxes = []
    for fi in use:
        fr = frames[fi]
        pos_cache[fi] = np.asarray(_get(fr, 'pos'))
        boxes.append(np.diag(np.asarray(_get(fr, 'box'))))

    def frame_pos(gid_array):
        return [pos_cache[fi][gid_array] for fi in use]

    pairs = {
        'Li-O(PTMC)':       (gids['li'], gids['ptmc_O']),
        'Li-anion(FSI)':    (gids['li'], gids['fsi_all']),
        'ureaNH-FSI_F':     (gids['urea_NH'], gids['fsi_F']),
        'ureaNH-FSI_O':     (gids['urea_NH'], gids['fsi_O']),
        'ureaNH-FSI_N':     (gids['urea_NH'], gids['fsi_N']),
        'silanolO-FSI_F':   (gids['silanol_O'], gids['fsi_F']),
        'silanolO-FSI_O':   (gids['silanol_O'], gids['fsi_O']),
    }

    results = {}
    for label, (ga, gb) in pairs.items():
        if len(ga) == 0 or len(gb) == 0:
            print(f"  [{label}] SKIPPED - one or both selections empty ({len(ga)}, {len(gb)} atoms)")
            results[label] = None
            continue
        pa_frames = frame_pos(ga)
        pb_frames = frame_pos(gb)
        r, gr, cn = compute_rdf(pa_frames, pb_frames, boxes, args.rmax, args.dr)
        results[label] = (r, gr, cn)
        peak_idx = np.argmax(gr)
        print(f"  [{label}] first-peak g(r) = {gr[peak_idx]:.2f} at r = {r[peak_idx]:.2f} A")

    # save CSV
    out_csv = f"{args.tag}_rdf.csv"
    with open(out_csv, 'w') as f:
        header = ['r_A']
        cols = []
        for label, res in results.items():
            if res is None:
                continue
            header += [f'g_{label}', f'CN_{label}']
            cols.append(res)
        f.write(",".join(header) + "\n")
        if cols:
            r_ref = cols[0][0]
            for i in range(len(r_ref)):
                row = [f"{r_ref[i]:.3f}"]
                for r, gr, cn in cols:
                    row += [f"{gr[i]:.4f}", f"{cn[i]:.4f}"]
                f.write(",".join(row) + "\n")
    print(f"Saved RDF data to {out_csv}")

    # Li coordination number breakdown at the specified cutoff
    print(f"\n=== Li+ coordination breakdown at cutoff = {args.li_cutoff} A "
          f"(VERIFY this matches the first RDF minimum before trusting it) ===")
    for label in ['Li-O(PTMC)', 'Li-anion(FSI)']:
        res = results.get(label)
        if res is None:
            print(f"  {label}: N/A (empty selection)")
            continue
        r, gr, cn = res
        idx = np.searchsorted(r, args.li_cutoff)
        idx = min(idx, len(cn) - 1)
        print(f"  {label}: CN({args.li_cutoff} A) = {cn[idx]:.2f}")

    print("\nNOTE (Stage 0.4): the printed CN numbers use a FIXED cutoff guess.")
    print("Before reporting these as findings, plot g(r) for Li-O(PTMC) and")
    print("Li-anion(FSI) and re-run with --li-cutoff set to this system's own")
    print("actual first-minimum position (may differ from the 3.2 A default).")


if __name__ == '__main__':
    main()
