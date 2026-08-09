"""
Paper1_PTMC_Fillers - Multi-chunk Li+ MSD (COM-drift corrected)
Stitches sequential production chunks (e.g. Prod, Prod_ext1, Prod_ext2...)
into one continuous unwrapped trajectory, giving much longer accessible
lag times than any single chunk allows.

WHY THIS WORKS: each chunk was started by copying the previous chunk's
final structure (not a true Desmond .cpt velocity-continuation), so:
- Positions ARE continuous across chunk boundaries (chunk N+1 frame 0 is
  the same atom coordinates as chunk N's last frame)
- Velocities are re-thermalized at each chunk start - irrelevant for a
  position-based MSD, only a negligible transient (system already
  equilibrated at 300K NPT before each restart)
- Chunk N+1's frame 0 duplicates chunk N's last frame - this script
  detects and drops that duplicate automatically before stitching.

USAGE (chunks MUST be given in chronological order):
  $SCHRODINGER/run python3 li_msd_multichunk.py \\
      Paper1_Baseline_Prod-out.cms Paper1_Baseline_Prod_trj \\
      Paper1_Baseline_Prod_ext1-out.cms Paper1_Baseline_Prod_ext1_trj \\
      --tau-start 5 --tau-end 100

GOTCHAS:
- --tau-end must stay within ~half the TOTAL combined trajectory length
  (e.g. for 150ns combined, max useful tau-end ~70-75ns).
- Output: Li_MSD_combined.csv + printed D / alpha summary (same format
  as the original single-chunk li_msd.py).
"""
import sys
import argparse
import numpy as np
from schrodinger.application.desmond.packages import topo, traj


def _get(obj, name, *a):
    x = getattr(obj, name)
    return x(*a) if callable(x) else x


def load_chunk_unwrapped(cms_file, trj_dir, seed_pos=None):
    _msys, cms = topo.read_cms(cms_file)
    frames = traj.read_traj(trj_dir)
    F = len(frames)
    li_gids = topo.aids2gids(cms, cms.select_atom("atom.ele Li"), include_pseudoatoms=False)
    all_aids = list(range(1, cms.atom_total + 1))
    all_gids = topo.aids2gids(cms, all_aids, include_pseudoatoms=False)
    mass = np.zeros(max(all_gids) + 1)
    for a, gd in zip(all_aids, all_gids):
        mass[gd] = cms.atom[a].atomic_weight
    M = mass.sum()

    times = np.array([_get(fr, 'time') for fr in frames], dtype=float)

    p0 = np.asarray(_get(frames[0], 'pos'))
    running = seed_pos.copy() if seed_pos is not None else p0.copy()
    prev = p0.copy()
    li_unw = np.zeros((F, len(li_gids), 3))
    com0 = (mass[:, None] * running).sum(0) / M
    li_unw[0] = running[li_gids] - com0
    for i in range(1, F):
        p = np.asarray(_get(frames[i], 'pos'))
        box = np.diag(np.asarray(_get(frames[i], 'box')))
        d = p - prev
        d -= box * np.round(d / box)
        running += d
        com_i = (mass[:, None] * running).sum(0) / M
        li_unw[i] = running[li_gids] - com_i
        prev = p

    return times, li_unw, running.copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("chunks", nargs="+",
                     help="alternating cms trj cms trj ... in chronological order")
    ap.add_argument("--tau-start", type=float, default=5.0, help="fit window start, ns")
    ap.add_argument("--tau-end", type=float, default=100.0, help="fit window end, ns")
    args = ap.parse_args()

    if len(args.chunks) % 2 != 0:
        sys.exit("ERROR: provide chunks as pairs of <cms> <trj_dir>")

    pairs = list(zip(args.chunks[0::2], args.chunks[1::2]))
    print(f"Stitching {len(pairs)} chunk(s):")
    for c, t in pairs:
        print(f"  {c}  +  {t}")

    all_li_pos, all_times = [], []
    seed = None
    time_offset = 0.0
    n_dropped = 0

    for idx, (cms_file, trj_dir) in enumerate(pairs):
        times, li_unw, seed = load_chunk_unwrapped(cms_file, trj_dir, seed_pos=seed)
        if idx > 0:
            times = times[1:]
            li_unw = li_unw[1:]
            n_dropped += 1
        times_global = times + time_offset
        all_times.append(times_global)
        all_li_pos.append(li_unw)
        time_offset = times_global[-1]

    times_cat = np.concatenate(all_times)
    li_cat = np.concatenate(all_li_pos, axis=0)
    F = len(times_cat)
    print(f"Combined trajectory: {F} frames, {n_dropped} boundary duplicate frame(s) dropped, "
          f"spanning 0 to {times_cat[-1]/1000:.1f} ns")

    dt = times_cat[1] - times_cat[0]
    max_lag_frames = F // 2
    lag_ps = np.arange(1, max_lag_frames) * dt
    msd = np.zeros(len(lag_ps))
    for k in range(1, max_lag_frames):
        disp = li_cat[k:] - li_cat[:-k]
        msd[k - 1] = (disp ** 2).sum(axis=2).mean()

    m = (lag_ps >= args.tau_start * 1000.0) & (lag_ps <= args.tau_end * 1000.0)
    if m.sum() < 5:
        sys.exit("Fit window too small; lower --tau-end (must be within max-lag ~half combined run).")

    lag_s = lag_ps[m] * 1e-12
    slope_lin = np.polyfit(lag_s, msd[m], 1)[0]   # A^2/s
    D = slope_lin / 6.0 * 1e-20                    # -> m^2/s

    logl = np.log(lag_ps[m])
    logm = np.log(msd[m])
    alpha = np.polyfit(logl, logm, 1)[0]

    print("================  RESULT (combined)  ================")
    print(f"Fit window      : {args.tau_start:.0f} - {args.tau_end:.0f} ns")
    print(f"Diffusion coeff : {D:.4e} m^2/s   ({D*1e4:.4e} cm^2/s)")
    print(f"log-log slope a : {alpha:.3f}   (need ~1.0 for diffusive)")
    print(f"MSD at last lag : {msd[m][-1]:.2f} A^2  (RMS {np.sqrt(msd[m][-1]):.2f} A)")
    if alpha < 0.85:
        print("WARNING: alpha < 0.85  ->  SUB-DIFFUSIVE (Li still caged).")
        print("         The D above is an APPARENT value, not a true Fickian D.")
    print("=======================================================")

    with open("Li_MSD_combined.csv", "w") as f:
        for l, v in zip(lag_ps, msd):
            f.write(f"{l},{v}\n")
    print("Saved: Li_MSD_combined.csv (lag_ps, MSD_A2)")


if __name__ == "__main__":
    main()
