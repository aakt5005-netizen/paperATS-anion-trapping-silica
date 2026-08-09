"""
ecc_scale.py — Apply ECC charge scaling to ionic species in a Desmond .cms file.

Scales the ffio-block charges (what MD actually reads) of cation and anion
component-templates by a factor (default 0.8), leaving solvent/polymer untouched.
Also scales atom-level charges for consistency. Verifies at the ffio level and
flags zero-charge sites (atom-typing failures).

Tested with Schrodinger release 2025-3.

USAGE
-----
  $SCHRODINGER/run python3 ecc_scale.py <in.cms> <out.cms> \
        --cation Li \
        --anion-marker Cl \
        [--factor 0.8] [--exclude N,H]

  --cation        element symbol of the single-atom cation CT (e.g. Li)
  --anion-marker  element that uniquely marks the anion CT (e.g. Cl for ClO4-,
                  B for bis-oxalato-borate). The anion CT is the CT that
                  contains this element and none of the --exclude elements.
  --factor        scaling factor (default 0.8)
  --exclude       comma-separated elements that must NOT be in the anion CT,
                  used to avoid matching polymer/solvent (default: N,H)

EXAMPLES
--------
  LiClO4 :  --cation Li --anion-marker Cl
  LiBOB  :  --cation Li --anion-marker B
  LiPF6  :  --cation Li --anion-marker P

NOTES
-----
- ffio sites are a template applied to all copies, so "1 cation CT, 1 anion CT
  scaled" is correct even with many ions.
- Output is written to a NEW file; the original is never modified.
- After scaling, do NOT re-assign/re-compute the force field on the output in
  Maestro, or the scaled charges will be overwritten back to full.
"""
import sys
import argparse
from schrodinger.application.desmond.packages import topo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("infile")
    ap.add_argument("outfile")
    ap.add_argument("--cation", default="Li")
    ap.add_argument("--anion-marker", required=True,
                    help="element uniquely marking the anion CT (Cl, B, P, ...)")
    ap.add_argument("--factor", type=float, default=0.8)
    ap.add_argument("--exclude", default="N,H",
                    help="comma-sep elements that must NOT be in anion CT")
    args = ap.parse_args()

    excl = set(e.strip() for e in args.exclude.split(",") if e.strip())
    SCALE = args.factor

    msys, cms = topo.read_cms(args.infile)

    q_before = sum(a.partial_charge for a in cms.atom)

    n_cat = n_an = 0
    cat_q_before = an_net_before = None

    for ct in cms.comp_ct:
        elems = [a.element for a in ct.atom]
        try:
            sites = list(ct.ffio.site)
        except Exception:
            continue

        is_cation = (set(elems) == {args.cation})
        is_anion = (args.anion_marker in elems
                    and not (excl & set(elems)))

        if is_cation:
            if cat_q_before is None:
                cat_q_before = sites[0].charge
            for s in sites:
                s.charge *= SCALE
            n_cat += 1
        elif is_anion:
            if an_net_before is None:
                an_net_before = sum(s.charge for s in sites)
            for s in sites:
                s.charge *= SCALE
            n_an += 1

    # scale atom-level charges too (for analysis-tool consistency)
    for mol in cms.molecule:
        elems = sorted(set(a.element for a in mol.atom))
        is_cat = (set(a.element for a in mol.atom) == {args.cation})
        is_an = (args.anion_marker in [a.element for a in mol.atom]
                 and not (excl & set(a.element for a in mol.atom)))
        if is_cat or is_an:
            for a in mol.atom:
                a.partial_charge *= SCALE

    print("=" * 55)
    print(f"ECC scaling  factor={SCALE}")
    print(f"  cation '{args.cation}'  CTs scaled: {n_cat}")
    print(f"  anion  (marker '{args.anion_marker}') CTs scaled: {n_an}")
    if cat_q_before is not None:
        print(f"  cation charge : {cat_q_before:.4f} -> {cat_q_before*SCALE:.4f}")
    if an_net_before is not None:
        print(f"  anion net/unit: {an_net_before:.4f} -> {an_net_before*SCALE:.4f}")

    # ---- verify at ffio level + zero-charge check ----
    zero_sites = []
    for ct in cms.comp_ct:
        elems = [a.element for a in ct.atom]
        try:
            sites = list(ct.ffio.site)
        except Exception:
            continue
        if (args.anion_marker in elems and not (excl & set(elems))) \
                or (set(elems) == {args.cation}):
            for i, s in enumerate(sites):
                if abs(s.charge) < 1e-6:
                    zero_sites.append((elems, i))

    q_after = sum(a.partial_charge for a in cms.atom)
    print(f"  total charge (atom-level): {q_before:.4f} -> {q_after:.4f}")
    if zero_sites:
        print(f"  !! WARNING zero-charge ion sites (atom-typing failure?): {zero_sites}")

    ok = (n_cat >= 1 and n_an >= 1 and not zero_sites)
    print("-" * 55)
    print(f"  >>> {'OK' if ok else 'CHECK FAILED'} <<<")
    print("=" * 55)

    if not ok:
        sys.exit("Refusing to save: verify --cation / --anion-marker / --exclude.")

    cms.write(args.outfile)
    print(f"Saved -> {args.outfile}")
    print("Verify independently at ffio level before running MD.")


if __name__ == "__main__":
    main()
