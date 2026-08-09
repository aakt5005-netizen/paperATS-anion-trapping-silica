"""
independent_verify.py — Independent ffio-level re-check of a scaled .cms file.

Per ECC_charge_scaling_guide.md: "VERIFY after writing by re-reading the file and
inspecting the ffio-level charges" and "verify independently at ffio level before
running MD." This is a SEPARATE script from ecc_scale.py's own internal check -
independent verification, not trusting the same code path that did the scaling.

USAGE:
  $SCHRODINGER/run python3 independent_verify.py PTMC_LiFSI_scaled.cms \
      --cation Li --anion-marker S --expected-factor 0.80 --exclude ""
"""
import argparse
import sys
from schrodinger.application.desmond.packages import topo

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("infile")
    ap.add_argument("--cation", default="Li")
    ap.add_argument("--anion-marker", required=True)
    ap.add_argument("--expected-factor", type=float, default=0.80)
    ap.add_argument("--exclude", default="")
    args = ap.parse_args()

    excl = set(e.strip() for e in args.exclude.split(",") if e.strip())
    tol = 0.02

    _, cms = topo.read_cms(args.infile)

    cat_charges, an_nets = [], []
    zero_sites = []

    for ct in cms.comp_ct:
        elems = [a.element for a in ct.atom]
        try:
            sites = list(ct.ffio.site)
        except Exception:
            continue
        is_cation = (set(elems) == {args.cation})
        is_anion = (args.anion_marker in elems and not (excl & set(elems)))
        if is_cation:
            for s in sites:
                cat_charges.append(s.charge)
                if abs(s.charge) < 1e-6:
                    zero_sites.append(("cation", s.charge))
        elif is_anion:
            an_nets.append(sum(s.charge for s in sites))
            for s in sites:
                if abs(s.charge) < 1e-6:
                    zero_sites.append(("anion", s.charge))

    expected_cat = 1.0 * args.expected_factor
    expected_an = -1.0 * args.expected_factor

    ok = True
    if not cat_charges:
        print(f"[FAIL] No cation CT found matching element '{args.cation}'")
        ok = False
    else:
        bad = [c for c in cat_charges if abs(c - expected_cat) > tol]
        print(f"[{'PASS' if not bad else 'FAIL'}] Cation ffio charge(s): {cat_charges} "
              f"(expected {expected_cat:.3f})")
        ok &= not bad

    if not an_nets:
        print(f"[FAIL] No anion CT found matching marker '{args.anion_marker}'")
        ok = False
    else:
        bad = [n for n in an_nets if abs(n - expected_an) > tol]
        print(f"[{'PASS' if not bad else 'FAIL'}] Anion ffio net charge(s): {an_nets} "
              f"(expected {expected_an:.3f})")
        ok &= not bad

    if zero_sites:
        print(f"[FAIL] Zero-charge ffio sites found (atom-typing failure): {zero_sites}")
        ok = False
    else:
        print("[PASS] No zero-charge ion sites")

    print(f"\n=== INDEPENDENT VERIFICATION: {'PASS' if ok else 'FAIL'} ===")
    if not ok:
        sys.exit(1)

if __name__ == "__main__":
    main()
