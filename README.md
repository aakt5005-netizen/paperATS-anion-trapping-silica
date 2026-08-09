# Anion-Trapping Silica — Code and Processed Data

This repository accompanies the manuscript:

**"From Free Molecules to Grafted Fillers: Covalent Urea Sites for Trapping
Anions in Lithium Battery Electrolytes"**
A. K. Tripathi, Ramjas College, University of Delhi

## Contents

### `scripts/`
Python scripts used for system construction, force-field charge scaling,
equilibration checking, and structural/transport analysis, run via the
Schrodinger 2025-3 suite (Desmond/Maestro):

- `build_sio2_by_mass.py` — mass-targeted amorphous SiO2 nanoparticle builder
- `graft_sio2_urea.py` — APTES-amine-urea surface grafting on SiO2, with built-in verifier gate
- `ecc_scale.py` — electronic continuum correction (ECC) charge scaling for Li+/FSI-
- `independent_verify.py` — independent ffio-level verification of scaled charges
- `checkpoint_ene.py` — slope-based NPT equilibration checkpoint
- `li_msd_multichunk.py` — multi-chunk Li+ MSD (mean-squared displacement) analysis
- `paperATS_rdf.py` — radial distribution function and coordination-number analysis
- `block_average_cn.py` — block-averaged Li-FSI coordination number with error bars (multi-chunk capable)

### `processed_data/`
Summary/processed data underlying the manuscript's tables and figures:

- `Table1_PhaseI_Summary.csv` — Phase-1 system summary (coordination number,
  ion-pairing populations, FSI diffusion coefficient) for all five systems
- `Table2_HBond_PeakSummary.csv` — hydrogen-bond RDF peak heights and positions
- `PaperATS_Phase1_CN_comparison_v2_07Aug.csv` — verified block-averaged
  coordination-number comparison across all five systems

## Software

All molecular dynamics simulations were performed with the OPLS4 force
field using the Desmond engine, accessed through Schrodinger Maestro
(Schrodinger Release 2025-3). Density functional theory calculations were
performed with Jaguar (B3LYP-D3/6-311+G**//B3LYP-D3/6-31G**).

## Raw data availability

Raw molecular dynamics trajectories (multiple gigabytes per system) are
not included in this repository due to size, and are available from the
corresponding author upon reasonable request.

## Contact

Dr. Alok Kumar Tripathi
Ramjas College, University of Delhi, India, 110007
aakt5005@gmail.com
