# LRSM tb-channel Analysis Pipeline

This directory contains the HTCondor-based analysis pipeline for studying the Left-Right Symmetric Model (LRSM) in the tb-channel.

## Purpose
This analysis searches for heavy right-handed W boson (WR) production in pp → WR → N μ → μμ tb events, where:
- WR decays to a heavy neutrino N and muon
- N decays to μμ + tb final state
- Uses both AK4 (small-R) and AK8 (large-R) jets for top reconstruction

## Directory Structure

### Analysis Strategies
- `1AK4_1AK8/`: Analysis using 1 AK4 jet (b-quark) + 1 AK8 jet (top-tagged)
- `2AK8/`: Analysis using 2 AK8 jets (both top-tagged)

### Selection Categories
Each strategy contains results organized by selection criteria:
- `btag_PN_tight_hlt_30/` or `tight_hlt_30/`: B-tagging and HLT requirements
- `WR_cut_0/` or `WR_cut_2000/`: WR mass cut thresholds (0 GeV or 2000 GeV)
- `top_0.9_120_250/`: Top tagging score > 0.9, soft drop mass 120-250 GeV
- `mll0pt0eta0/` or `mll300mu1pt50eta2.4/`: Dilepton mass and kinematic cuts

### Analysis Components
- `bkg/`: Background sample analysis results
- `sig/`: Signal sample analysis results
- `condorfile/`: HTCondor job submission scripts and configuration files
- `scripts/`: Analysis code templates like `arun_template.py`

## Analysis Template (`arun_template.py`)

The main analysis script performs:

1. **Object Selection & Cleaning**:
   - Muons: High-pT ID, isolation cuts, overlap removal
   - Jets: AK4 (b-tagging with ParticleNet) and AK8 (top-tagging with ParticleNetWithMass)
   - Overlap removal between muons, jets using ΔR cuts

2. **Event Selection**:
   - HLT_IsoMu30 trigger
   - ≥2 muons, ≥1 b-tagged AK4 jet, ≥1 top-tagged AK8 jet
   - Kinematic cuts on pT and η
   - Invariant mass requirements

3. **Physics Objects**:
   - Reconstructed WR mass: M(μμbt)
   - Heavy neutrino mass: M(μbt)
   - Kinematic distributions of final state objects

4. **Output**:
   - Histograms of key variables (WR mass, dilepton mass, pT distributions)
   - CSV files with binned event counts
   - Weighted event yields for different mass bins

## Key Physics Parameters
- **B-tagging**: ParticleNet working point (0.6734)
- **Top-tagging**: ParticleNetWithMass score > 0.9
- **Soft drop mass**: 120-250 GeV for top candidates
- **Luminosity**: 300 fb⁻¹ (Run 3 projection)

## Condor Workflow
The pipeline uses HTCondor to process multiple signal and background samples in parallel, with automatic job submission and result collection for systematic studies of the LRSM parameter space.