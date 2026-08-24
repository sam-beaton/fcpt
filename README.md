# fcpt

Code for functional change point (FCPt) detection and threshold-free cluster enhancement (TFCE) analysis of infant fNIRS/HD-DOT data, developed for two related studies conducted in rural Gambia as part of the BRIGHT and INDiGO projects:

- **Beaton et al.**, *"Characterizing developmental changes in infant habituation using functional change point detection"*, Neurophotonics 13(3), 035006 (2026). [DOI: 10.1117/1.NPh.13.3.035006](https://doi.org/10.1117/1.NPh.13.3.035006) — group-level, sparse-array fNIRS.
- **Beaton et al.**, *"Beyond group averages: individual differences in infant auditory habituation timing using high-density diffuse optical tomography in rural Gambia"* (In Prep.) — individual-level, HD-DOT.

The individual-level analysis (TFCE + FCPt applied to individual infants, using HD-DOT parcel-space data) is the primary focus of this repository going forward, and is documented in detail below. Code for the earlier group-level, sparse-array analysis is also present (see [Repository structure](#repository-structure)) but is not separately documented here.

## Repository structure

```
fcpt/
├── scripts/         # Analysis notebooks (TFCE, FCPt detection, mesh generation/visualisation)
├── statsScripts/     # Statistical testing notebooks (OLR, Wald, Tukey HSD on FCPt output)
├── plotting/         # Result visualisation notebooks
├── functions/        # Shared Python modules imported across notebooks
├── examples/         # Worked examples on public climate data (see below)
├── dataRaw/          # Expected location for input data (not included — see Data availability)
├── requirements.txt
└── LICENSE
```

## Requirements

- Python 3.10+ (tested on 3.13)
- MATLAB, if reproducing the upstream image reconstruction step (not part of this repository — see the companion `imageReconstruction` repository)

Install dependencies into a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` is a broad environment snapshot covering several projects that share this codebase; not every package is needed for the individual-level pipeline described below.

**A note for Windows users**

*This pipeline has been developed and tested on macOS only. File paths throughout use Python's pathlib, which is cross-platform by design, but the wider installation and runtime environment hasn't been verified on Windows. A couple of things worth expecting:*

*Some dependencies may need to be compiled from source during installation, which can require build tools that aren't present on Windows by default.*
*A small number of scripts assume a Mac/Linux-style external toolchain on the system PATH; these may need adapting, or running via WSL, on Windows.*

*If you're setting this up on Windows, budget some extra time for installation in particular.*

## Reproducing the individual-level analysis

Notebooks are run in this order. Each expects the outputs of the previous step to already exist in `dataRaw/` or `data/` (created automatically as you go).

1. **`scripts/getParcelTrialsDataframe.ipynb`** — converts MATLAB-format reconstructed parcel data (produced by the companion `imageReconstruction` repository) into the Python-format metadata CSV + HDF5 time series used by everything downstream.
2. **`scripts/buildSurfAdjacencyMatrices.ipynb`** — builds the spatial parcel-adjacency graph used by spatiotemporal TFCE.
3. **`scripts/indDotTfceTTestSurfParcelSpatiotemporal4fcpt.ipynb`** — individual-level TFCE: activation, habituation, delayed habituation, long-range habituation, and novelty tests, with circular-shift / exhaustive-label-permutation empirical nulls.
4. **`scripts/grpDotContrastsSurfParcel.ipynb`** and **`scripts/groupAndIndividualComparisons.ipynb`** — group-level paired t-tests and cross-level concordance testing between group- and individual-level results.
5. **`scripts/postHocMechanisticTestsHbO.ipynb`** / **`HbR.ipynb`** — chromophore-specific post-hoc mechanistic tests on TFCE output.
6. **`scripts/indDotChangeParcel.ipynb`** — functional change point (FCPt) detection on individual-level parcel data, restricted to the categories reported in the paper.
7. **`statsScripts/statIindigoIndividualChangePoints.ipynb`** and **`...ChromSpecific.ipynb`** — ordinal logistic regression, Wald tests, and Tukey HSD on the detected change points, cross-chromophore and chromophore-specific.
8. **`scripts/makeBrainMesh.ipynb`** and **`scripts/plotMeshParcellation.ipynb`** — cortical surface mesh generation and parcellation visualisation, used throughout the above for figures.

The shared logic underlying steps 3, 6, and 7 (spatiotemporal TFCE, wild binary segmentation, Monte Carlo Brownian-bridge significance testing) lives in `functions/fNIRSData.py`, `functions/auxFuncChPt.py`, and `functions/funcChangePoint.py`.

## Example usage on public data

`examples/` contains worked examples of the FF and FPCA-based change point detection approaches (`EXAMPLEclimateFFChange.ipynb`, `EXAMPLEclimateFPCAChange.ipynb`, and the corresponding fNIRS-flavoured examples) applied to publicly available climate data, for anyone wanting to try the method without needing access to the study data itself.

## Data availability

The infant fNIRS/HD-DOT data used in both studies are stored in the Brain Imaging for Global Health Data Repository. Ethics approval does not permit public archiving of pseudo-anonymised study data, given the combination of neuroimaging, sociodemographic, geographic, and health measures involved. Access is granted to named individuals who pre-register their proposed analysis and complete a data sharing agreement, via the Contact page of the BRIGHT project website.

## License

See [LICENSE](LICENSE).
