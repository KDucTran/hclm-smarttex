# HCLM-SmartTex

Research code for *HCLM-SmartTex: Mechanistic attribution and cost-sensitive uncertainty regulation for smart-textile monitoring*, by Kim Duc Tran, Guillaume Tartare, Ludovic Koehl, Xianyi Zeng, Anthony Fleury and Kim Phuc Tran.

The synthetic experiment uses eight simulated garment streams with respiratory events and textile faults. A logistic regression supplies the base anomaly score. The HCLM policy combines it with physiological and textile mechanism scores, adjusts the fused score, and selects an action. A separate experiment on eight Apnea-ECG recordings evaluates the physiological alert. The public recordings carry minute-level apnea annotations. The detachment, drift and misplacement labels come from the simulations.

[Read the 15 manuscript tables](MANUSCRIPT_TABLES.md) · [View the figures](figures/) · [Inspect the reference CSVs](reference_csv/) · [Inspect the paired counts](paired_csv/)

## Reproduce the results

Use Python 3.12 from the repository root. A full run downloads 24 files from PhysioNet on first use and verifies their checksums. The eight synthetic streams are included in the repository.

```sh
python -m pip install -r requirements.txt
python hclm_smarttex.py --strict --output runs
```

The command prints the new run directory under `runs/`. It contains 25 CSVs and `run_manifest.json`, which records the input checksums, software versions, fold assignments, fitted configurations, and bootstrap intervals. The strict check compares each output CSV with its reference SHA-256. The hashes are also available in [SHA256SUMS.csv](SHA256SUMS.csv).

For the synthetic experiment alone, use `python hclm_smarttex.py --study synthetic --strict --output runs`. For a full run with data already on disk, use `python hclm_smarttex.py --public-data /path/to/apnea-ecg --strict --output runs`. The supplied folder must contain the eight annotation files and the corresponding respiratory waveform and header files.

## Rebuild the tables and figures

These commands use the distributed reference results. To draw from a new run, replace `reference_csv` in all three commands with the directory printed by the full run.

```sh
python paired_decisions.py reference_csv runs/paired
python render_tables.py reference_csv runs/paired runs/MANUSCRIPT_TABLES.md
python render_figures.py reference_csv runs/figures
```

The table file opens directly on GitHub. Each numerical table links to its source CSV or fold manifest. Table 1 reproduces the manuscript’s qualitative scope statement. Tables S11 and S12 use matched decisions on the same synthetic window or respiratory minute. The two paired files supplement the 25 CSVs from the main analysis.

| Figure | Material |
| --- | --- |
| [1 · Signal flow](figures/figure1.pdf) | [Editable TikZ](Submission/HCLM_R2_LaTeX/signal_flow_tikz.tex) |
| [2 · All 75 cost settings](figures/cost_all_slices.pdf) | [Cost grid](reference_csv/cost_sensitivity.csv) |
| [3 · Held-out records a04 and b01](figures/external_failure_records.pdf) | [Minute predictions](reference_csv/external_predictions_HCLM.csv) and [selected thresholds](reference_csv/run_manifest.json) |
| [S1 · Synthetic subject 1](figures/synthetic_trace.pdf) | [Window diagnostics](reference_csv/hclm_diagnostics.csv) |
| [S2 · Tolerance perturbations](figures/threshold_sensitivity.pdf) | [Perturbation results](reference_csv/threshold_misspecification.csv) |
| [S3 · Amplitude calibration](figures/adaptive_stress.pdf) | [Calibration results](reference_csv/adaptive_stress.csv) |

Figure 1 is a vector diagram. A TeX installation with the `standalone` class and TikZ can rebuild it with `pdflatex -output-directory=figures figure1.tex`.

## Data, method and attribution

The eight simulated streams contain 14,400 observations at 1 Hz. The 16 CSV parts retain the original row order and decimal precision. The main program verifies the checksum of the reconstructed input before extracting the features. [Synthetic streams](data/synthetic/README.md)

The measured data come from G. Moody and R. Mark, *Apnea-ECG Database*, version 1.0.0, PhysioNet (2000). The data are distributed under the **Open Data Commons Attribution License v1.0**. Retain attribution to the original data when reusing the respiratory derivatives. [Dataset](https://physionet.org/content/apnea-ecg/1.0.0/) · [Dataset DOI](https://doi.org/10.13026/C23W2R) · [License](https://opendatacommons.org/licenses/by/1-0/)

T. Penzel, G. B. Moody, R. G. Mark, A. L. Goldberger and J. H. Peter. “The Apnea-ECG Database.” *Computers in Cardiology* 27 (2000), 255–258. [Paper](https://doi.org/10.1109/CIC.2000.898505)

T. Pollard et al. “PhysioNet as a global platform for biomedical research.” *Nature Health* 1 (2026), 792–795. [Platform citation](https://doi.org/10.1038/s44360-026-00096-z)

K. P. Tran. *Human-Centered Learning Mechanics: Control, Calibration, and Governed Action for Industrial AI Decision Systems*. Springer Nature, 2026, in press. The separate 2026 preprint, *Human-Centered Learning Mechanics: Observable Entropy Control under Protected-Risk Constraints*, describes the entropy-control formulation. [Preprint](https://doi.org/10.48550/arXiv.2605.22940)

The software is distributed under the [MIT License](LICENSE).
