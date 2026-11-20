# HCLM-SmartTex manuscript tables

Tables 1–3 appear in the six main sections. Tables S1–S12 appear in the appendix of the same article.

## Contents

| Table | Contents |
| --- | --- |
| [Table 1](#table-1) | Scope and limits of the two evaluations |
| [Table 2](#table-2) | Synthetic model comparison |
| [Table 3](#table-3) | HCLM errors by recording |
| [Table S1](#table-s1) | Apnea-ECG minute counts |
| [Table S2](#table-s2) | Settings for held-out recordings |
| [Table S3](#table-s3) | Family confusion matrix |
| [Table S4](#table-s4) | Performance by synthetic root label |
| [Table S5](#table-s5) | Settings for held-out synthetic subjects |
| [Table S6](#table-s6) | Operating ranges by delay weight |
| [Table S7](#table-s7) | Selected cost settings |
| [Table S8](#table-s8) | Tolerance perturbations |
| [Table S9](#table-s9) | Online calibration under drift |
| [Table S10](#table-s10) | Apnea-ECG model comparison |
| [Table S11](#table-s11) | Paired synthetic alert changes |
| [Table S12](#table-s12) | Paired changes by recording |

## Table 1

**Scope and limits of the two evaluations**

What the two evaluations establish and which measurements are still needed.

| Concern | Completed work | Remaining limit |
| --- | --- | --- |
| Measured respiratory events | Eight held-out recordings, 3,802 annotated minutes and errors reported for each recording. | Independence of participants across recordings is unverified. |
| Action costs | 75 cost configurations used for policy selection. Each selected policy was also evaluated against one fixed reference loss. | Deployment costs and acceptable alert burden require stakeholder input. |
| Changing garment conditions | A bounded, guarded update of the amplitude references was tested with zero and 20% imposed drift. | Responses to measured humidity and posture, and federated personalization, are untested. |
| Garment-fault attribution | Labeled synthetic detachment, drift and misplacement were evaluated. A physical detachment protocol is specified. | The sign of the score difference and the resulting action remain unverified on a physical garment fault. |
| Transfer to other tasks | A reusable decision rule, a reduced physiological module and an illustrative electrocardiogram score are specified. | Accuracy in other pathologies or sensing tasks remains untested. |

[Back to contents](#contents)

## Table 2

**Synthetic model comparison**

Pooled held-out windows at the reference cost. A physiological miss includes both normal and textile decisions on a physiological window.

| Method | F1 | AUROC | Family acc. | Phys. miss | Cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| Rules | 0.6527 | 0.8890 | 0.6233 | 0.1008 | 0.5933 |
| IsolationForest | 0.8074 | 0.9603 | 0.8198 | 0.0853 | 0.4402 |
| OneClassSVM | 0.8591 | 0.9814 | 0.8655 | 0.0620 | 0.3195 |
| Logistic | 0.7808 | 0.9482 | 0.7879 | 0.0620 | 0.3753 |
| ExtraTrees | 0.8534 | 0.9484 | 0.8681 | 0.0620 | 0.3267 |
| Hybrid | 0.8349 | 0.9626 | 0.8422 | 0.0620 | 0.3494 |
| HCLM | 0.8341 | 0.9673 | 0.8414 | 0.0620 | 0.3499 |

Source files · [synthetic_results.csv](<reference_csv/synthetic_results.csv>)

[Back to contents](#contents)

## Table 3

**HCLM errors by recording**

FN counts missed apnea minutes and FP counts false alerts. A dash marks sensitivity with no positive minutes.

| Record | Positive | Negative | FN | FP | Sensitivity | Specificity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| a01 | 470 | 14 | 0 | 6 | 1.000 | 0.571 |
| a02 | 420 | 103 | 0 | 52 | 1.000 | 0.495 |
| a03 | 246 | 268 | 1 | 83 | 0.996 | 0.690 |
| a04 | 388 | 34 | 0 | 33 | 1.000 | 0.029 |
| b01 | 18 | 454 | 16 | 5 | 0.111 | 0.989 |
| c01 | 0 | 462 | 0 | 20 | — | 0.957 |
| c02 | 0 | 477 | 0 | 9 | — | 0.981 |
| c03 | 0 | 448 | 0 | 3 | — | 0.993 |

Source files · [external_record_errors.csv](<reference_csv/external_record_errors.csv>)

[Back to contents](#contents)

## Table S1

**Apnea-ECG minute counts**

The warm-up column is the zero-based index of its final minute. Invalid minutes were excluded independently of the annotations.

| Record | Evaluated min. | Apnea min. | Invalid SpO₂ min. | Warm-up final index |
| --- | ---: | ---: | ---: | ---: |
| a01 | 484 | 470 | 0 | 4 |
| a02 | 523 | 420 | 0 | 4 |
| a03 | 514 | 246 | 0 | 4 |
| a04 | 422 | 388 | 65 | 4 |
| b01 | 472 | 18 | 9 | 7 |
| c01 | 462 | 0 | 16 | 20 |
| c02 | 477 | 0 | 19 | 21 |
| c03 | 448 | 0 | 0 | 4 |

Source files · [external_record_counts.csv](<reference_csv/external_record_counts.csv>)

[Back to contents](#contents)

## Table S2

**Settings for held-out recordings**

Two recordings supplied validation data and the other five supplied training data. All selected entropy thresholds were 0.45.

| Test record | Validation records | ω | λ | h |
| --- | ---: | ---: | ---: | ---: |
| a01 | a02, c01 | 0.20 | 0.60 | 0.2654 |
| a02 | a03, c02 | 0.05 | 0.60 | 0.1552 |
| a03 | a04, c03 | 0.05 | 0.00 | 0.0410 |
| a04 | b01, c01 | 0.05 | 0.00 | 0.0018 |
| b01 | a01, c02 | 0.05 | 0.00 | 0.3500 |
| c01 | a01, c03 | 0.05 | 0.00 | 0.1312 |
| c02 | a02, c01 | 0.50 | 0.15 | 0.2092 |
| c03 | a03, c02 | 0.50 | 0.30 | 0.1426 |

Source files · [run_manifest.json](<reference_csv/run_manifest.json>)

[Back to contents](#contents)

## Table S3

**Family confusion matrix**

Rows identify the reference family and columns give predicted families.

| Reference | Normal | Physiological | Textile |
| --- | ---: | ---: | ---: |
| Normal | 545 | 70 | 89 |
| Phys | 3 | 121 | 5 |
| Textile | 13 | 4 | 310 |

Source files · [family_confusion.csv](<reference_csv/family_confusion.csv>)

[Back to contents](#contents)

## Table S4

**Performance by synthetic root label**

Binary recall measures anomaly detection. Family accuracy uses the normal, physiological and textile families.

| Reference root | Windows | Binary recall | Family acc. |
| --- | ---: | ---: | ---: |
| Normal | 704 | — | 0.7741 |
| Apnea-like | 18 | 1.0000 | 1.0000 |
| Hypopnea-like | 111 | 0.9730 | 0.9279 |
| Detachment | 72 | 0.9722 | 0.9722 |
| Drift | 72 | 0.8472 | 0.8333 |
| Misplacement | 183 | 1.0000 | 0.9836 |

Source files · [predictions_HCLM.csv](<reference_csv/predictions_HCLM.csv>)

[Back to contents](#contents)

## Table S5

**Settings for held-out synthetic subjects**

Columns give the validation subjects, label count, mixture weight, correction strength, entropy threshold and alert threshold.

| Test subject | Validation | Labels | ω | λ | H₀ | h |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1,3 | 271 | 0.50 | 0.60 | 0.45 | 0.0797 |
| 1 | 2,4 | 227 | 0.50 | 0.00 | 0.45 | 0.2920 |
| 2 | 3,5 | 283 | 0.50 | 0.30 | 0.45 | 0.0765 |
| 3 | 4,6 | 256 | 0.50 | 0.00 | 0.45 | 0.0712 |
| 4 | 5,7 | 251 | 0.20 | 0.60 | 0.45 | 0.0716 |
| 5 | 6,0 | 299 | 0.50 | 0.00 | 0.45 | 0.0715 |
| 6 | 7,1 | 280 | 0.50 | 0.60 | 0.45 | 0.0559 |
| 7 | 0,2 | 296 | 0.05 | 0.30 | 0.75 | 0.1946 |

Source files · [run_manifest.json](<reference_csv/run_manifest.json>)

[Back to contents](#contents)

## Table S6

**Operating ranges by delay weight**

Each slice contains 25 configurations evaluated at common reference weights.

| Assumed c_D | Reference cost | Phys. miss (%) | Normal alert (%) | Delay (s) |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.391–0.486 | 8.5–11.6 | 5.5–15.9 | 39.0–49.0 |
| 0.5 | 0.341–0.350 | 6.2–6.2 | 19.3–22.6 | 34.0–34.0 |
| 2 | 0.361–0.368 | 5.4–5.4 | 38.6–40.6 | 27.3–27.3 |

Source files · [cost_slices.csv](<reference_csv/cost_slices.csv>)

[Back to contents](#contents)

## Table S7

**Selected cost settings**

Scenario cost uses the indicated weights. Reference cost uses the nominal weights.

| c_M | c_F | c_D | Scenario cost | Reference cost | Phys. miss | Textile → phys. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.5 | 0.0 | 0.1043 | 0.4372 | 0.1008 | 0.0092 |
| 2 | 8.0 | 0.5 | 0.2410 | 0.3405 | 0.0620 | 0.0122 |
| 8 | 2.0 | 0.5 | 0.3499 | 0.3499 | 0.0620 | 0.0122 |
| 32 | 0.5 | 0.5 | 0.8399 | 0.3499 | 0.0620 | 0.0122 |
| 32 | 8.0 | 2.0 | 1.0475 | 0.3611 | 0.0543 | 0.0122 |

Source files · [cost_sensitivity.csv](<reference_csv/cost_sensitivity.csv>)

[Back to contents](#contents)

## Table S8

**Tolerance perturbations**

Reference cost at each multiplier with the classifier and decision threshold held fixed.

| Perturbed tolerance | 0.75× | 1.00× | 1.25× |
| --- | ---: | ---: | ---: |
| Desaturation | 0.3504 | 0.3499 | 0.3499 |
| Heart-rate rise | 0.3513 | 0.3499 | 0.3442 |
| Contact drop | 0.3458 | 0.3499 | 0.3519 |
| Strain deviation | 0.3556 | 0.3499 | 0.3524 |
| Correlation | 0.3519 | 0.3499 | 0.3438 |
| All jointly | 0.3590 | 0.3499 | 0.3454 |

Source files · [threshold_misspecification.csv](<reference_csv/threshold_misspecification.csv>)

[Back to contents](#contents)

## Table S9

**Online calibration under drift**

Update counts cover 1,160 synthetic windows per condition.

| Amplitude shift | Baseline | Updates | F1 | Family acc. | Cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0% | static | 0 | 0.8341 | 0.8414 | 0.3499 |
| 0% | unrestricted | 1160 | 0.7971 | 0.7655 | 0.4602 |
| 0% | guarded | 393 | 0.8232 | 0.8293 | 0.3519 |
| 20% | static | 0 | 0.6377 | 0.4853 | 0.6677 |
| 20% | unrestricted | 1160 | 0.7690 | 0.7198 | 0.4529 |
| 20% | guarded | 369 | 0.7106 | 0.6155 | 0.5557 |

Source files · [adaptive_stress.csv](<reference_csv/adaptive_stress.csv>)

[Back to contents](#contents)

## Table S10

**Apnea-ECG model comparison**

Each recording was held out in full. Cost equals eight times the false-negative rate plus the false-positive rate.

| Method | F1 | AUROC | Sensitivity | Specificity | Cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic | 0.9242 | 0.9946 | 0.9877 | 0.8978 | 0.2008 |
| Mechanistic | 0.8788 | 0.9743 | 0.9870 | 0.8230 | 0.2808 |
| Hybrid | 0.9273 | 0.9903 | 0.9890 | 0.9018 | 0.1864 |
| HCLM | 0.9304 | 0.9895 | 0.9890 | 0.9066 | 0.1816 |

Source files · [external_results.csv](<reference_csv/external_results.csv>)

[Back to contents](#contents)

## Table S11

**Paired synthetic alert changes**

The first comparison fixes the HCLM fusion weight and threshold. The second compares independently selected HCLM and hybrid policies. All differences in the second comparison occurred on normal windows.

| Subject | Fixed · normal alerts removed | Fixed · maintenance alerts added | Separate · alerts removed | Separate · alerts added |
| --- | ---: | ---: | ---: | ---: |
| 0 | 6 | 0 | 0 | 2 |
| 1 | 0 | 0 | 0 | 0 |
| 2 | 4 | 0 | 1 | 2 |
| 3 | 0 | 0 | 0 | 0 |
| 4 | 5 | 1 | 2 | 0 |
| 5 | 0 | 0 | 0 | 0 |
| 6 | 1 | 0 | 1 | 1 |
| 7 | 0 | 0 | 0 | 0 |
| Total | 16 | 1 | 4 | 5 |

Source files · [paired_synthetic.csv](<paired_csv/paired_synthetic.csv>)

[Back to contents](#contents)

## Table S12

**Paired changes by recording**

Changes from logistic regression to reduced HCLM on the same held-out minute. Negative minutes supply the false-alert counts.

| Record | False alerts removed | False alerts added | Previously missed minutes recovered | Newly missed |
| --- | ---: | ---: | ---: | ---: |
| a01 | 3 | 0 | 0 | 0 |
| a02 | 4 | 1 | 1 | 0 |
| a03 | 0 | 7 | 1 | 0 |
| a04 | 0 | 0 | 0 | 0 |
| b01 | 0 | 0 | 0 | 0 |
| c01 | 0 | 0 | 0 | 0 |
| c02 | 17 | 1 | 0 | 0 |
| c03 | 6 | 1 | 0 | 0 |
| Total | 30 | 10 | 2 | 0 |

Source files · [paired_respiratory.csv](<paired_csv/paired_respiratory.csv>)

[Back to contents](#contents)

Measured respiratory data are from the Apnea-ECG Database v1.0.0 (Moody and Mark, 2000), distributed under the Open Data Commons Attribution License v1.0. [PhysioNet record](https://physionet.org/content/apnea-ecg/1.0.0/) · [Dataset DOI](https://doi.org/10.13026/C23W2R)
