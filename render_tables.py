"""Build the manuscript's fifteen GitHub tables from a completed run and its paired counts."""

import argparse
import json
import os
from pathlib import Path

import pandas as pd


# Table 1 records design choices and limitations from the manuscript.
SCOPE = [
    ("Measured respiratory events",
     "Eight held-out recordings, 3,802 annotated minutes and errors reported for each recording.",
     "Independence of participants across recordings is unverified."),
    ("Action costs",
     "75 cost configurations used for policy selection. Each selected policy was also evaluated against one fixed reference loss.",
     "Deployment costs and acceptable alert burden require stakeholder input."),
    ("Changing garment conditions",
     "A bounded, guarded update of the amplitude references was tested with zero and 20% imposed drift.",
     "Responses to measured humidity and posture, and federated personalization, are untested."),
    ("Garment-fault attribution",
     "Labeled synthetic detachment, drift and misplacement were evaluated. A physical detachment protocol is specified.",
     "The sign of the score difference and the resulting action remain unverified on a physical garment fault."),
    ("Transfer to other tasks",
     "A reusable decision rule, a reduced physiological module and an illustrative electrocardiogram score are specified.",
     "Accuracy in other pathologies or sensing tasks remains untested."),
]


def formatted_rows(frame, columns, formats):
    rows = []
    for values in frame.loc[:, columns].itertuples(index=False, name=None):
        rows.append(["—" if pd.isna(v) else format(v, spec) for v, spec in zip(values, formats, strict=True)])
    return rows


def append_total(frame, key, columns):
    rows = formatted_rows(frame, [key, *columns], [""] + ["d"] * len(columns))
    totals = [str(frame[name].sum()) for name in columns]
    return rows + [["Total", *totals]]


def build_tables(source, paired):
    manifest_path = source / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    yield ("1", "Scope and limits of the two evaluations",
           "What the two evaluations establish and which measurements are still needed.",
           ["Concern", "Completed work", "Remaining limit"], SCOPE, [])

    path = source / "synthetic_results.csv"
    metrics = ["F1", "AUROC", "family_accuracy", "phys_miss", "cost"]
    yield ("2", "Synthetic model comparison",
           "Pooled held-out windows at the reference cost. A physiological miss includes both normal and textile decisions on a physiological window.",
           ["Method", "F1", "AUROC", "Family acc.", "Phys. miss", "Cost"],
           formatted_rows(pd.read_csv(path), ["method", *metrics], [""] + [".4f"] * len(metrics)), [path])

    path = source / "external_record_errors.csv"
    yield ("3", "HCLM errors by recording",
           "FN counts missed apnea minutes and FP counts false alerts. A dash marks sensitivity with no positive minutes.",
           ["Record", "Positive", "Negative", "FN", "FP", "Sensitivity", "Specificity"],
           formatted_rows(pd.read_csv(path), ["record", "positive", "negative", "fn", "fp", "sensitivity", "specificity"],
                          ["", "d", "d", "d", "d", ".3f", ".3f"]), [path])

    path = source / "external_record_counts.csv"
    yield ("S1", "Apnea-ECG minute counts",
           "The warm-up column is the zero-based index of its final minute. Invalid minutes were excluded independently of the annotations.",
           ["Record", "Evaluated min.", "Apnea min.", "Invalid SpO₂ min.", "Warm-up final index"],
           formatted_rows(pd.read_csv(path), ["record", "evaluated_minutes", "apnea_minutes", "excluded_bad_spo2_minutes", "warmup_end_minute"],
                          ["", "d", "d", "d", "d"]), [path])

    rows = []
    for fold in manifest["respiratory"]["folds"]:
        weight, strength, _ = fold["config"]
        rows.append([fold["test_record"], ", ".join(fold["validation_records"]), f"{weight:.2f}",
                     f"{strength:.2f}", f"{fold['threshold']:.4f}"])
    yield ("S2", "Settings for held-out recordings",
           "Two recordings supplied validation data and the other five supplied training data. All selected entropy thresholds were 0.45.",
           ["Test record", "Validation records", "ω", "λ", "h"], rows, [manifest_path])

    path = source / "family_confusion.csv"
    matrix = pd.read_csv(path, index_col=0)
    rows = [[str(label).capitalize(), *map(str, values)] for label, values in matrix.iterrows()]
    yield ("S3", "Family confusion matrix", "Rows identify the reference family and columns give predicted families.",
           ["Reference", "Normal", "Physiological", "Textile"], rows, [path])

    path = source / "predictions_HCLM.csv"
    windows = pd.read_csv(path)
    root_names = ["Normal", "Apnea-like", "Hypopnea-like", "Detachment", "Drift", "Misplacement"]
    rows = []
    for root, group in windows.groupby("root", sort=True):
        recall = "—" if root == 0 else f"{(group.pred > 0).mean():.4f}"
        rows.append([root_names[root], str(len(group)), recall, f"{(group.pred == group.family).mean():.4f}"])
    yield ("S4", "Performance by synthetic root label",
           "Binary recall measures anomaly detection. Family accuracy uses the normal, physiological and textile families.",
           ["Reference root", "Windows", "Binary recall", "Family acc."], rows, [path])

    rows = []
    for fold in manifest["synthetic"]["folds"]:
        weight, strength, entropy = fold["config"]
        rows.append([str(fold["test"]), ",".join(map(str, fold["validation"])), str(fold["labelled_windows"]),
                     f"{weight:.2f}", f"{strength:.2f}", f"{entropy:.2f}", f"{fold['threshold']:.4f}"])
    yield ("S5", "Settings for held-out synthetic subjects",
           "Columns give the validation subjects, label count, mixture weight, correction strength, entropy threshold and alert threshold.",
           ["Test subject", "Validation", "Labels", "ω", "λ", "H₀", "h"], rows, [manifest_path])

    path = source / "cost_slices.csv"
    rows = []
    for item in pd.read_csv(path).to_dict("records"):
        row = [f"{item['assumed_c_delay']:g}"]
        for key, factor, precision in [("cost", 1, 3), ("phys_miss", 100, 1),
                                       ("normal_alarm", 100, 1), ("delay_seconds", 1, 1)]:
            lower = factor * item[f"{key}_min"]
            upper = factor * item[f"{key}_max"]
            row.append(f"{lower:.{precision}f}–{upper:.{precision}f}")
        rows.append(row)
    yield ("S6", "Operating ranges by delay weight", "Each slice contains 25 configurations evaluated at common reference weights.",
           ["Assumed c_D", "Reference cost", "Phys. miss (%)", "Normal alert (%)", "Delay (s)"], rows, [path])

    path = source / "cost_sensitivity.csv"
    scenarios = pd.read_csv(path).set_index(["c_miss", "c_textile_alarm", "c_delay"], verify_integrity=True)
    settings = [(2, 0.5, 0), (2, 8, 0.5), (8, 2, 0.5), (32, 0.5, 0.5), (32, 8, 2)]
    rows = []
    for miss, textile, delay in settings:
        selected = scenarios.loc[(miss, textile, delay)]
        rows.append([f"{miss:g}", f"{textile:.1f}", f"{delay:.1f}", f"{selected.scenario_cost:.4f}",
                     f"{selected.cost:.4f}", f"{selected.phys_miss:.4f}", f"{selected.textile_to_phys:.4f}"])
    yield ("S7", "Selected cost settings", "Scenario cost uses the indicated weights. Reference cost uses the nominal weights.",
           ["c_M", "c_F", "c_D", "Scenario cost", "Reference cost", "Phys. miss", "Textile → phys."], rows, [path])

    path = source / "threshold_misspecification.csv"
    tolerances = pd.read_csv(path).pivot(index="parameter", columns="scale", values="cost")
    names = {"d": "Desaturation", "h": "Heart-rate rise", "c": "Contact drop", "q": "Strain deviation",
             "rho": "Correlation", "joint": "All jointly"}
    rows = [[label, *(f"{tolerances.at[key, scale]:.4f}" for scale in (0.75, 1, 1.25))] for key, label in names.items()]
    yield ("S8", "Tolerance perturbations", "Reference cost at each multiplier with the classifier and decision threshold held fixed.",
           ["Perturbed tolerance", "0.75×", "1.00×", "1.25×"], rows, [path])

    path = source / "adaptive_stress.csv"
    yield ("S9", "Online calibration under drift", "Update counts cover 1,160 synthetic windows per condition.",
           ["Amplitude shift", "Baseline", "Updates", "F1", "Family acc.", "Cost"],
           formatted_rows(pd.read_csv(path), ["amplitude_shift", "mode", "update_count", "F1", "family_accuracy", "cost"],
                          [".0%", "", "d", ".4f", ".4f", ".4f"]), [path])

    path = source / "external_results.csv"
    yield ("S10", "Apnea-ECG model comparison",
           "Each recording was held out in full. Cost equals eight times the false-negative rate plus the false-positive rate.",
           ["Method", "F1", "AUROC", "Sensitivity", "Specificity", "Cost"],
           formatted_rows(pd.read_csv(path), ["method", "F1", "AUROC", "sensitivity", "specificity", "cost"],
                          ["", ".4f", ".4f", ".4f", ".4f", ".4f"]), [path])

    path = paired / "paired_synthetic.csv"
    cols = ["fixed_normal_alerts_removed", "fixed_textile_alerts_added", "tuned_alerts_removed", "tuned_alerts_added"]
    yield ("S11", "Paired synthetic alert changes",
           "The first comparison fixes the HCLM fusion weight and threshold. The second compares independently selected HCLM and hybrid policies. All differences in the second comparison occurred on normal windows.",
           ["Subject", "Fixed · normal alerts removed", "Fixed · maintenance alerts added",
            "Separate · alerts removed", "Separate · alerts added"], append_total(pd.read_csv(path), "subject", cols), [path])

    path = paired / "paired_respiratory.csv"
    cols = ["false_alerts_removed", "false_alerts_added", "positive_minutes_recovered", "positive_minutes_newly_missed"]
    yield ("S12", "Paired changes by recording",
           "Changes from logistic regression to reduced HCLM on the same held-out minute. Negative minutes supply the false-alert counts.",
           ["Record", "False alerts removed", "False alerts added", "Previously missed minutes recovered", "Newly missed"],
           append_total(pd.read_csv(path), "record", cols), [path])


def markdown_row(values):
    return "| " + " | ".join(str(value) for value in values) + " |"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Directory containing the run manifest and reference CSVs")
    parser.add_argument("paired", type=Path, help="Output directory of paired_decisions.py")
    parser.add_argument("output", type=Path, help="Output Markdown file")
    args = parser.parse_args()
    tables = list(build_tables(args.source, args.paired))
    lines = ["# HCLM-SmartTex manuscript tables", "",
             "Tables 1–3 appear in the six main sections. Tables S1–S12 appear in the appendix of the same article.", "",
             "## Contents", "", "| Table | Contents |", "| --- | --- |"]
    for number, title, *_ in tables:
        lines.append(f"| [Table {number}](#table-{number.lower()}) | {title} |")

    for number, title, note, header, rows, sources in tables:
        alignment = "---" if number == "1" else "---:"
        lines.extend(["", f"## Table {number}", "", f"**{title}**", "", note, "", markdown_row(header),
                      markdown_row(["---"] + [alignment] * (len(header) - 1))])
        lines.extend(markdown_row(row) for row in rows)
        if sources:
            links = []
            for path in sources:
                relative = Path(os.path.relpath(path, args.output.parent)).as_posix()
                links.append(f"[{path.name}](<{relative}>)")
            lines.extend(["", "Source files · " + " · ".join(links)])
        lines.extend(["", "[Back to contents](#contents)"])
    lines.extend(["", "Measured respiratory data are from the Apnea-ECG Database v1.0.0 (Moody and Mark, 2000), "
                  "distributed under the Open Data Commons Attribution License v1.0. "
                  "[PhysioNet record](https://physionet.org/content/apnea-ecg/1.0.0/) · "
                  "[Dataset DOI](https://doi.org/10.13026/C23W2R)", ""])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes("\n".join(lines).encode("utf-8"))


if __name__ == "__main__":
    main()
