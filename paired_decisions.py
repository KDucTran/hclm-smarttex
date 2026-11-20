"""Tabulate the paired held-out decisions reported in Supplementary Tables S11 and S12."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load(folder, name, keys):
    frame = pd.read_csv(folder / f"{name}.csv").set_index(keys, verify_integrity=True)
    return frame.sort_index()


def align(left, right, truth):
    if not left.index.equals(right.index) or not left[truth].equals(right[truth]):
        raise ValueError("The prediction files must contain the same labeled observations.")


def synthetic(folder):
    keys = ["subject", "start", "end"]
    diag = load(folder, "hclm_diagnostics", keys)
    hybrid = load(folder, "predictions_Hybrid", keys)
    align(diag, hybrid, "family")
    routed = np.where(diag.phys >= diag.tex, 1, 2)
    within = np.where(diag.hybrid > diag.h, routed, 0)
    if not np.array_equal(np.where(diag.score > diag.h, routed, 0), diag.pred):
        raise ValueError("Saved HCLM actions disagree with the decision rule.")
    counts = pd.DataFrame(
        {
            "fixed_normal_alerts_removed": (diag.family == 0) & (within > 0) & (diag.pred == 0),
            "fixed_textile_alerts_added": (diag.family == 2) & (within == 0) & (diag.pred == 2),
            "fixed_all_action_changes": within != diag.pred,
            "tuned_alerts_removed": (hybrid.pred > 0) & (diag.pred == 0),
            "tuned_alerts_added": (hybrid.pred == 0) & (diag.pred > 0),
            "tuned_all_action_changes": hybrid.pred != diag.pred,
        }, index=diag.index,
    )
    return counts.groupby(level="subject").sum().astype(int)


def respiratory(folder):
    keys = ["record", "minute"]
    hclm = load(folder, "external_predictions_HCLM", keys)
    logistic = load(folder, "external_predictions_Logistic", keys)
    align(hclm, logistic, "label")
    gained = (logistic.pred == 0) & (hclm.pred == 1)
    removed = (logistic.pred == 1) & (hclm.pred == 0)
    counts = pd.DataFrame(
        {
            "false_alerts_removed": removed & (hclm.label == 0),
            "false_alerts_added": gained & (hclm.label == 0),
            "positive_minutes_recovered": gained & (hclm.label == 1),
            "positive_minutes_newly_missed": removed & (hclm.label == 1),
        }, index=hclm.index,
    )
    return counts.groupby(level="record").sum().astype(int)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path, help="Directory containing the 25 reference CSVs")
    parser.add_argument("output", type=Path, help="Directory for the two supplementary tables")
    args = parser.parse_args()
    outputs = {
        "paired_synthetic.csv": synthetic(args.predictions),
        "paired_respiratory.csv": respiratory(args.predictions),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        frame.to_csv(args.output / name)


if __name__ == "__main__":
    main()
