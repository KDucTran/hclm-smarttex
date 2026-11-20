"""Render the five data figures in the HCLM-SmartTex article and its appendix."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_figure(figure, path):
    figure.savefig(path, metadata={"CreationDate": None, "ModDate": None})
    plt.close(figure)


def plot_cost_slices(source, output):
    costs = pd.read_csv(source / "cost_sensitivity.csv")
    minimum, maximum = costs.cost.min(), costs.cost.max()
    slices = costs.groupby("c_delay", sort=True)
    figure, axes = plt.subplots(1, len(slices), figsize=(6.45, 2.7), constrained_layout=True)

    for axis, (delay, values) in zip(axes, slices, strict=True):
        grid = values.pivot(index="c_miss", columns="c_textile_alarm", values="cost").sort_index()
        image = axis.imshow(grid, origin="lower", vmin=minimum, vmax=maximum,
                            cmap="cividis", aspect="auto")
        axis.set_xticks(range(len(grid.columns)), [f"{value:g}" for value in grid.columns])
        axis.set_yticks(range(len(grid)), [f"{value:g}" for value in grid.index])
        axis.set_xticks(np.arange(-0.5, len(grid.columns), 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(grid), 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=0.35)
        axis.tick_params(which="minor", bottom=False, left=False)
        axis.set_xlabel("Assumed $c_F$", fontsize=9)
        axis.set_title(f"$c_D = {delay:g}$", fontsize=10)
        axis.tick_params(labelsize=8)
        for (row, col), cost in np.ndenumerate(grid.to_numpy()):
            axis.text(col, row, f"{cost:.3f}".lstrip("0"), ha="center", va="center",
                      fontsize=8, color="white" if cost < (minimum + maximum) / 2 else "black")

    axes[0].set_ylabel("Assumed $c_M$", fontsize=9)
    colorbar = figure.colorbar(image, ax=axes, shrink=0.87, label="Reference cost")
    colorbar.ax.tick_params(labelsize=8)
    colorbar.ax.yaxis.label.set_size(9)
    save_figure(figure, output / "cost_all_slices.pdf")


def plot_record_errors(source, output):
    predictions = pd.read_csv(source / "external_predictions_HCLM.csv")
    manifest = json.loads((source / "run_manifest.json").read_text(encoding="utf-8"))
    thresholds = {fold["test_record"]: fold["threshold"] for fold in manifest["respiratory"]["folds"]}
    figure, axes = plt.subplots(2, 1, figsize=(6.25, 3.8), constrained_layout=True)

    for axis, record in zip(axes, ("a04", "b01"), strict=True):
        minutes = predictions[predictions.record == record]
        positives = minutes.loc[minutes.label == 1, "minute"]
        errors = minutes[minutes.pred != minutes.label]
        axis.plot(minutes.minute, minutes.score, lw=1, color="#215a8d", label="Held-out score")
        axis.axhline(thresholds[record], lw=0.9, ls="--", color="#555555", label="Selected threshold")
        axis.scatter(positives, np.full(len(positives), 1.04), marker="|", s=30,
                     color="#7a2260", label="Positive reference minute")
        axis.scatter(errors.minute, errors.score, s=15, facecolors="none",
                     edgecolors="#d04928", label="Incorrect binary decision")
        axis.set(ylim=(-0.04, 1.1), ylabel="Score", title=record)

    axes[-1].set_xlabel("Recording minute")
    axes[0].legend(loc="center right", fontsize=8, ncol=2)
    save_figure(figure, output / "external_failure_records.pdf")


def plot_synthetic_trace(source, output):
    trace = pd.read_csv(source / "hclm_diagnostics.csv").query("subject == 1")
    figure, axes = plt.subplots(3, 1, figsize=(6.25, 4.7), sharex=True, constrained_layout=True)
    axes[0].plot(trace.end, trace.phys, label="Physiological")
    axes[0].plot(trace.end, trace.tex, label="Textile")
    axes[0].legend(fontsize=8)
    axes[0].set_ylabel("Mechanistic score", fontsize=9)
    axes[1].plot(trace.end, trace.hybrid, label="Unregulated")
    axes[1].plot(trace.end, trace.score, label="HCLM")
    axes[1].plot(trace.end, trace.h, "k--", label="Threshold")
    axes[1].legend(fontsize=8, ncol=3)
    axes[1].set_ylabel("Score", fontsize=9)
    axes[2].step(trace.end, trace.family, label="Reference", where="post")
    axes[2].step(trace.end, trace.pred, label="Predicted", where="post", linestyle="--")
    axes[2].set_yticks([0, 1, 2], ["Normal", "Phys.", "Textile"])
    axes[2].legend(fontsize=8)
    axes[2].set_xlabel("Decision availability time (s)", fontsize=9)
    for axis in axes:
        axis.tick_params(labelsize=9)
    save_figure(figure, output / "synthetic_trace.pdf")


def plot_tolerances(source, output):
    results = pd.read_csv(source / "threshold_misspecification.csv")
    names = {"d": "Desaturation", "h": "Heart rate", "c": "Contact",
             "q": "Strain", "rho": "Correlation", "joint": "joint"}
    figure, axis = plt.subplots(figsize=(5.3, 3.4), constrained_layout=True)
    for key, series in results.groupby("parameter"):
        axis.plot(series.scale, series.cost, "o-", markersize=4, label=names[key])
    axis.set_xlabel("Multiplier on mechanistic tolerance", fontsize=9)
    axis.set_ylabel("Reference cost", fontsize=9)
    axis.set_xticks([0.75, 1, 1.25])
    axis.tick_params(labelsize=9)
    axis.legend(ncol=3, fontsize=8, loc="upper center")
    save_figure(figure, output / "threshold_sensitivity.pdf")


def plot_calibration(source, output):
    results = pd.read_csv(source / "adaptive_stress.csv")
    figure, axes = plt.subplots(1, 2, figsize=(6.25, 3.1), constrained_layout=True)
    for axis, metric, label in zip(axes, ("cost", "phys_miss"),
                                   ("Reference cost", "Physiological miss rate"), strict=True):
        for amount, group in results.groupby("amplitude_shift"):
            axis.plot(group["mode"], group[metric], "o-", markersize=4, label=f"{amount:.0%} shift")
        axis.set_ylabel(label, fontsize=9)
        axis.tick_params(axis="x", rotation=20, labelsize=8)
        axis.tick_params(axis="y", labelsize=9)
        axis.legend(fontsize=8, loc="best")
    save_figure(figure, output / "adaptive_stress.pdf")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Completed study run with its CSVs and run_manifest.json")
    parser.add_argument("output", type=Path, help="Directory for the five PDF figures")
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    plt.rcParams.update({
        "font.size": 10, "font.family": "DejaVu Sans", "axes.spines.top": False,
        "axes.spines.right": False, "pdf.fonttype": 42, "savefig.bbox": "tight",
    })
    plot_cost_slices(args.source, args.output)
    plot_record_errors(args.source, args.output)
    plot_synthetic_trace(args.source, args.output)
    plot_tolerances(args.source, args.output)
    plot_calibration(args.source, args.output)


if __name__ == "__main__":
    main()
