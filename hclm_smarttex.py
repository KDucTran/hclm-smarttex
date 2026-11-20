#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy==2.3.5",
#   "pandas==2.2.3",
#   "scipy==1.17.0",
#   "scikit-learn==1.8.0",
# ]
# ///
"""Reproduce the HCLM-SmartTex experiments on garment faults and respiratory events.

    uv run hclm_smarttex.py --strict
    uv run hclm_smarttex.py --study synthetic
    python hclm_smarttex.py --public-data /path/to/apnea-ecg --output results

For plain Python, install the pinned dependencies listed above. A full run writes
25 CSVs and a run manifest under results/. Use --strict to check the reference hashes.
The 14,400 synthetic observations are stored in data/synthetic/ beside this script.
Pass --synthetic to use the original single CSV. Respiratory runs download and verify
eight Apnea-ECG recordings on first use, or read existing files with --public-data.

K. D. Tran, G. Tartare, L. Koehl, X. Zeng, A. Fleury and K. P. Tran.
HCLM-SmartTex: Mechanistic attribution and cost-sensitive uncertainty regulation
for smart-textile monitoring.
K. P. Tran. Human-Centered Learning Mechanics: Control, Calibration, and Governed
Action for Industrial AI Decision Systems. Springer Nature (2026, in press).
Related preprint on entropy control: https://doi.org/10.48550/arXiv.2605.22940

G. Moody and R. Mark, Apnea-ECG Database (PhysioNet, 2000). Open Data Commons
Attribution License v1.0. https://doi.org/10.13026/C23W2R
T. Penzel et al., Computers in Cardiology 27, 255–258 (2000).
PhysioNet platform citation: https://doi.org/10.1038/s44360-026-00096-z
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import logging
import os
import platform
import shutil
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from http.client import IncompleteRead
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import NamedTuple
from urllib.request import urlopen

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.special import expit
from sklearn.ensemble import ExtraTreesClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

LOG = logging.getLogger("hclm_smarttex")
SEED = 42
BOOTSTRAP_SEED = 20261021
BOOTSTRAP_SAMPLES = 1000
MAX_DELAY = 180.0
SYNTHETIC_SHA256 = "e94144f93e6691d46e2b0b22525eee50df1b239114765d54ebfa0a5fd412c00c"
APNEA_SOURCE = "https://physionet-open.s3.amazonaws.com/apnea-ecg/1.0.0/"
RECORDS = ("a01", "a02", "a03", "a04", "b01", "c01", "c02", "c03")
FEATURES = [
    "amp_rel",
    "amp_drop",
    "effort_rel",
    "effort_drop",
    "desat",
    "hr_rise",
    "contact_loss",
    "contact_drop",
    "q_z",
    "q_iqr_rel",
    "corr",
    "corr_drop",
]
RESPIRATORY_FEATURES = [
    "air_rel",
    "air_low_rel",
    "air_cv",
    "chest_rel",
    "abd_rel",
    "sat_mean",
    "sat_min",
    "sat_range",
    "desat",
]
WINDOW_KEYS = ["subject", "start", "end", "root", "family"]
METHODS = ("Rules", "IsolationForest", "OneClassSVM", "Logistic", "ExtraTrees", "Hybrid", "HCLM")
RESPIRATORY_METHODS = ("Logistic", "Mechanistic", "Hybrid", "HCLM")
TOLERANCES = {"d": 0.8, "h": 4.0, "c": 0.06, "q": 1.5, "rho": 0.45}
REFERENCE_COSTS = (8.0, 2.0, 0.5)
COST_GRID = list(
    itertools.product((2.0, 4.0, 8.0, 16.0, 32.0), (0.5, 1.0, 2.0, 4.0, 8.0), (0.0, 0.5, 2.0))
)
QUANTILES = np.linspace(0, 1, 61)


class Controller(NamedTuple):
    weight: float
    strength: float
    entropy_threshold: float


HYBRID_GRID = [Controller(w, 0.0, 0.45) for w in (0.05, 0.2, 0.5)]
CONTROL_GRID = [
    Controller(w, strength, threshold)
    for w, strength, threshold in itertools.product(
        (0.05, 0.2, 0.5), (0.0, 0.15, 0.3, 0.6), (0.45, 0.75)
    )
]


# Protocol inputs and reference outputs are identified by SHA-256.
RECORD_HASHES = {
    "a01r.hea": "9cb44adf30f78742cdc34027e59dd8b316a079938f2b8c0ae2d703582bd18571",
    "a01r.dat": "368612afa4c755a240e6e66c77ec9957a07ac98c9f4560784d999c64d8302d0d",
    "a01.apn": "b7404271c576572ce8c99af1fe455cae2954dce1c18caf21a61a25bf1c00f803",
    "a02r.hea": "197f4b46140eedd5e90777b33da8a9bd5950e511f23be23427779e3c0eeec212",
    "a02r.dat": "48f62081ea11e8825740f96e1d1e69160bd8912ed8f037212c17e2d9babd456e",
    "a02.apn": "4f2541a460552115815bc6281c9a4158e15ababdbf3dd8f98bc7252a4ab9bc10",
    "a03r.hea": "82e3d29f38abfb2b1829a4e5e2638d77afb4a7041d1c9b17d780cf0761f96de6",
    "a03r.dat": "c303cc80b921d556fa08dd0d7c4c90484d236b8b62a399a9ce2c7c6b4b8dec0b",
    "a03.apn": "74a415ec32df680de4f23a741c70d56dffad72e39784ca88cfd1a834a556bc0b",
    "a04r.hea": "e2556d33ec05650ac9c50da335db8e090cbd39aa86f3f1c7bc8d10cab60eca9b",
    "a04r.dat": "ea0b6ebbf343229c302bfa22c52faecdbb05d0d3a7aee6125ef0920c526d89b6",
    "a04.apn": "3c3dfc0443e041a391eee8027ebec371bc01b88d45f9c00dff897f4dba8ea1ba",
    "b01r.hea": "16c7aa93e9150607652a8c9e7f90cdb1f2c386cea81bee9e794b5ddcfc70a445",
    "b01r.dat": "88f41557437cc3a69c9a00b4f2ca2bc4083642bb1fe531e964c36645a03ea65d",
    "b01.apn": "f217a9702e885eff675cb9092530de06826c136463f80bf51c5099e47e5cd7d6",
    "c01r.hea": "7aab6eedb57920f17017136df5414bf390168274cb67c17c0fec10e7387a755c",
    "c01r.dat": "693e3c0740526b5a7b6667a9d3431b31517d963e432aab7213310e3a7e94fc85",
    "c01.apn": "92fc5b054de48704d7608543aab7cc9220cb716027c82ce63488ac5cc3409bb5",
    "c02r.hea": "004a6cb8ae11980d2f1229ce16f04d1de4d1cb7a834acb763b6796deeb1b592e",
    "c02r.dat": "48cd1d39e8288d0d2f4b1185863b64112411fc91bc7403ddf86c922fff29cc99",
    "c02.apn": "53e0b39c105932192d7729e486b481f8e8d64729c3de4d84cc5a6e80fbc2e6d4",
    "c03r.hea": "4d9a4706727bdaa59fa7051e833ffbd55c5b53b31a7a21b094145da9d215bee4",
    "c03r.dat": "4b30016578e67403fb670617ccf466881deb2d7b87f76b7b359f6b5e2b8073fc",
    "c03.apn": "c2e4fb5ed0dc412e1ba75bd77ef296168e0c33944eaf7103be2699eeeb11e1b5",
}
TABLE_HASHES = {
    "synthetic": {
        "adaptive_stress.csv": "7a616751aa508456cc5a3def09b762861e9db3eb47871973160f488508dd93e4",
        "cost_sensitivity.csv": "e63fd044659711cd7fa9aef49577b17b0d573ea6b52160aad3695d56d84558fd",
        "cost_slices.csv": "44566ecff18c18cd6456307faf6359b4ff9d42c0d8427627c2f3589678711bef",
        "family_confusion.csv": "40eea0479655670aa9af1f44ba632c66f5b3666aab270d58e3d308e12e3c2289",
        "hclm_diagnostics.csv": "2738320e67c07ea067a992dfbdb1c6c2be6901ed76d582654293396c2ccd258d",
        "predictions_ExtraTrees.csv": "6d7f3422e5857d1905b4a0a09be73411e10624b00ac2b9e1ea70a85c98e04174",
        "predictions_HCLM.csv": "337ffec564d14634ff03646835baa43ed85123b2d63dbaee949d9c1f84cb85b5",
        "predictions_Hybrid.csv": "d89c5cfb23a05bcd5967f1b0d056b36d4e145adce6c3bb98c45983c96792750e",
        "predictions_IsolationForest.csv": "20729b2a16112eb8c8d73adb3e89c1bd795b5acb4bf676e4acbd6f68b3c18552",
        "predictions_Logistic.csv": "bc57f380aa3444f2408650eb64a724e23cd65ac9f6f2020b8c93279916846aa9",
        "predictions_OneClassSVM.csv": "006965dda4660319b338433c796fd25f62ddead29e958fdbd7720d5f0cef8bd3",
        "predictions_Rules.csv": "c9c26dded85c8c4f81f4bf175b5810a78a0bb754a4b69ef7ce1e7ce598c0bc2a",
        "synthetic_features.csv": "524805396a00ae29edc7a1cbd8efeb062fc2cba0273c0c6350f2f5b448730902",
        "synthetic_physiological_events.csv": "24fc8f6b092d1d7da960bc464e20ff49e33aa53ba128294858792f3fb1ff1496",
        "synthetic_results.csv": "911ac58702f8851593c813467c45f81343451b23ec722e234e9ca2ea4c856aab",
        "threshold_misspecification.csv": "a476c6e19dd45ff34e32458814b9f861578504d836f5f542e9027c1038f138b8",
    },
    "respiratory": {
        "apnea_ecg_minute_features.csv": "8d8eb8ade59c9be1837c6a3eb6e3f13d782b2b8bdb9bf0615704c1db4ec46067",
        "external_predictions_HCLM.csv": "b5a3bb06ac65f78477ed113ec4dafcb81ef1937f4300e3e14ee3977893d26a2f",
        "external_predictions_Hybrid.csv": "969fcba291823cfdd6bda52722a8cde0964383049d6d7cb08e7c959c95d72856",
        "external_predictions_Logistic.csv": "acb73da0991d48f5850bea411afbf05784b8b60f473fc7eed15c4490f3aaea79",
        "external_predictions_Mechanistic.csv": "9184106bbbf6e682e28fa3f0e6dca4f90a1e44a3eb310eab1d00c92363e2cc1d",
        "external_record_counts.csv": "3089ca49d3b8b2785012d01578ddefa1411e29c29fc2eb29498317bf25a4b81c",
        "external_record_errors.csv": "4a1f387a2db1d953a15ba27bc0503954220c3601e00fdf2d538dfa9c02f32141",
        "external_results.csv": "c7e6448b55865842fafcea751a123607d6d830e72ae7a5c4b356b79d63649b0c",
        "validation_split_composition.csv": "b164c4a2fc67be555639881b463e74c5a3e5e8958fa5f17200adceca3ef967ad",
    },
}


def write_csv(frame: pd.DataFrame, directory: Path, name: str, *, index: bool = False) -> None:
    frame.to_csv(directory / name, index=index, lineterminator="\n")


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_benchmark(source: Path | None) -> pd.DataFrame:
    """Read the released observations and verify them before decimal parsing."""
    if source is not None:
        raw = source.read_bytes().replace(b"\r\n", b"\n")
    else:
        directory = Path(__file__).resolve().parent / "data" / "synthetic"
        chunks = []
        header = None
        for subject, part in itertools.product(range(8), (1, 2)):
            path = directory / f"subject_{subject:02d}_part_{part}.csv"
            try:
                payload = path.read_bytes().replace(b"\r\n", b"\n")
            except FileNotFoundError as error:
                raise FileNotFoundError(
                    f"Missing {path}. Keep data/synthetic beside the script "
                    "or pass --synthetic /path/to/synthetic_timeseries.csv."
                ) from error
            title, separator, rows = payload.partition(b"\n")
            if not separator or not rows or (header is not None and title != header):
                raise ValueError(f"Invalid synthetic CSV header or empty data in {path.name}.")
            if header is None:
                header = title
                chunks.append(header + b"\n")
            chunks.append(rows)
        raw = b"".join(chunks)
    if hashlib.sha256(raw).hexdigest() != SYNTHETIC_SHA256:
        raise ValueError("Synthetic CSV does not match the fixed benchmark checksum.")
    # Match the decimal parsing used for the original window calculations.
    return pd.read_csv(BytesIO(raw))


def prepare_recordings(cache: Path, supplied: Path | None) -> Path:
    """Verify each waveform, header and annotation before opening any recording."""
    folder = supplied if supplied is not None else cache
    LOG.info("Checking Apnea-ECG 1.0.0 input files in %s", folder)
    if supplied is None:
        folder.mkdir(parents=True, exist_ok=True)

    def fetch(name: str) -> None:
        path = folder / name
        expected = RECORD_HASHES[name]
        if path.is_file() and file_hash(path) == expected:
            return
        if supplied is not None:
            raise ValueError(f"Missing or incorrect Apnea-ECG file: {path}")
        if name.endswith(".dat"):
            LOG.info("Downloading %s", name)
        for attempt in range(3):
            with NamedTemporaryFile(prefix=".part-", dir=folder, delete=False) as local:
                temporary = Path(local.name)
            try:
                with (
                    urlopen(APNEA_SOURCE + name, timeout=60) as remote,
                    temporary.open("wb") as local,
                ):
                    shutil.copyfileobj(remote, local, length=1024 * 1024)
                if file_hash(temporary) != expected:
                    raise ValueError(f"Checksum mismatch for {name}")
                os.replace(temporary, path)
                return
            except (OSError, ValueError, IncompleteRead) as error:
                if attempt == 2:
                    raise OSError(f"Could not download {name}: {error}") from error
            finally:
                temporary.unlink(missing_ok=True)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch, RECORD_HASHES))
    return folder


def amplitude(samples) -> float:
    return float(np.quantile(samples, 0.95) - np.quantile(samples, 0.05))


def correlation(first, second) -> float:
    if min(np.std(first), np.std(second)) <= 1e-9:
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])


def extract_windows(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Build causal 60-s windows from 1-Hz streams, with a 300-s commissioning period."""
    rows, episodes = [], []
    for subject, stream in raw.groupby("subject_id", sort=True):
        stream = stream.reset_index(drop=True)
        initial = stream.iloc[:300]
        if not initial.label.eq(0).all():
            raise ValueError(f"Unexpected event in commissioning for subject {subject}.")
        starts = range(0, 241, 10)
        resp_range = [amplitude(initial.respiration.iloc[j : j + 60]) for j in starts]
        effort_range = [amplitude(initial.thoracic_strain.iloc[j : j + 60]) for j in starts]
        textile_means = [float(initial.textile_strain.iloc[j : j + 60].mean()) for j in starts]
        resp_ref, effort_ref = np.mean(resp_range), np.mean(effort_range)
        resp_sd = max(np.std(resp_range, ddof=1), 0.05 * resp_ref)
        effort_sd = max(np.std(effort_range, ddof=1), 0.05 * effort_ref)
        textile_mean = float(np.mean(textile_means))
        textile_sd = max(float(np.std(textile_means, ddof=1)), 0.01)
        textile_iqr = float(
            initial.textile_strain.quantile(0.75) - initial.textile_strain.quantile(0.25)
        )
        contact_ref = float(initial.sensor_contact.mean())
        saturation_ref, heart_ref = float(initial.spo2.mean()), float(initial.heart_rate.mean())
        corr_ref = correlation(initial.respiration, initial.thoracic_strain)

        labels = stream.label.to_numpy()
        onsets = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1]
        for onset, offset in zip(onsets, np.r_[onsets[1:], len(labels)], strict=True):
            if labels[onset] in (1, 2):
                episodes.append(
                    {
                        "subject": int(subject),
                        "onset": int(onset),
                        "offset": int(offset),
                        "root": int(labels[onset]),
                    }
                )

        for start in range(300, len(stream) - 59, 10):
            window = stream.iloc[start : start + 60]
            resp = amplitude(window.respiration)
            effort = amplitude(window.thoracic_strain)
            root = int(window.label.value_counts().sort_index().idxmax())
            family = 0 if root == 0 else (1 if root in (1, 2) else 2)
            corr = correlation(window.respiration, window.thoracic_strain)
            contact = float(window.sensor_contact.mean())
            rows.append(
                {
                    "subject": int(subject),
                    "start": start,
                    "end": start + 60,
                    "root": root,
                    "family": family,
                    "amp": resp,
                    "effort": effort,
                    "amp_ref": resp_ref,
                    "amp_sd": resp_sd,
                    "effort_ref": effort_ref,
                    "effort_sd": effort_sd,
                    "amp_rel": resp / resp_ref,
                    "amp_drop": (resp_ref - resp) / resp_sd,
                    "effort_rel": effort / effort_ref,
                    "effort_drop": (effort_ref - effort) / effort_sd,
                    "desat": saturation_ref - float(window.spo2.min()),
                    "hr_rise": float(window.heart_rate.max()) - heart_ref,
                    "contact_loss": 1 - contact,
                    "contact_drop": contact_ref - contact,
                    "q_z": (float(window.textile_strain.mean()) - textile_mean) / textile_sd,
                    "q_iqr_rel": float(
                        window.textile_strain.quantile(0.75) - window.textile_strain.quantile(0.25)
                    )
                    / max(textile_iqr, 1e-6),
                    "corr": corr,
                    "corr_drop": corr_ref - corr,
                }
            )
    return pd.DataFrame(rows), episodes


def mechanism_scores(frame: pd.DataFrame, tolerances: dict | None = None):
    t = TOLERANCES if tolerances is None else TOLERANCES | tolerances
    physiology = (
        np.maximum(0, frame.amp_drop.to_numpy())
        + 0.9 * np.maximum(0, frame.desat.to_numpy() - t["d"])
        + 0.15 * np.maximum(0, frame.hr_rise.to_numpy() - t["h"])
    )
    textile = (
        3 * np.maximum(0, frame.contact_drop.to_numpy() - t["c"])
        + 0.7 * np.maximum(0, np.abs(frame.q_z.to_numpy()) - t["q"])
        + 0.9 * np.maximum(0, t["rho"] - frame["corr"].to_numpy())
    )
    risk = (
        np.maximum(0, frame.contact_drop.to_numpy())
        + 0.5 * np.maximum(0, frame.corr_drop.to_numpy())
        + 0.1 * np.abs(frame.q_z.to_numpy())
    )
    return physiology, textile, risk


def quantile_limits(values) -> tuple[float, float]:
    return float(np.quantile(values, 0.01)), float(np.quantile(values, 0.99))


def normalize(values, limits):
    lower, upper = limits
    return np.clip((values - lower) / max(upper - lower, 1e-9), 0, 1)


def training_scales(training: pd.DataFrame):
    physiology, textile, risk = mechanism_scores(training)
    return (
        quantile_limits(0.55 * physiology + 0.45 * textile),
        quantile_limits(risk),
        quantile_limits(np.maximum(physiology, textile)),
    )


def entropy(prob):
    return -(
        prob * np.log(np.maximum(prob, 1e-12)) + (1 - prob) * np.log(np.maximum(1 - prob, 1e-12))
    ) / np.log(2)


def score_windows(frame, probability, scales, controller, tolerances=None):
    """Fuse logistic and mechanistic evidence and assign an alert family."""
    phys, textile, risk = mechanism_scores(frame, tolerances)
    mixture = normalize(0.55 * phys + 0.45 * textile, scales[0])
    adverse = normalize(risk, scales[1])
    stronger = normalize(np.maximum(phys, textile), scales[2])
    weight, strength, gate = controller
    hybrid = weight * mixture + (1 - weight) * probability
    margin = np.abs(phys - textile) / (phys + textile + 1e-9)
    uncertainty = entropy(probability) + 0.45 * (1 - margin)
    correction = strength * expit(5 * (uncertainty - gate))
    evidence = 0.5 * adverse + 0.25 * mixture + 0.25 * stronger
    score = hybrid + correction * (evidence - hybrid)
    diagnostics = {
        "phys": phys,
        "tex": textile,
        "hybrid": hybrid,
        "alpha": correction,
        "heff": uncertainty,
        "evidence": evidence,
    }
    return score, np.where(phys >= textile, 1, 2), diagnostics


def episode_delays(frame, actions, episodes):
    """Delay as a fraction of 180 seconds. An undetected episode incurs the cap."""
    subjects, ends = frame.subject.to_numpy(), frame.end.to_numpy()
    delays = []
    for event in episodes:
        onset = event["onset"]
        eligible = (
            (subjects == event["subject"])
            & (ends > onset)
            & (ends <= min(event["offset"], onset + MAX_DELAY))
            & (actions == 1)
        )
        hits = ends[eligible]
        delays.append(min(float(hits.min() - onset), MAX_DELAY) / MAX_DELAY if len(hits) else 1.0)
    return np.asarray(delays)


def conditional_rate(errors, mask) -> float:
    return float(np.mean(errors[mask])) if np.any(mask) else 0.0


def loss_terms(frame, actions, episodes):
    family = frame.family.to_numpy()
    delays = episode_delays(frame, actions, episodes)
    return np.array(
        [
            conditional_rate(actions != 0, family == 0) / 3,
            conditional_rate(actions != 1, family == 1) / 3,
            conditional_rate(actions == 1, family == 2) / 3,
            conditional_rate(actions == 0, family == 2) / 3,
            float(delays.mean()) if len(delays) else 0.0,
        ]
    )


def loss_weights(costs):
    missed, textile_alarm, delay = costs
    return np.array([1.0, missed, textile_alarm, 0.5, delay])


def synthetic_metrics(frame, episodes):
    truth, actions = frame.family.to_numpy(), frame.pred.to_numpy()
    terms = loss_terms(frame, actions, episodes)
    return {
        "n": len(frame),
        "F1": float(f1_score(truth > 0, actions > 0, zero_division=0)),
        "AUROC": float(roc_auc_score(truth > 0, frame.score)),
        "family_accuracy": float(np.mean(truth == actions)),
        "normal_alarm": float(terms[0] * 3),
        "phys_miss": float(terms[1] * 3),
        "textile_to_phys": float(terms[2] * 3),
        "textile_miss": float(terms[3] * 3),
        "delay_seconds": float(episode_delays(frame, actions, episodes).mean() * MAX_DELAY),
        "cost": float(terms @ loss_weights(REFERENCE_COSTS)),
    }


def threshold_grid(scores, lower=-1e-6, upper=1.000001):
    return np.unique(np.r_[lower, np.quantile(scores, QUANTILES), upper])


def validation_grid(frame, probability, scales, episodes, controllers):
    terms, choices = [], []
    for controller in controllers:
        scores, family, _ = score_windows(frame, probability, scales, controller)
        for threshold in threshold_grid(scores):
            terms.append(loss_terms(frame, np.where(scores > threshold, family, 0), episodes))
            choices.append((controller, float(threshold)))
    return np.asarray(terms), choices


def prediction_rows(frame, scores, family, threshold):
    rows = frame[WINDOW_KEYS].copy()
    rows["pred"] = np.where(scores > threshold, family, 0)
    rows["score"] = scores
    return rows


def adjusted_references(frame: pd.DataFrame, model: Pipeline, shift: float, mode: str):
    """Update the amplitude references after the current window has been scored."""
    changed = frame.copy()
    progress = (changed.start.to_numpy() - 300) / (1800 - 360)
    factor = 1 - shift * np.clip(progress, 0, 1)
    changed["amp"] = changed.amp * factor
    changed["effort"] = changed.effort * factor
    initial_resp = resp = float(changed.amp_ref.iloc[0])
    initial_effort = effort = float(changed.effort_ref.iloc[0])
    scaler, logistic = model.steps[0][1], model.steps[1][1]
    rows, updates = [], 0
    for _, window in changed.iterrows():
        window = window.copy()
        window["amp_rel"] = window.amp / resp
        window["amp_drop"] = (resp - window.amp) / window.amp_sd
        window["effort_rel"] = window.effort / effort
        window["effort_drop"] = (effort - window.effort) / window.effort_sd
        values = window[FEATURES].to_numpy(dtype=float)
        prob = float(
            expit(
                ((values - scaler.mean_) / scaler.scale_) @ logistic.coef_[0]
                + logistic.intercept_[0]
            )
        )
        trusted = (
            prob < 0.2
            and window.contact_drop < 0.06
            and window["corr"] > 0.6
            and window.desat < 0.8
            and window.hr_rise < 4
            and abs(window.q_z) < 2
        )
        rows.append(window)
        if mode == "unrestricted" or (mode == "guarded" and trusted):
            resp = float(
                np.clip(0.95 * resp + 0.05 * window.amp, 0.7 * initial_resp, 1.3 * initial_resp)
            )
            effort = float(
                np.clip(
                    0.95 * effort + 0.05 * window.effort, 0.7 * initial_effort, 1.3 * initial_effort
                )
            )
            updates += 1
    return pd.DataFrame(rows), updates


def logistic_pipeline() -> Pipeline:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1, max_iter=2000, class_weight="balanced", random_state=SEED),
    )


def rule_score(frame):
    return (
        0.35 * (frame.amp_rel.to_numpy() < 0.6)
        + 0.25 * (frame.desat.to_numpy() > 1.2)
        + 0.2 * (frame.contact_drop.to_numpy() > 0.12)
        + 0.2 * (frame["corr"].to_numpy() < 0.25)
    )


def baseline_predictions(training, validation, test, labelled, val_prob, test_prob, episodes):
    trees = ExtraTreesClassifier(
        n_estimators=128, max_depth=6, class_weight="balanced", random_state=SEED, n_jobs=1
    )
    trees.fit(labelled[FEATURES], labelled.family > 0)
    normal = training.loc[training.family == 0, FEATURES]
    isolation = make_pipeline(
        StandardScaler(), IsolationForest(n_estimators=100, random_state=SEED, n_jobs=1)
    ).fit(normal)
    svm = make_pipeline(StandardScaler(), OneClassSVM(nu=0.08, gamma="scale")).fit(normal)
    scores = {
        "Rules": (rule_score(validation), rule_score(test)),
        "IsolationForest": (
            -isolation.decision_function(validation[FEATURES]),
            -isolation.decision_function(test[FEATURES]),
        ),
        "OneClassSVM": (
            -svm.decision_function(validation[FEATURES]),
            -svm.decision_function(test[FEATURES]),
        ),
        "Logistic": (val_prob, test_prob),
        "ExtraTrees": (
            trees.predict_proba(validation[FEATURES])[:, 1],
            trees.predict_proba(test[FEATURES])[:, 1],
        ),
    }
    val_phys, val_textile, _ = mechanism_scores(validation)
    test_phys, test_textile, _ = mechanism_scores(test)
    val_family = np.where(val_phys >= val_textile, 1, 2)
    test_family = np.where(test_phys >= test_textile, 1, 2)
    for name, (val_scores, test_scores) in scores.items():
        thresholds = threshold_grid(val_scores, val_scores.min() - 1e-6, val_scores.max() + 1e-6)
        losses = [
            loss_terms(validation, np.where(val_scores > h, val_family, 0), episodes)
            @ loss_weights(REFERENCE_COSTS)
            for h in thresholds
        ]
        yield name, prediction_rows(test, test_scores, test_family, thresholds[np.argmin(losses)])


def sensitivity_predictions(
    test, probability, model, scales, controller, threshold, tolerance_runs, shift_runs
):
    for name in (*TOLERANCES, "joint"):
        for scale in (0.75, 1.0, 1.25):
            adjusted = {k: v * scale for k, v in TOLERANCES.items() if name in (k, "joint")}
            score, family, _ = score_windows(test, probability, scales, controller, adjusted)
            tolerance_runs[name, scale].append(prediction_rows(test, score, family, threshold))
    for shift, mode in itertools.product((0.0, 0.2), ("static", "unrestricted", "guarded")):
        adjusted, count = adjusted_references(test, model, shift, mode)
        prob = model.predict_proba(adjusted[FEATURES])[:, 1]
        score, family, _ = score_windows(adjusted, prob, scales, controller)
        rows = prediction_rows(adjusted, score, family, threshold)
        shift_runs[shift, mode].append((rows, count))


def percentile_intervals(samples: list[dict]) -> dict:
    frame = pd.DataFrame(samples)
    return {
        column: {
            "lower": float(frame[column].quantile(0.025)),
            "upper": float(frame[column].quantile(0.975)),
        }
        for column in frame
    }


def synthetic_intervals(hclm, hybrid, episodes):
    """Resample subjects as blocks; an episode belongs to its sampled subject."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = []
    for _ in range(BOOTSTRAP_SAMPLES):
        h_blocks, b_blocks, sampled_events = [], [], []
        for copy_id, subject in enumerate(rng.integers(0, 8, 8)):
            h_blocks.append(hclm[hclm.subject == subject].assign(subject=copy_id))
            b_blocks.append(hybrid[hybrid.subject == subject].assign(subject=copy_id))
            sampled_events.extend(
                dict(event, subject=copy_id) for event in episodes if event["subject"] == subject
            )
        h_sample = pd.concat(h_blocks)
        if sampled_events and h_sample.family.eq(1).any():
            h = synthetic_metrics(h_sample, sampled_events)
            b = synthetic_metrics(pd.concat(b_blocks), sampled_events)
            samples.append(
                {"F1": h["F1"], "cost": h["cost"], "paired_cost_difference": h["cost"] - b["cost"]}
            )
    return percentile_intervals(samples)


def save_cost_slices(output: Path) -> None:
    # Use the serialized operating points, as in the published cost summary.
    cost = pd.read_csv(output / "cost_sensitivity.csv")
    groups = []
    for delay, group in cost.groupby("c_delay"):
        row = {"assumed_c_delay": delay, "scenarios": len(group)}
        for field in ("cost", "phys_miss", "normal_alarm", "delay_seconds"):
            row[field + "_min"] = float(group[field].min())
            row[field + "_max"] = float(group[field].max())
        row["same_nominal_cost_count"] = int(
            np.isclose(group.excess_reference_cost, 0, atol=1e-12).sum()
        )
        groups.append(row)
    write_csv(pd.DataFrame(groups), output, "cost_slices.csv")


def run_synthetic(raw: pd.DataFrame, output: Path) -> dict:
    windows, episodes = extract_windows(raw)
    write_csv(windows, output, "synthetic_features.csv")
    write_csv(pd.DataFrame(episodes), output, "synthetic_physiological_events.csv")
    predictions = {name: [] for name in METHODS}
    cost_runs = {cost: [] for cost in COST_GRID}
    tolerance_runs, shift_runs = defaultdict(list), defaultdict(list)
    diagnostics, folds = [], []
    for held_out in range(8):
        validation_ids = [(held_out + 1) % 8, (held_out + 3) % 8]
        training_ids = [i for i in range(8) if i != held_out and i not in validation_ids]
        training = windows[windows.subject.isin(training_ids)]
        validation = windows[windows.subject.isin(validation_ids)]
        test = windows[windows.subject == held_out]
        val_episodes = [event for event in episodes if event["subject"] in validation_ids]
        labelled = pd.concat(
            [
                group.sample(min(60, len(group)), random_state=1000 + held_out + int(root))
                for root, group in training.groupby("root")
            ]
        )
        model = logistic_pipeline().fit(labelled[FEATURES], labelled.family > 0)
        val_prob = model.predict_proba(validation[FEATURES])[:, 1]
        test_prob = model.predict_proba(test[FEATURES])[:, 1]
        scales = training_scales(training)
        for name, rows in baseline_predictions(
            training, validation, test, labelled, val_prob, test_prob, val_episodes
        ):
            predictions[name].append(rows)

        for name, grid in (("Hybrid", HYBRID_GRID), ("HCLM", CONTROL_GRID)):
            losses, choices = validation_grid(validation, val_prob, scales, val_episodes, grid)
            reference = losses @ loss_weights(REFERENCE_COSTS)
            best = int(np.argmin(reference))
            controller, threshold = choices[best]
            score, family, detail = score_windows(test, test_prob, scales, controller)
            rows = prediction_rows(test, score, family, threshold)
            predictions[name].append(rows)
            if name == "Hybrid":
                continue
            logit = model.named_steps["logisticregression"]
            scaler = model.named_steps["standardscaler"]
            folds.append(
                {
                    "test": held_out,
                    "validation": validation_ids,
                    "training": training_ids,
                    "labelled_windows": len(labelled),
                    "config": controller,
                    "threshold": threshold,
                    "validation_cost": float(reference[best]),
                    "norms": scales,
                    "feature_means": scaler.mean_.tolist(),
                    "feature_scales": scaler.scale_.tolist(),
                    "coef": logit.coef_[0].tolist(),
                    "intercept": float(logit.intercept_[0]),
                }
            )
            diagnostics.append(rows.assign(**detail, h=threshold))
            for costs in COST_GRID:
                selected, h = choices[int(np.argmin(losses @ loss_weights(costs)))]
                score, family, _ = score_windows(test, test_prob, scales, selected)
                cost_runs[costs].append(prediction_rows(test, score, family, h))
            sensitivity_predictions(
                test, test_prob, model, scales, controller, threshold, tolerance_runs, shift_runs
            )
        LOG.info("Synthetic subject %d of 8 complete", held_out + 1)

    pooled = {
        name: pd.concat(parts, ignore_index=True).sort_values(["subject", "start"])
        for name, parts in predictions.items()
    }
    summary = []
    for name, frame in pooled.items():
        write_csv(frame, output, f"predictions_{name}.csv")
        summary.append({"method": name, **synthetic_metrics(frame, episodes)})
    write_csv(pd.DataFrame(summary), output, "synthetic_results.csv")
    hclm = pooled["HCLM"]
    families = ["normal", "phys", "textile"]
    matrix = confusion_matrix(hclm.family, hclm.pred, labels=[0, 1, 2])
    write_csv(
        pd.DataFrame(matrix, index=families, columns=families),
        output,
        "family_confusion.csv",
        index=True,
    )
    write_csv(pd.concat(diagnostics), output, "hclm_diagnostics.csv")

    baseline = synthetic_metrics(pd.concat(cost_runs[REFERENCE_COSTS]), episodes)["cost"]
    rows = []
    for costs, parts in cost_runs.items():
        frame = pd.concat(parts)
        metrics = synthetic_metrics(frame, episodes)
        scenario = float(loss_terms(frame, frame.pred.to_numpy(), episodes) @ loss_weights(costs))
        rows.append(
            dict(
                c_miss=costs[0],
                c_textile_alarm=costs[1],
                c_delay=costs[2],
                **metrics,
                scenario_cost=scenario,
                excess_reference_cost=metrics["cost"] - baseline,
            )
        )
    write_csv(pd.DataFrame(rows), output, "cost_sensitivity.csv")
    save_cost_slices(output)
    tolerance_summary = [
        dict(parameter=name, scale=scale, **synthetic_metrics(pd.concat(parts), episodes))
        for (name, scale), parts in tolerance_runs.items()
    ]
    write_csv(pd.DataFrame(tolerance_summary), output, "threshold_misspecification.csv")
    shifts = []
    for (amount, mode), parts in shift_runs.items():
        frames, counts = zip(*parts)
        shifts.append(
            dict(
                amplitude_shift=amount,
                mode=mode,
                update_count=sum(counts),
                **synthetic_metrics(pd.concat(frames), episodes),
            )
        )
    write_csv(pd.DataFrame(shifts), output, "adaptive_stress.csv")
    return {
        "input_sha256": SYNTHETIC_SHA256,
        "features": FEATURES,
        "folds": folds,
        "conditional_intervals": synthetic_intervals(hclm, pooled["Hybrid"], episodes),
    }


def read_annotations(path: Path) -> list[tuple[int, int]]:
    """Read sample offsets and binary labels from a WFDB apnea annotation file."""
    data = path.read_bytes()
    cursor, sample = 0, 0
    annotations = []
    while cursor + 1 < len(data):
        low, high = data[cursor : cursor + 2]
        cursor += 2
        code, delta = high >> 2, ((high & 3) << 8) | low
        if code == 0 and delta == 0:
            break
        if code == 59:
            # WFDB SKIP stores the high 16-bit word first, both words little-endian.
            if cursor + 4 > len(data):
                raise ValueError(f"Incomplete SKIP in {path.name}")
            jump = int.from_bytes(data[cursor : cursor + 2], "little") << 16
            jump += int.from_bytes(data[cursor + 2 : cursor + 4], "little")
            sample += jump - 2**32 if jump >= 2**31 else jump
            cursor += 4
        elif code == 63:
            cursor += delta + (delta % 2)
        elif code not in (60, 61, 62):
            sample += delta
            if code in (1, 8):
                annotations.append((sample, int(code == 8)))
    return annotations


def extract_record(folder: Path, record: str):
    header = (folder / f"{record}r.hea").read_text(encoding="ascii").splitlines()
    fields = header[0].split()
    if int(fields[1]) != 4 or int(fields[2]) != 100:
        raise ValueError(f"Unexpected respiratory signal layout for {record}")
    sample_count = int(fields[3])
    signal = np.memmap(folder / f"{record}r.dat", dtype="<i2", mode="r").reshape(sample_count, 4)
    gain = np.array([float(line.split()[2]) for line in header[1:5]])
    labels = read_annotations(folder / f"{record}.apn")
    if len(labels) <= 300 or any(offset % 6000 for offset, _ in labels):
        raise ValueError(f"Unexpected minute annotations for {record}")
    rows, rejected = [], 0
    for offset, label in labels:
        if offset + 6000 > sample_count:
            continue
        minute = np.asarray(signal[offset : offset + 6000 : 5], float) / gain
        saturation = minute[:, 3]
        valid = (saturation >= 50) & (saturation <= 100)
        if valid.mean() < 0.8:
            rejected += 1
            continue
        saturation = saturation[valid]
        # Ten six-second segments at 20 Hz, in chest, abdominal and airflow order.
        amps = np.array(
            [[amplitude(minute[j : j + 120, k]) for k in (0, 1, 2)] for j in range(0, 1200, 120)]
        )
        chest, abdomen, air = np.median(amps, axis=0)
        rows.append(
            {
                "record": record,
                "minute": offset // 6000,
                "label": label,
                "chest": chest,
                "abd": abdomen,
                "air": air,
                "air_low": float(np.quantile(amps[:, 2], 0.1)),
                "air_cv": float(np.std(amps[:, 2]) / max(np.mean(amps[:, 2]), 1e-6)),
                "sat_mean": float(saturation.mean()),
                "sat_min": float(saturation.min()),
                "sat_range": float(saturation.max() - saturation.min()),
            }
        )
    frame = pd.DataFrame(rows)
    warmup = frame.iloc[:5]
    if len(warmup) < 5:
        raise ValueError(f"Fewer than five valid commissioning minutes in {record}")
    for channel in ("air", "chest", "abd"):
        reference = max(float(warmup[channel].quantile(0.9)), 1e-6)
        frame[channel + "_rel"] = frame[channel] / reference
        if channel == "air":
            frame["air_low_rel"] = frame.air_low / reference
    frame["desat"] = float(warmup.sat_mean.max()) - frame.sat_min
    frame["phys"] = np.maximum(0, (1 - frame.air_rel) / 0.15) + 0.9 * np.maximum(
        0, frame.desat - 0.8
    )
    evaluated = frame.iloc[5:].copy()
    counts = {
        "record": record,
        "evaluated_minutes": len(evaluated),
        "apnea_minutes": int(evaluated.label.sum()),
        "excluded_bad_spo2_minutes": rejected,
        "warmup_end_minute": int(warmup.minute.iloc[-1]),
    }
    return evaluated, counts


def respiratory_score(probability, evidence, controller):
    weight, strength, gate = controller
    hybrid = weight * evidence + (1 - weight) * probability
    return hybrid + strength * expit(5 * (entropy(probability) - gate)) * (evidence - hybrid)


def respiratory_cost(labels, prediction):
    return 8 * np.mean(prediction[labels == 1] == 0) + np.mean(prediction[labels == 0])


def select_respiratory(controllers, probability, evidence, labels):
    """Choose the first minimum on the ordered policy and threshold grid."""
    best_loss, best_controller, best_threshold = float("inf"), None, None
    for controller in controllers:
        scores = respiratory_score(probability, evidence, controller)
        for threshold in threshold_grid(scores):
            loss = respiratory_cost(labels, scores > threshold)
            if loss < best_loss:
                best_loss, best_controller, best_threshold = (
                    float(loss),
                    controller,
                    float(threshold),
                )
    return best_loss, best_controller, best_threshold


def respiratory_intervals(hclm, logistic):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = []
    for _ in range(BOOTSTRAP_SAMPLES):
        selected = rng.choice(RECORDS, len(RECORDS), replace=True)
        current = pd.concat([hclm[hclm.record == r] for r in selected])
        baseline = pd.concat([logistic[logistic.record == r] for r in selected])
        labels = current.label.to_numpy()
        if len(np.unique(labels)) < 2:
            continue
        cost = respiratory_cost(labels, current.pred.to_numpy())
        samples.append(
            {
                "F1": float(f1_score(labels, current.pred)),
                "AUROC": float(roc_auc_score(labels, current.score)),
                "cost": float(cost),
                "paired_cost_difference": float(
                    cost - respiratory_cost(labels, baseline.pred.to_numpy())
                ),
            }
        )
    return percentile_intervals(samples)


def record_tables(output: Path, folds: list[dict]) -> None:
    predictions = pd.read_csv(output / "external_predictions_HCLM.csv")
    errors, composition = [], []
    for fold in folds:
        record, val = fold["test_record"], fold["validation_records"]
        train = fold["training_records"]
        if record in val + train or set(val) & set(train):
            raise ValueError(f"Overlapping record assignments for {record}")
        test = predictions[predictions.record == record]
        validation = predictions[predictions.record.isin(val)]
        np.testing.assert_array_equal(test.pred, test.score > fold["threshold"])
        truth, pred = test.label.to_numpy(), test.pred.to_numpy()
        tp = int(((truth == 1) & (pred == 1)).sum())
        fp = int(((truth == 0) & (pred == 1)).sum())
        fn = int(((truth == 1) & (pred == 0)).sum())
        tn = int(((truth == 0) & (pred == 0)).sum())
        errors.append(
            {
                "record": record,
                "n": len(test),
                "positive": tp + fn,
                "negative": tn + fp,
                "tp": tp,
                "fn": fn,
                "fp": fp,
                "tn": tn,
                "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
                "specificity": tn / (tn + fp) if tn + fp else np.nan,
            }
        )
        composition.append(
            {
                "test_record": record,
                "validation_records": "+".join(val),
                "validation_positive": int(validation.label.sum()),
                "validation_total": len(validation),
                "test_positive": int(test.label.sum()),
                "test_total": len(test),
                "selected_threshold": fold["threshold"],
                "controller_strength": fold["config"].strength,
                "test_false_positive": fp,
                "test_false_negative": fn,
            }
        )
    write_csv(pd.DataFrame(errors), output, "external_record_errors.csv")
    write_csv(pd.DataFrame(composition), output, "validation_split_composition.csv")


def run_respiratory(folder: Path, output: Path) -> dict:
    frames, counts = [], []
    for record in RECORDS:
        minutes, accounting = extract_record(folder, record)
        frames.append(minutes)
        counts.append(accounting)
    data = pd.concat(frames, ignore_index=True)
    write_csv(data, output, "apnea_ecg_minute_features.csv")
    write_csv(pd.DataFrame(counts), output, "external_record_counts.csv")
    predictions = {name: [] for name in RESPIRATORY_METHODS}
    policies = {
        "Logistic": [Controller(0.0, 0.0, 0.45)],
        "Mechanistic": [Controller(1.0, 0.0, 0.45)],
        "Hybrid": HYBRID_GRID,
        "HCLM": CONTROL_GRID,
    }
    folds = []
    for index, record in enumerate(RECORDS):
        ab = [r for r in RECORDS if r[0] in "ab" and r != record]
        controls = [r for r in RECORDS if r[0] == "c" and r != record]
        val_ids = [ab[index % len(ab)], controls[index % len(controls)]]
        train_ids = [r for r in RECORDS if r != record and r not in val_ids]
        training = data[data.record.isin(train_ids)]
        validation = data[data.record.isin(val_ids)]
        test = data[data.record == record]
        model = logistic_pipeline().fit(training[RESPIRATORY_FEATURES], training.label)
        val_prob = model.predict_proba(validation[RESPIRATORY_FEATURES])[:, 1]
        test_prob = model.predict_proba(test[RESPIRATORY_FEATURES])[:, 1]
        limits = quantile_limits(training.phys)
        val_evidence = normalize(validation.phys.to_numpy(), limits)
        test_evidence = normalize(test.phys.to_numpy(), limits)
        for name, grid in policies.items():
            loss, controller, threshold = select_respiratory(
                grid, val_prob, val_evidence, validation.label.to_numpy()
            )
            score = respiratory_score(test_prob, test_evidence, controller)
            result = test[["record", "minute", "label"]].copy()
            result["score"] = score
            result["pred"] = (score > threshold).astype(int)
            predictions[name].append(result)
            if name == "HCLM":
                folds.append(
                    {
                        "test_record": record,
                        "validation_records": val_ids,
                        "training_records": train_ids,
                        "config": controller,
                        "threshold": threshold,
                        "validation_cost": loss,
                    }
                )
        LOG.info("Respiratory record %s complete", record)

    pooled = {name: pd.concat(segments) for name, segments in predictions.items()}
    summary = []
    for name, frame in pooled.items():
        write_csv(frame, output, f"external_predictions_{name}.csv")
        labels, pred = frame.label.to_numpy(), frame.pred.to_numpy()
        summary.append(
            {
                "method": name,
                "n": len(frame),
                "F1": float(f1_score(labels, pred)),
                "AUROC": float(roc_auc_score(labels, frame.score)),
                "sensitivity": float(np.mean(pred[labels == 1])),
                "specificity": float(np.mean(pred[labels == 0] == 0)),
                "cost": float(respiratory_cost(labels, pred)),
            }
        )
    write_csv(pd.DataFrame(summary), output, "external_results.csv")
    record_tables(output, folds)
    return {
        "source": APNEA_SOURCE,
        "source_sha256": RECORD_HASHES,
        "features": RESPIRATORY_FEATURES,
        "folds": folds,
        "conditional_intervals": respiratory_intervals(pooled["HCLM"], pooled["Logistic"]),
    }


def check_tables(output: Path, studies: tuple[str, ...]) -> tuple[dict[str, str], list[str]]:
    expected = {
        filename: digest for study in studies for filename, digest in TABLE_HASHES[study].items()
    }
    actual = {file.name: file_hash(file) for file in sorted(output.glob("*.csv"))}
    if actual.keys() != expected.keys():
        raise ValueError(
            f"Unexpected set of output tables: {sorted(actual.keys() ^ expected.keys())}"
        )
    return actual, [name for name in actual if actual[name] != expected[name]]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reproduce the HCLM-SmartTex benchmark and Apnea-ECG tables.",
        epilog="Each run creates a new results directory. Synthetic observations are in data/synthetic/.",
    )
    parser.add_argument(
        "--study",
        choices=("all", "synthetic", "respiratory"),
        default="all",
        help="Select the evaluation (default: all).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results"),
        help="Parent directory for completed runs (default: results).",
    )
    parser.add_argument(
        "--public-data",
        type=Path,
        metavar="DIR",
        help="Read the 24 existing, checksum-verified Apnea-ECG files from DIR.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".hclm-cache") / "apnea-ecg-1.0.0",
        help="Cache for waveform downloads when --public-data is omitted.",
    )
    parser.add_argument(
        "--synthetic",
        type=Path,
        metavar="CSV",
        help="Read the original single CSV in place of the bundled data/synthetic/ files.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return a nonzero status if any CSV differs from its reference SHA-256.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    studies = ("synthetic", "respiratory") if args.study == "all" else (args.study,)
    manifest = {
        "study": args.study,
        "seed": SEED,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_SAMPLES,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__,
        },
    }
    try:
        raw = load_benchmark(args.synthetic) if "synthetic" in studies else None
        public = (
            prepare_recordings(args.cache_dir, args.public_data)
            if "respiratory" in studies
            else None
        )
        root = args.output.resolve()
        root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".hclm-", dir=root) as tmp:
            stage = Path(tmp) / "tables"
            stage.mkdir()
            if raw is not None:
                manifest["synthetic"] = run_synthetic(raw, stage)
            if public is not None:
                manifest["respiratory"] = run_respiratory(public, stage)
            hashes, differences = check_tables(stage, studies)
            manifest["csv_sha256"] = hashes
            manifest["csv_reference_mismatches"] = differences
            (stage / "run_manifest.json").write_text(
                json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            result = root / f"{args.study}-{stamp}-{Path(tmp).name.removeprefix('.hclm-')}"
            stage.rename(result)
    except (ValueError, OSError, EOFError, AssertionError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")
    print(f"{len(hashes)} CSV tables: {result}")
    if differences:
        LOG.warning("Different reference hashes (%d): %s", len(differences), ", ".join(differences))
        return 1 if args.strict else 0
    print("All CSVs match their reference SHA-256.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
