from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_bits_module(path: str | Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cannot find BITS script: {path}")
    spec = importlib.util.spec_from_file_location("bits_v6", str(path))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None

    import sys
    sys.modules[spec.name] = mod

    spec.loader.exec_module(mod)
    return mod


def clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def safe_div(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return a / (b + eps)


def load_npz(npz_path: str | Path) -> Dict[str, Any]:
    data = np.load(npz_path)
    required = ["b01", "b02", "b03", "b04", "b05", "b06", "b07", "state_1km", "burn_2021"]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"NPZ missing required keys: {missing}")

    out = {}
    for k in ["b01", "b02", "b03", "b04", "b05", "b06", "b07"]:
        arr = data[k].astype(np.float32)
        if np.nanmax(arr) > 2.0:
            arr = arr * 0.0001
        out[k] = clip01(arr)

    out["state_1km"] = data["state_1km"].astype(np.uint16)
    out["burn_2021"] = data["burn_2021"].astype(np.int32)
    out["dates"] = [str(x) for x in data["dates"].tolist()] if "dates" in data else [str(i) for i in range(out["b01"].shape[0])]
    out["window_labels"] = [str(x) for x in data["window_labels"].tolist()] if "window_labels" in data else [f"t{i:02d}" for i in range(out["b01"].shape[0])]
    return out


def valid_mask(data: Dict[str, Any], t: int) -> np.ndarray:
    stack = np.stack([data[f"b{i:02d}"][t] for i in range(1, 8)], axis=0)
    return np.isfinite(stack).all(axis=0) & (np.abs(stack).sum(axis=0) > 1e-6)


def rgb_from_bands(bands: Dict[str, np.ndarray]) -> np.ndarray:
    return np.dstack([clip01(bands["b01"]), clip01(bands["b04"]), clip01(bands["b03"])])


def grayscale(rgb: np.ndarray) -> np.ndarray:
    return 0.2989 * rgb[:, :, 0] + 0.5870 * rgb[:, :, 1] + 0.1140 * rgb[:, :, 2]


def image_entropy(gray: np.ndarray, mask: np.ndarray, bins: int = 128) -> float:
    vals = gray[mask & np.isfinite(gray)]
    if vals.size == 0:
        return float("nan")
    hist, _ = np.histogram(vals, bins=bins, range=(0, 1), density=False)
    p = hist.astype(np.float64)
    p = p[p > 0] / max(p.sum(), 1)
    return float(-(p * np.log2(p)).sum())


def local_mean_3x3(x: np.ndarray) -> np.ndarray:
    xp = np.pad(x, 1, mode="edge")
    return (
        xp[:-2, :-2] + xp[:-2, 1:-1] + xp[:-2, 2:] +
        xp[1:-1, :-2] + xp[1:-1, 1:-1] + xp[1:-1, 2:] +
        xp[2:, :-2] + xp[2:, 1:-1] + xp[2:, 2:]
    ) / 9.0


def local_std_3x3(x: np.ndarray) -> np.ndarray:
    m = local_mean_3x3(x)
    m2 = local_mean_3x3(x * x)
    return np.sqrt(np.maximum(m2 - m * m, 0.0))


def local_contrast(gray: np.ndarray, mask: np.ndarray) -> float:
    lc = local_std_3x3(gray)
    return float(np.nanmean(lc[mask])) if mask.any() else float("nan")


def laplacian_variance(gray: np.ndarray, mask: np.ndarray) -> float:
    xp = np.pad(gray, 1, mode="edge")
    lap = (
        -4 * xp[1:-1, 1:-1] +
        xp[:-2, 1:-1] + xp[2:, 1:-1] + xp[1:-1, :-2] + xp[1:-1, 2:]
    )
    return float(np.nanvar(lap[mask])) if mask.any() else float("nan")


def ndvi(bands: Dict[str, np.ndarray]) -> np.ndarray:
    return safe_div(bands["b02"] - bands["b01"], bands["b02"] + bands["b01"])


def nbr(bands: Dict[str, np.ndarray]) -> np.ndarray:
    return safe_div(bands["b02"] - bands["b07"], bands["b02"] + bands["b07"])


def metric_mean(x: np.ndarray, mask: np.ndarray) -> float:
    vals = x[mask & np.isfinite(x)]
    return float(np.nanmean(vals)) if vals.size else float("nan")


def hamming_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(a != b))


def binary_bundle(hvs: List[np.ndarray]) -> np.ndarray:
    if not hvs:
        raise ValueError("No hypervectors to bundle")
    stack = np.stack(hvs, axis=0)
    votes = stack.sum(axis=0)
    return (votes >= (len(hvs) / 2)).astype(np.uint8)


def tile_burn_labels(burn: np.ndarray, tile_positions: List[Tuple[int, int]], shape: Tuple[int, int], tile_size: int, threshold: float) -> Dict[Tuple[int, int], Dict[str, float]]:
    labels = {}
    burned = burn > 0
    h, w = shape
    for pos in tile_positions:
        r, c = pos
        rs = slice(r * tile_size, min((r + 1) * tile_size, h))
        cs = slice(c * tile_size, min((c + 1) * tile_size, w))
        frac = float(np.mean(burned[rs, cs]))
        labels[pos] = {"burn_fraction": frac, "burn_label": int(frac >= threshold)}
    return labels


def confusion_counts(y_true: List[int], y_pred: List[int]) -> Dict[str, int]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def f1_from_counts(c: Dict[str, int]) -> float:
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return float(2 * precision * recall / max(precision + recall, 1e-12))


def accuracy_from_counts(c: Dict[str, int]) -> float:
    total = c["tp"] + c["tn"] + c["fp"] + c["fn"]
    return float((c["tp"] + c["tn"]) / max(total, 1))


def best_threshold(y_true: List[int], scores: List[float]) -> Dict[str, Any]:
    if not scores:
        return {}
    lo, hi = min(scores), max(scores)
    thresholds = np.linspace(lo, hi, 101)
    best = None
    for th in thresholds:
        pred = [int(s >= th) for s in scores]
        c = confusion_counts(y_true, pred)
        f1 = f1_from_counts(c)
        acc = accuracy_from_counts(c)
        item = {"threshold": float(th), "f1": f1, "accuracy": acc, **c}
        if best is None or (item["f1"], item["accuracy"]) > (best["f1"], best["accuracy"]):
            best = item
    return best or {}


def compute_before_after_hdc_metrics(bits, data: Dict[str, Any], cleaner, labels: List[str], eval_label: str, burn_threshold: float, D: int, tile_size: int) -> Dict[str, Any]:
    cfg = bits.NBAIConfig(D=D, tile_size=tile_size)
    nbai_before = bits.NBAILandSignature(cfg)
    nbai_after = bits.NBAILandSignature(cfg)

    label_to_index = {lab: i for i, lab in enumerate(data["window_labels"])}

    pre_indices = [i for i, lab in enumerate(data["window_labels"]) if lab.startswith("pre_")]
    if not pre_indices:
        pre_indices = [0]

    before_pre_tile_hvs = []
    after_pre_tile_hvs = []
    for t in pre_indices:
        bands = {f"b{i:02d}": data[f"b{i:02d}"][t] for i in range(1, 8)}
        enc_before = nbai_before.encode_scene(bands)
        cleaned = cleaner.clean_bands(bands, data["state_1km"][t])["clean_bands"]
        enc_after = nbai_after.encode_scene(cleaned)
        before_pre_tile_hvs.append(enc_before["tile_hvs"])
        after_pre_tile_hvs.append(enc_after["tile_hvs"])

    positions = sorted(before_pre_tile_hvs[0].keys())
    baseline_before = {pos: binary_bundle([d[pos] for d in before_pre_tile_hvs]) for pos in positions}
    baseline_after = {pos: binary_bundle([d[pos] for d in after_pre_tile_hvs]) for pos in positions}

    if eval_label not in label_to_index:
        raise ValueError(f"eval_label {eval_label} not found. Available: {data['window_labels']}")
    t_eval = label_to_index[eval_label]
    bands_eval = {f"b{i:02d}": data[f"b{i:02d}"][t_eval] for i in range(1, 8)}
    enc_before_eval = nbai_before.encode_scene(bands_eval)
    cleaned_eval = cleaner.clean_bands(bands_eval, data["state_1km"][t_eval])["clean_bands"]
    enc_after_eval = nbai_after.encode_scene(cleaned_eval)

    tile_labels = tile_burn_labels(data["burn_2021"], positions, data["b01"][0].shape, tile_size, burn_threshold)

    rows = []
    for pos in positions:
        score_before = hamming_distance(enc_before_eval["tile_hvs"][pos], baseline_before[pos])
        score_after = hamming_distance(enc_after_eval["tile_hvs"][pos], baseline_after[pos])
        info = tile_labels[pos]
        rows.append({
            "pos_row": pos[0],
            "pos_col": pos[1],
            "burn_fraction": info["burn_fraction"],
            "burn_label": info["burn_label"],
            "score_before_sid": score_before,
            "score_after_sid": score_after,
        })

    df = pd.DataFrame(rows)
    y = df["burn_label"].astype(int).tolist()
    before_scores = df["score_before_sid"].astype(float).tolist()
    after_scores = df["score_after_sid"].astype(float).tolist()

    before_best = best_threshold(y, before_scores)
    after_best = best_threshold(y, after_scores)

    burned = df[df["burn_label"] == 1]
    unburned = df[df["burn_label"] == 0]

    summary = {
        "eval_label": eval_label,
        "eval_date": data["dates"][t_eval],
        "n_tiles": int(len(df)),
        "n_burned_tiles": int(df["burn_label"].sum()),
        "before_sid": {
            "mean_burned_score": float(burned["score_before_sid"].mean()),
            "mean_unburned_score": float(unburned["score_before_sid"].mean()),
            "burned_unburned_ratio": float(burned["score_before_sid"].mean() / max(unburned["score_before_sid"].mean(), 1e-12)),
            "best_threshold_metrics": before_best,
        },
        "after_sid": {
            "mean_burned_score": float(burned["score_after_sid"].mean()),
            "mean_unburned_score": float(unburned["score_after_sid"].mean()),
            "burned_unburned_ratio": float(burned["score_after_sid"].mean() / max(unburned["score_after_sid"].mean(), 1e-12)),
            "best_threshold_metrics": after_best,
        },
    }
    summary["separability_gain_ratio"] = summary["after_sid"]["burned_unburned_ratio"] / max(summary["before_sid"]["burned_unburned_ratio"], 1e-12)
    summary["tile_scores"] = df
    return summary


def compute_temporal_stability(bits, data: Dict[str, Any], cleaner, D: int, tile_size: int, burn_threshold: float) -> Dict[str, Any]:
    cfg = bits.NBAIConfig(D=D, tile_size=tile_size)
    nbai_before = bits.NBAILandSignature(cfg)
    nbai_after = bits.NBAILandSignature(cfg)

    pre_indices = [i for i, lab in enumerate(data["window_labels"]) if lab.startswith("pre_")]
    if not pre_indices:
        pre_indices = [0]

    before_pre = []
    after_pre = []
    for t in pre_indices:
        bands = {f"b{i:02d}": data[f"b{i:02d}"][t] for i in range(1, 8)}
        before_pre.append(nbai_before.encode_scene(bands)["tile_hvs"])
        cleaned = cleaner.clean_bands(bands, data["state_1km"][t])["clean_bands"]
        after_pre.append(nbai_after.encode_scene(cleaned)["tile_hvs"])

    positions = sorted(before_pre[0].keys())
    baseline_before = {pos: binary_bundle([d[pos] for d in before_pre]) for pos in positions}
    baseline_after = {pos: binary_bundle([d[pos] for d in after_pre]) for pos in positions}

    tile_labels = tile_burn_labels(data["burn_2021"], positions, data["b01"][0].shape, tile_size, burn_threshold)
    unburned_positions = [pos for pos in positions if tile_labels[pos]["burn_label"] == 0]

    rows = []
    for t, lab in enumerate(data["window_labels"]):
        bands = {f"b{i:02d}": data[f"b{i:02d}"][t] for i in range(1, 8)}
        enc_b = nbai_before.encode_scene(bands)
        cleaned = cleaner.clean_bands(bands, data["state_1km"][t])["clean_bands"]
        enc_a = nbai_after.encode_scene(cleaned)
        for pos in unburned_positions:
            rows.append({
                "window_label": lab,
                "date": data["dates"][t],
                "pos_row": pos[0],
                "pos_col": pos[1],
                "unburned_score_before_sid": hamming_distance(enc_b["tile_hvs"][pos], baseline_before[pos]),
                "unburned_score_after_sid": hamming_distance(enc_a["tile_hvs"][pos], baseline_after[pos]),
            })
    df = pd.DataFrame(rows)
    return {
        "unburned_temporal_std_before_sid": float(df["unburned_score_before_sid"].std()),
        "unburned_temporal_std_after_sid": float(df["unburned_score_after_sid"].std()),
        "unburned_temporal_std_reduction_ratio": float(df["unburned_score_before_sid"].std() / max(df["unburned_score_after_sid"].std(), 1e-12)),
        "temporal_rows": df,
    }


def compute_image_metrics_for_label(bits, data: Dict[str, Any], cleaner, label: str) -> Dict[str, Any]:
    idx = data["window_labels"].index(label)
    date = data["dates"][idx]
    bands = {f"b{i:02d}": data[f"b{i:02d}"][idx] for i in range(1, 8)}
    state = data["state_1km"][idx]
    valid = valid_mask(data, idx)

    l1 = cleaner.clean_bands(bands, state)
    cleaned = l1["clean_bands"]

    rgb_in = rgb_from_bands(bands)
    rgb_out = rgb_from_bands(cleaned)
    gray_in = grayscale(rgb_in)
    gray_out = grayscale(rgb_out)

    ndvi_in = ndvi(bands)
    ndvi_out = ndvi(cleaned)
    nbr_in = nbr(bands)
    nbr_out = nbr(cleaned)

    metrics = {
        "label": label,
        "date": date,
        "quality_class": l1["quality_class"],
        "quality_score": float(l1["quality_score"]),
        "valid_fraction": float(valid.mean()),
        "entropy_before": image_entropy(gray_in, valid),
        "entropy_after": image_entropy(gray_out, valid),
        "entropy_delta": image_entropy(gray_out, valid) - image_entropy(gray_in, valid),
        "local_contrast_before": local_contrast(gray_in, valid),
        "local_contrast_after": local_contrast(gray_out, valid),
        "local_contrast_delta": local_contrast(gray_out, valid) - local_contrast(gray_in, valid),
        "sharpness_before": laplacian_variance(gray_in, valid),
        "sharpness_after": laplacian_variance(gray_out, valid),
        "sharpness_delta": laplacian_variance(gray_out, valid) - laplacian_variance(gray_in, valid),
        "mean_abs_rgb_enhancement": float(np.nanmean(np.abs(rgb_out[valid] - rgb_in[valid]))) if valid.any() else float("nan"),
        "mean_ndvi_before": metric_mean(ndvi_in, valid),
        "mean_ndvi_after": metric_mean(ndvi_out, valid),
        "mean_abs_delta_ndvi": metric_mean(np.abs(ndvi_out - ndvi_in), valid),
        "mean_nbr_before": metric_mean(nbr_in, valid),
        "mean_nbr_after": metric_mean(nbr_out, valid),
        "mean_abs_delta_nbr": metric_mean(np.abs(nbr_out - nbr_in), valid),
    }

    return metrics


def plot_metric_bar(summary_df: pd.DataFrame, outdir: Path) -> Path:
    labels = summary_df["label"].tolist()
    x = np.arange(len(labels))
    width = 0.35

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].bar(x - width/2, summary_df["local_contrast_before"], width, label="before")
    axes[0, 0].bar(x + width/2, summary_df["local_contrast_after"], width, label="after")
    axes[0, 0].set_title("Local contrast")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(labels, rotation=25, ha="right")
    axes[0, 0].legend()

    axes[0, 1].bar(x - width/2, summary_df["sharpness_before"], width, label="before")
    axes[0, 1].bar(x + width/2, summary_df["sharpness_after"], width, label="after")
    axes[0, 1].set_title("Sharpness / Laplacian variance")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(labels, rotation=25, ha="right")
    axes[0, 1].legend()

    axes[1, 0].bar(x, summary_df["mean_abs_rgb_enhancement"])
    axes[1, 0].set_title("Mean absolute RGB enhancement")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(labels, rotation=25, ha="right")

    axes[1, 1].bar(x, summary_df["mean_abs_delta_ndvi"], label="|ΔNDVI|")
    axes[1, 1].bar(x, summary_df["mean_abs_delta_nbr"], bottom=summary_df["mean_abs_delta_ndvi"], label="|ΔNBR|")
    axes[1, 1].set_title("Spectral-index perturbation")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(labels, rotation=25, ha="right")
    axes[1, 1].legend()

    fig.tight_layout()
    path = outdir / "sid_no_reference_metrics_summary.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", default="dixie_wildfire_dense.npz")
    parser.add_argument("--bits-script", default="bits_sid_nbai_demo_v6_wildfire_fixed.py")
    parser.add_argument("--out", default=r"runs\sid_no_reference_metrics")
    parser.add_argument("--label", default=None, help="Single label to analyze")
    parser.add_argument("--labels", nargs="*", default=None, help="Multiple labels to analyze")
    parser.add_argument("--eval-label", default="fire_202108b", help="Label for burned/unburned HDC F1 before/after SID")
    parser.add_argument("--burn-fraction-threshold", type=float, default=0.20)
    parser.add_argument("--D", type=int, default=4096)
    parser.add_argument("--tile-size", type=int, default=16)
    args = parser.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    bits = load_bits_module(args.bits_script)
    data = load_npz(args.npz)
    cleaner = bits.SID3D4Cleaner(bits.SIDConfig())

    labels = args.labels
    if args.label:
        labels = [args.label]
    if not labels:
        labels = ["fire_202108a", "fire_202108b", "rec_202304", "rec_202404"]

    labels = [lab for lab in labels if lab in data["window_labels"]]
    if not labels:
        raise ValueError("No valid labels selected.")

    image_metrics = [compute_image_metrics_for_label(bits, data, cleaner, lab) for lab in labels]
    image_df = pd.DataFrame(image_metrics)
    image_df.to_csv(outdir / "sid_no_reference_image_metrics.csv", index=False)

    hdc_summary = compute_before_after_hdc_metrics(
        bits, data, cleaner, labels, args.eval_label,
        burn_threshold=args.burn_fraction_threshold, D=args.D, tile_size=args.tile_size
    )
    tile_df = hdc_summary.pop("tile_scores")
    tile_df.to_csv(outdir / "sid_hdc_tile_scores_before_after.csv", index=False)

    stability = compute_temporal_stability(
        bits, data, cleaner, D=args.D, tile_size=args.tile_size,
        burn_threshold=args.burn_fraction_threshold
    )
    temporal_df = stability.pop("temporal_rows")
    temporal_df.to_csv(outdir / "sid_hdc_unburned_temporal_stability.csv", index=False)

    all_summary = {
        "note": "No paired cloud-free ground truth is available; these are no-reference and downstream HDC reliability metrics, not PSNR/SSIM.",
        "selected_labels": labels,
        "image_metrics_csv": "sid_no_reference_image_metrics.csv",
        "hdc_before_after": hdc_summary,
        "temporal_stability": stability,
    }
    (outdir / "sid_no_reference_summary.json").write_text(json.dumps(all_summary, indent=2), encoding="utf-8")

    plot_path = plot_metric_bar(image_df, outdir)

    print("SID no-reference metrics completed.")
    print(f"Output directory: {outdir.resolve()}")
    print("\nImage metrics:")
    print(image_df.to_string(index=False))
    print("\nHDC before/after SID summary:")
    print(json.dumps(hdc_summary, indent=2))
    print("\nTemporal stability summary:")
    print(json.dumps(stability, indent=2))
    print("\nCreated:")
    print(outdir / "sid_no_reference_image_metrics.csv")
    print(outdir / "sid_hdc_tile_scores_before_after.csv")
    print(outdir / "sid_hdc_unburned_temporal_stability.csv")
    print(outdir / "sid_no_reference_summary.json")
    print(plot_path)


if __name__ == "__main__":
    main()