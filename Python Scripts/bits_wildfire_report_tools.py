from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


NASA_DATASET_CITATION = {
    "primary_dataset": "MOD09GA.061",
    "primary_dataset_doi": "10.5067/MODIS/MOD09GA.061",
    "burn_label_dataset": "MCD64A1.061 BurnDate",
}


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def robust_scale(x: np.ndarray, valid: np.ndarray | None = None, p_low: float = 2, p_high: float = 98) -> np.ndarray:
    x = x.astype(np.float32)
    if valid is None:
        valid = np.isfinite(x)
    vals = x[valid & np.isfinite(x)]
    if vals.size == 0:
        return clip01(x)
    lo, hi = np.percentile(vals, [p_low, p_high])
    if abs(hi - lo) < 1e-8:
        return clip01(x)
    return clip01((x - lo) / (hi - lo))


def load_npz_reflectance(npz_path: str | Path) -> Dict[str, Any]:
    data = np.load(npz_path)
    required = ["b01", "b02", "b03", "b04", "b05", "b06", "b07", "state_1km", "burn_2021"]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"Missing keys in NPZ: {missing}")

    out = {k: data[k] for k in required}
    out["dates"] = [str(x) for x in data["dates"].tolist()] if "dates" in data else [str(i) for i in range(out["b01"].shape[0])]
    out["window_labels"] = [str(x) for x in data["window_labels"].tolist()] if "window_labels" in data else [f"t{i:02d}" for i in range(out["b01"].shape[0])]
    out["bad_frac"] = data["bad_frac"].astype(float).tolist() if "bad_frac" in data else [float("nan")] * out["b01"].shape[0]
    out["roi"] = data["roi"].tolist() if "roi" in data else None

    for k in ["b01", "b02", "b03", "b04", "b05", "b06", "b07"]:
        arr = out[k].astype(np.float32)
        if np.nanmax(arr) > 2.0:
            arr = arr * 0.0001
        out[k] = clip01(arr)

    out["state_1km"] = out["state_1km"].astype(np.uint16)
    out["burn_2021"] = out["burn_2021"].astype(np.int32)
    return out


def rgb_from_npz(data: Dict[str, Any], t: int, p_low: float = 2, p_high: float = 98) -> np.ndarray:
    r = data["b01"][t]
    g = data["b04"][t]
    b = data["b03"][t]
    valid = valid_mask_for_time(data, t)
    rgb = np.dstack([
        robust_scale(r, valid, p_low, p_high),
        robust_scale(g, valid, p_low, p_high),
        robust_scale(b, valid, p_low, p_high),
    ])
    rgb[~valid] = 1.0
    return rgb


def valid_mask_for_time(data: Dict[str, Any], t: int) -> np.ndarray:
    stack = np.stack([data[f"b{i:02d}"][t] for i in range(1, 8)], axis=0)
    valid = np.isfinite(stack).all(axis=0) & (np.abs(stack).sum(axis=0) > 1e-6)
    return valid


def crop_to_valid(rgb: np.ndarray, valid: np.ndarray, pad: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(valid)
    if len(xs) == 0 or len(ys) == 0:
        return rgb, valid
    y0, y1 = max(0, ys.min() - pad), min(rgb.shape[0], ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(rgb.shape[1], xs.max() + pad + 1)
    return rgb[y0:y1, x0:x1], valid[y0:y1, x0:x1]


def save_image(path: Path, image: np.ndarray, title: str | None = None) -> None:
    plt.figure(figsize=(6, 4.8))
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.06)
    plt.close()


def make_native_and_cropped_images(npz_path: str | Path, outdir: str | Path, save_native_debug: bool = True) -> Dict[str, Any]:
    data = load_npz_reflectance(npz_path)
    outdir = ensure_dir(outdir)
    img_dir = ensure_dir(outdir / "paper_figures")
    native_dir = ensure_dir(img_dir / "native_debug")
    crop_dir = ensure_dir(img_dir / "cropped_valid_rgb")

    created = []
    for t, (label, date) in enumerate(zip(data["window_labels"], data["dates"])):
        valid = valid_mask_for_time(data, t)
        rgb = rgb_from_npz(data, t)
        cropped, cropped_valid = crop_to_valid(rgb, valid)

        if save_native_debug:
            native_path = native_dir / f"{t:02d}_{label}_{date}_native_debug.png"
            save_image(native_path, rgb, f"{label} {date} native MODIS grid")
            created.append(str(native_path))

        crop_path = crop_dir / f"{t:02d}_{label}_{date}_cropped_rgb.png"
        save_image(crop_path, cropped, f"{label} {date}")
        created.append(str(crop_path))

    burn_mask = data["burn_2021"] > 0
    plt.figure(figsize=(6, 4.8))
    plt.imshow(burn_mask)
    plt.title("MCD64A1 BurnDate > 0")
    plt.axis("off")
    plt.tight_layout()
    burn_path = img_dir / "burn_mask_cropped_native.png"
    plt.savefig(burn_path, dpi=220, bbox_inches="tight", pad_inches=0.06)
    plt.close()
    created.append(str(burn_path))

    cropped_files = sorted(crop_dir.glob("*.png"))
    if cropped_files:
        n = len(cropped_files)
        cols = 4 if n >= 4 else n
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.2 * rows))
        axes = np.array(axes).reshape(rows, cols)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, f in zip(axes.ravel(), cropped_files):
            img = plt.imread(f)
            ax.imshow(img)
            ax.set_title(f.stem.replace("_cropped_rgb", ""), fontsize=9)
            ax.axis("off")
        plt.tight_layout()
        sheet = img_dir / "cropped_rgb_contact_sheet.png"
        fig.savefig(sheet, dpi=180, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)
        created.append(str(sheet))

    return {"created_files": created, "note": "These are cropped native-grid figures. For north-up EPSG:4326 paper figures, use gee_export_dixie_paper_rgb.py."}


def read_json(path: Path) -> Dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def create_charts_and_dashboard(run_dir: str | Path, outdir: str | Path) -> Dict[str, str]:
    run_dir = Path(run_dir)
    outdir = ensure_dir(outdir)
    fig_dir = ensure_dir(outdir / "paper_figures")

    timeline_path = run_dir / "wildfire_timeline.csv"
    metrics_path = run_dir / "wildfire_tile_detection_metrics.json"
    if not timeline_path.exists():
        raise FileNotFoundError(f"Missing {timeline_path}")

    df = pd.read_csv(timeline_path)
    metrics = read_json(metrics_path)
    best = metrics.get("threshold_sweep_best", {})

    paths = {}

    rec = df[["window_label", "date", "ndvi_recovery_ratio_burned"]].copy()
    pre = rec[rec["window_label"].str.startswith("pre")]
    labels = ["Pre-fire\nbaseline"]
    values = [1.0]
    for _, row in df[~df["window_label"].str.startswith("pre")].iterrows():
        labels.append(f"{row['window_label']}\n{row['date']}")
        values.append(float(row["ndvi_recovery_ratio_burned"]))

    plt.figure(figsize=(8.5, 4.8))
    plt.plot(labels, values, marker="o")
    plt.ylim(0, max(1.05, max(values) + 0.1))
    plt.ylabel("Recovery ratio")
    plt.title("Vegetation recovery over burned pixels")
    plt.xticks(rotation=25, ha="right")
    for i, v in enumerate(values):
        plt.text(i, v + 0.03, f"{v:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    p = fig_dir / "paper_recovery_ratio_chart.png"
    plt.savefig(p, dpi=220, bbox_inches="tight", pad_inches=0.08)
    plt.close()
    paths["recovery_chart"] = str(p)

    df2 = df[~df["window_label"].str.startswith("pre")].copy()
    xlabels = [f"{r.window_label}\n{r.date}" for r in df2.itertuples()]
    plt.figure(figsize=(8.5, 4.8))
    plt.plot(xlabels, df2["mean_tile_change_burned"], marker="o", label="Burned tiles")
    plt.plot(xlabels, df2["mean_tile_change_unburned"], marker="s", label="Unburned tiles")
    plt.ylabel("Mean tile HDC change score")
    plt.title("HDC signature shift: burned vs. unburned tiles")
    plt.xticks(rotation=25, ha="right")
    plt.legend()
    for i, v in enumerate(df2["mean_tile_change_burned"]):
        plt.text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=8)
    for i, v in enumerate(df2["mean_tile_change_unburned"]):
        plt.text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=8)
    plt.tight_layout()
    p = fig_dir / "paper_hdc_burned_vs_unburned_chart.png"
    plt.savefig(p, dpi=220, bbox_inches="tight", pad_inches=0.08)
    plt.close()
    paths["hdc_chart"] = str(p)

    fig = plt.figure(figsize=(13.5, 7.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    ax0.axis("off")
    ax0.text(0, 0.95, "Scenario timeline", fontsize=15, fontweight="bold", va="top")
    timeline_text = (
        "Pre-fire baseline → Fire peak (2021-08) → Post-fire (2021-10) → Recovery (2022–2025)\n\n"
        "Goal: detect abrupt disturbance and track long-term vegetation/soil recovery."
    )
    ax0.text(0, 0.72, timeline_text, fontsize=11, va="top")

    ax1.axis("off")
    ax1.text(0, 0.95, "Key tile-level result", fontsize=15, fontweight="bold", va="top")
    result_text = (
        f"Accuracy: {best.get('accuracy', float('nan')):.3f}\n"
        f"F1-score: {best.get('f1', float('nan')):.3f}\n"
        f"TP/TN/FP/FN: {best.get('tp','?')}/{best.get('tn','?')}/{best.get('fp','?')}/{best.get('fn','?')}\n"
        f"Tiles: {metrics.get('n_tiles','?')} total, {metrics.get('n_burned_tiles','?')} burned\n"
        "Label: MCD64A1 BurnDate"
    )
    ax1.text(0, 0.72, result_text, fontsize=12, va="top")

    ax2.plot(labels, values, marker="o")
    ax2.set_title("NDVI-based recovery ratio")
    ax2.set_ylabel("Recovery ratio")
    ax2.tick_params(axis="x", labelrotation=30)

    ax3.plot(xlabels, df2["mean_tile_change_burned"], marker="o", label="Burned")
    ax3.plot(xlabels, df2["mean_tile_change_unburned"], marker="s", label="Unburned")
    ax3.set_title("HDC tile-signature shift")
    ax3.set_ylabel("Mean HDC change")
    ax3.legend()
    ax3.tick_params(axis="x", labelrotation=30)

    fig.suptitle("BITS: Wildfire recovery monitoring — Dixie Fire case study", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = fig_dir / "paper_dashboard_summary.png"
    fig.savefig(p, dpi=220, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    paths["dashboard"] = str(p)

    return paths


def estimate_downlink_decision_counts(df: pd.DataFrame) -> Dict[str, Any]:
    counts = df["decision"].value_counts().to_dict()
    n = len(df)
    return {
        "n_observations": int(n),
        "decision_counts": {str(k): int(v) for k, v in counts.items()},
        "decision_rates": {str(k): float(v / n) for k, v in counts.items()},
    }


def create_performance_summary(npz_path: str | Path, run_dir: str | Path, outdir: str | Path, D: int = 4096, n_bands: int = 7, bits_per_band: int = 16) -> Dict[str, Any]:
    data = load_npz_reflectance(npz_path)
    run_dir = Path(run_dir)
    outdir = ensure_dir(outdir)

    df = pd.read_csv(run_dir / "wildfire_timeline.csv")
    metrics = read_json(run_dir / "wildfire_tile_detection_metrics.json")

    T, H, W = data["b01"].shape
    tile_scores = pd.read_csv(run_dir / "wildfire_tile_scores.csv")
    n_tiles = int(tile_scores[["pos_row", "pos_col"]].drop_duplicates().shape[0])

    raw_scene_bytes = int(H * W * n_bands * bits_per_band / 8)
    raw_sequence_bytes = int(T * raw_scene_bytes)
    scene_signature_bytes = int(D / 8)
    tile_signature_set_bytes = int(n_tiles * D / 8)

    feature_payload_bytes = scene_signature_bytes + tile_signature_set_bytes

    downlink = estimate_downlink_decision_counts(df)
    decision_counts = downlink["decision_counts"]

    estimated_bytes = 0
    for _, row in df.iterrows():
        decision = str(row["decision"])
        if "priority_downlink_full" in decision:
            estimated_bytes += raw_scene_bytes
        elif "features" in decision:
            estimated_bytes += feature_payload_bytes
        elif "thumbnail" in decision:
            estimated_bytes += int(0.10 * raw_scene_bytes)
        else:
            estimated_bytes += scene_signature_bytes

    raw_all_bytes = T * raw_scene_bytes
    downlink_reduction = raw_all_bytes / max(estimated_bytes, 1)

    post = df[df["window_label"].astype(str).str.startswith("post")]
    final_rec = df[df["window_label"].astype(str).str.startswith("rec")].tail(1)
    recovery_metrics = {
        "pre_fire_ndvi_burned": float(df["pre_ndvi_burned"].iloc[0]),
        "post_fire_recovery_ratio": float(post["ndvi_recovery_ratio_burned"].iloc[0]) if len(post) else None,
        "latest_recovery_ratio": float(final_rec["ndvi_recovery_ratio_burned"].iloc[0]) if len(final_rec) else None,
        "latest_recovery_date": str(final_rec["date"].iloc[0]) if len(final_rec) else None,
        "max_burned_tile_hdc_change": float(df["mean_tile_change_burned"].max()),
        "max_unburned_tile_hdc_change": float(df["mean_tile_change_unburned"].max()),
        "burned_vs_unburned_hdc_peak_ratio": float(df["mean_tile_change_burned"].max() / max(df["mean_tile_change_unburned"].max(), 1e-12)),
    }

    summary = {
        "scenario": "Dixie Fire wildfire recovery",
        "datasets": NASA_DATASET_CITATION,
        "input_shape": {"T": int(T), "H": int(H), "W": int(W), "bands_used": n_bands},
        "signature_config": {
            "D_bits": int(D),
            "scene_signature_bytes": scene_signature_bytes,
            "n_tiles": n_tiles,
            "tile_signature_set_bytes": tile_signature_set_bytes,
            "scene_plus_tile_payload_bytes": feature_payload_bytes,
        },
        "raw_data_estimate": {
            "raw_scene_bytes": raw_scene_bytes,
            "raw_scene_kb": raw_scene_bytes / 1024,
            "raw_sequence_bytes": raw_sequence_bytes,
            "raw_sequence_kb": raw_sequence_bytes / 1024,
            "scene_signature_compression_ratio": raw_scene_bytes / max(scene_signature_bytes, 1),
            "scene_plus_tile_payload_compression_ratio": raw_scene_bytes / max(feature_payload_bytes, 1),
        },
        "downlink_decision_metrics": {
            **downlink,
            "estimated_raw_all_downlink_bytes": int(raw_all_bytes),
            "estimated_bits_policy_downlink_bytes": int(estimated_bytes),
            "estimated_downlink_reduction_ratio": float(downlink_reduction),
            "policy_note": "Estimate assumes priority events send raw scene; feature-only sends scene + tile signatures; skip/reobserve sends scene status signature only.",
        },
        "burn_detection_metrics": metrics,
        "recovery_metrics": recovery_metrics,
        "software_runtime_note": "Runtime profiling should be collected during a fresh run; this postprocessor records storage, memory, compression, downlink, accuracy, and recovery metrics from existing outputs.",
    }

    out_path = outdir / "performance_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True, help="Wildfire NPZ, e.g., dixie_wildfire_bits.npz")
    parser.add_argument("--run-dir", required=True, help="Run directory, e.g., runs/dixie_wildfire")
    parser.add_argument("--out", default=None, help="Output directory; defaults to run-dir")
    parser.add_argument("--D", type=int, default=4096)
    parser.add_argument("--no-native-debug", action="store_true")
    args = parser.parse_args()

    outdir = ensure_dir(args.out or args.run_dir)

    summary = create_performance_summary(args.npz, args.run_dir, outdir, D=args.D)
    chart_paths = create_charts_and_dashboard(args.run_dir, outdir)
    image_paths = make_native_and_cropped_images(args.npz, outdir, save_native_debug=not args.no_native_debug)

    print("Created performance summary:")
    print(outdir / "performance_summary.json")
    print("\nKey storage/downlink values:")
    print(json.dumps(summary["raw_data_estimate"], indent=2))
    print("\nCreated chart files:")
    print(json.dumps(chart_paths, indent=2))
    print("\nCreated remote-sensing image files:")
    print(json.dumps(image_paths, indent=2))


if __name__ == "__main__":
    main()