from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="runs/dixie_wildfire_dense_aug")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out) if args.out else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    timeline = pd.read_csv(run_dir / "wildfire_timeline.csv")
    perf = load_json(run_dir / "performance_summary.json")
    burn_metrics = load_json(run_dir / "wildfire_tile_detection_metrics.json")

    valid = timeline.copy()
    valid = valid[
        (valid["quality_score"].astype(float) >= 0.85)
        & (valid["ndvi_burned"].astype(float) > 0.05)
        & (valid["ndvi_recovery_ratio_burned"].astype(float) > 0.05)
    ].copy()

    post_candidates = valid[valid["window_label"].astype(str).str.startswith("post_")]
    post_202110 = post_candidates[post_candidates["window_label"].astype(str).eq("post_202110")]
    if len(post_202110):
        post_row = post_202110.iloc[0]
    elif len(post_candidates):
        post_row = post_candidates.iloc[0]
    else:
        post_row = None

    june_rows = valid[valid["window_label"].astype(str).str.contains("06", regex=False)].copy()
    june_rec_rows = june_rows[june_rows["window_label"].astype(str).str.startswith("rec_")]
    latest_june = june_rec_rows.tail(1).iloc[0] if len(june_rec_rows) else None

    best = burn_metrics.get("threshold_sweep_best", {})
    raw = perf.get("raw_data_estimate", {})
    dl = perf.get("downlink_decision_metrics", {})
    sig = perf.get("signature_config", {})
    rec = perf.get("recovery_metrics", {})

    final = {
        "scenario": "Dixie Fire dense wildfire recovery",
        "recommended_for_presentation": True,
        "datasets": perf.get("datasets", {}),
        "headline_detection": {
            "eval_window_label": burn_metrics.get("eval_window_label"),
            "eval_date": burn_metrics.get("eval_date"),
            "n_tiles": burn_metrics.get("n_tiles"),
            "n_burned_tiles": burn_metrics.get("n_burned_tiles"),
            "accuracy": best.get("accuracy"),
            "f1": best.get("f1"),
            "tp": best.get("tp"),
            "tn": best.get("tn"),
            "fp": best.get("fp"),
            "fn": best.get("fn"),
            "note": "Exploratory threshold sweep against MCD64A1 BurnDate proxy labels.",
        },
        "clean_recovery_metrics": {
            "pre_fire_ndvi_burned": float(timeline["pre_ndvi_burned"].iloc[0]),
            "post_fire_reference_window": None if post_row is None else str(post_row["window_label"]),
            "post_fire_reference_date": None if post_row is None else str(post_row["date"]),
            "post_fire_recovery_ratio": None if post_row is None else float(post_row["ndvi_recovery_ratio_burned"]),
            "latest_june_recovery_window": None if latest_june is None else str(latest_june["window_label"]),
            "latest_june_recovery_date": None if latest_june is None else str(latest_june["date"]),
            "latest_june_recovery_ratio": None if latest_june is None else float(latest_june["ndvi_recovery_ratio_burned"]),
            "excluded_note": "post_202109 was excluded from recovery headline if NDVI/recovery ratio was zero or invalid.",
        },
        "hdc_peak_and_separation": {
            "max_burned_tile_hdc_change": rec.get("max_burned_tile_hdc_change"),
            "max_unburned_tile_hdc_change": rec.get("max_unburned_tile_hdc_change"),
            "burned_vs_unburned_hdc_peak_ratio": rec.get("burned_vs_unburned_hdc_peak_ratio"),
        },
        "software_memory_downlink": {
            "raw_scene_kb": raw.get("raw_scene_kb"),
            "raw_sequence_kb": raw.get("raw_sequence_kb"),
            "scene_signature_bytes": sig.get("scene_signature_bytes"),
            "tile_signature_set_bytes": sig.get("tile_signature_set_bytes"),
            "scene_plus_tile_payload_bytes": sig.get("scene_plus_tile_payload_bytes"),
            "scene_signature_compression_ratio": raw.get("scene_signature_compression_ratio"),
            "scene_plus_tile_payload_compression_ratio": raw.get("scene_plus_tile_payload_compression_ratio"),
            "estimated_downlink_reduction_ratio": dl.get("estimated_downlink_reduction_ratio"),
            "decision_counts": dl.get("decision_counts"),
        },
    }

    (out_dir / "final_presentation_metrics.json").write_text(json.dumps(final, indent=2), encoding="utf-8")

    txt = []
    txt.append("BITS Dixie Fire dense validation — presentation-ready metrics")
    txt.append("=" * 68)
    txt.append(f"Dataset: {final['datasets'].get('primary_dataset')} DOI {final['datasets'].get('primary_dataset_doi')}")
    txt.append(f"Burn label: {final['datasets'].get('burn_label_dataset')}")
    txt.append("")
    hd = final["headline_detection"]
    txt.append(f"Peak window: {hd['eval_window_label']} ({hd['eval_date']})")
    txt.append(f"Tile detection: accuracy={hd['accuracy']:.4f}, F1={hd['f1']:.4f}, TP/TN/FP/FN={hd['tp']}/{hd['tn']}/{hd['fp']}/{hd['fn']}")
    txt.append(f"Tiles: {hd['n_tiles']} total, {hd['n_burned_tiles']} burned")
    txt.append("")
    crm = final["clean_recovery_metrics"]
    txt.append(f"Post-fire recovery reference: {crm['post_fire_reference_window']} ({crm['post_fire_reference_date']}), ratio={crm['post_fire_recovery_ratio']:.4f}")
    txt.append(f"Latest June recovery: {crm['latest_june_recovery_window']} ({crm['latest_june_recovery_date']}), ratio={crm['latest_june_recovery_ratio']:.4f}")
    txt.append("")
    hdc = final["hdc_peak_and_separation"]
    txt.append(f"HDC burned/unburned peak ratio: {hdc['burned_vs_unburned_hdc_peak_ratio']:.2f}x")
    txt.append("")
    sw = final["software_memory_downlink"]
    txt.append(f"Raw scene: {sw['raw_scene_kb']:.1f} KB")
    txt.append(f"Scene signature: {sw['scene_signature_bytes']} bytes")
    txt.append(f"Scene-level compression: {sw['scene_signature_compression_ratio']:.1f}x")
    txt.append(f"Scene + tile payload compression: {sw['scene_plus_tile_payload_compression_ratio']:.2f}x")
    txt.append(f"Estimated downlink reduction: {sw['estimated_downlink_reduction_ratio']:.2f}x")
    txt.append(f"Decision counts: {sw['decision_counts']}")
    txt.append("")
    txt.append("Recommended claim:")
    txt.append("BITS captures the temporal emergence of wildfire disturbance and tracks long-term vegetation recovery using compact binary HDC signatures, while reducing estimated downlink by prioritizing feature-only or low-priority transmissions.")
    (out_dir / "final_presentation_metrics.txt").write_text("\n".join(txt), encoding="utf-8")

    print("\n".join(txt))
    print("\nCreated:")
    print(out_dir / "final_presentation_metrics.json")
    print(out_dir / "final_presentation_metrics.txt")


if __name__ == "__main__":
    main()