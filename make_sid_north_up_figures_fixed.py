from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling


def load_bits_module(path: str | Path = "bits_sid_nbai_demo_v6_wildfire_fixed.py"):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cannot find {path}. Put this script in the same folder as bits_sid_nbai_demo_v6_wildfire_fixed.py")
    spec = importlib.util.spec_from_file_location("bits_v6", str(path))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None

    import sys
    sys.modules[spec.name] = mod

    spec.loader.exec_module(mod)
    return mod


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


def load_npz(npz_path: str | Path) -> Dict[str, Any]:
    data = np.load(npz_path)
    out = {}
    for k in ["b01", "b02", "b03", "b04", "b05", "b06", "b07"]:
        arr = data[k].astype(np.float32)
        if np.nanmax(arr) > 2.0:
            arr = arr * 0.0001
        out[k] = clip01(arr)
    out["state_1km"] = data["state_1km"].astype(np.uint16)
    out["dates"] = [str(x) for x in data["dates"].tolist()]
    out["window_labels"] = [str(x) for x in data["window_labels"].tolist()]
    return out


def valid_mask(data: Dict[str, Any], t: int) -> np.ndarray:
    stack = np.stack([data[f"b{i:02d}"][t] for i in range(1, 8)], axis=0)
    return np.isfinite(stack).all(axis=0) & (np.abs(stack).sum(axis=0) > 1e-6)


def rgb_from_bands_scaled(bands: Dict[str, np.ndarray], valid: np.ndarray) -> np.ndarray:
    rgb = np.dstack([
        robust_scale(bands["b01"], valid),
        robust_scale(bands["b04"], valid),
        robust_scale(bands["b03"], valid),
    ])
    rgb[~valid] = np.nan
    return rgb


def reproject_rgb(rgb: np.ndarray, src_crs, src_transform, dst_crs: str = "EPSG:4326") -> np.ndarray:
    h, w = rgb.shape[:2]
    dst_transform, dst_w, dst_h = calculate_default_transform(src_crs, dst_crs, w, h, *rasterio.transform.array_bounds(h, w, src_transform))

    dst = np.zeros((3, dst_h, dst_w), dtype=np.float32)
    for c in range(3):
        src = np.nan_to_num(rgb[:, :, c], nan=0.0).astype(np.float32)
        reproject(
            source=src,
            destination=dst[c],
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
            dst_nodata=0.0,
        )

    out = np.moveaxis(dst, 0, -1)
    mask = out.sum(axis=2) > 1e-6
    ys, xs = np.where(mask)
    if len(xs) and len(ys):
        y0, y1 = max(0, ys.min() - 2), min(out.shape[0], ys.max() + 3)
        x0, x1 = max(0, xs.min() - 2), min(out.shape[1], xs.max() + 3)
        out = out[y0:y1, x0:x1]
    return clip01(out)


def save_rgb(path: Path, rgb: np.ndarray, title: str) -> None:
    plt.figure(figsize=(6.4, 5.2))
    plt.imshow(rgb)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.06)
    plt.close()


def make_difference(input_rgb: np.ndarray, clean_rgb: np.ndarray) -> np.ndarray:
    h = min(input_rgb.shape[0], clean_rgb.shape[0])
    w = min(input_rgb.shape[1], clean_rgb.shape[1])
    diff = np.abs(clean_rgb[:h, :w] - input_rgb[:h, :w])
    diff = diff / max(float(np.nanmax(diff)), 1e-8)
    return clip01(diff)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", default="dixie_wildfire_dense.npz")
    parser.add_argument("--tif", default="dixie_wildfire_dense_stack.tif", help="Original dense GeoTIFF with MODIS sinusoidal georeferencing.")
    parser.add_argument("--out", default=r"runs\dixie_wildfire_dense_aug\paper_figures\sid_north_up")
    parser.add_argument("--bits-script", default="bits_sid_nbai_demo_v6_wildfire_fixed.py")
    parser.add_argument("--labels", nargs="*", default=["fire_202108a", "fire_202108b", "rec_202304", "rec_202404"])
    args = parser.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    bits = load_bits_module(args.bits_script)
    data = load_npz(args.npz)

    with rasterio.open(args.tif) as src:
        src_crs = src.crs
        src_transform = src.transform

    sid = bits.SID3D4Cleaner(bits.SIDConfig())

    created = []
    for label in args.labels:
        if label not in data["window_labels"]:
            print(f"Skipping {label}: not found.")
            continue
        t = data["window_labels"].index(label)
        date = data["dates"][t]

        bands = {f"b{i:02d}": data[f"b{i:02d}"][t] for i in range(1, 8)}
        state = data["state_1km"][t]
        valid = valid_mask(data, t)

        cleaned = sid.clean_bands(bands, state)["clean_bands"]

        input_rgb = rgb_from_bands_scaled(bands, valid)
        clean_rgb = rgb_from_bands_scaled(cleaned, valid)

        input_north = reproject_rgb(input_rgb, src_crs, src_transform)
        clean_north = reproject_rgb(clean_rgb, src_crs, src_transform)
        diff_north = make_difference(input_north, clean_north)

        base = f"{t:02d}_{label}_{date}"
        p1 = outdir / f"{base}_input_north_up.png"
        p2 = outdir / f"{base}_sid_clean_north_up.png"
        p3 = outdir / f"{base}_sid_difference_north_up.png"

        save_rgb(p1, input_north, f"{label} {date} input")
        save_rgb(p2, clean_north, f"{label} {date} SID-conditioned")
        save_rgb(p3, diff_north, f"{label} {date} |SID - input|")

        created.extend([str(p1), str(p2), str(p3)])

    pngs = [Path(p) for p in created]
    if pngs:
        cols = 3
        rows = int(np.ceil(len(pngs) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        axes = np.array(axes).reshape(rows, cols)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, p in zip(axes.ravel(), pngs):
            img = plt.imread(p)
            ax.imshow(img)
            ax.set_title(p.stem, fontsize=8)
            ax.axis("off")
        plt.tight_layout()
        sheet = outdir / "sid_north_up_contact_sheet.png"
        fig.savefig(sheet, dpi=180, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)
        created.append(str(sheet))

    print("Created SID north-up presentation figures:")
    for p in created:
        print(p)


if __name__ == "__main__":
    main()