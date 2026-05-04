from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

GEE_PROJECT = "rapid-being-484808-t1"

ROI = [-121.65, 39.55, -120.85, 40.40]

EXPORT_FOLDER = "BITS_exports"
EXPORT_PREFIX = "dixie_paper_rgb_stack"
METADATA_JSON = "dixie_paper_rgb_metadata.json"

WINDOWS = [
    ("pre_202105",  "2021-05-10", "2021-05-31"),
    ("pre_202106",  "2021-06-10", "2021-06-30"),
    ("fire_202108", "2021-08-05", "2021-08-25"),
    ("post_202110", "2021-10-05", "2021-10-31"),
    ("rec_202206",  "2022-06-10", "2022-06-30"),
    ("rec_202306",  "2023-06-10", "2023-06-30"),
    ("rec_202406",  "2024-06-10", "2024-06-30"),
    ("rec_202506",  "2025-06-10", "2025-06-30"),
]

MOD09GA_BANDS = [
    "sur_refl_b01", "sur_refl_b02", "sur_refl_b03", "sur_refl_b04",
    "sur_refl_b05", "sur_refl_b06", "sur_refl_b07", "state_1km"
]


def _date_from_millis(ms: int) -> str:
    return dt.datetime.utcfromtimestamp(ms / 1000).strftime("%Y%m%d")


def export_paper_rgb() -> None:
    import ee

    ee.Initialize(project=GEE_PROJECT)
    roi = ee.Geometry.Rectangle(ROI)

    base_col = (
        ee.ImageCollection("MODIS/061/MOD09GA")
        .filterBounds(roi)
        .select(MOD09GA_BANDS)
    )

    def add_bad_fraction(img):
        state = img.select("state_1km")
        cloud_state = state.bitwiseAnd(3)
        cloud_shadow = state.rightShift(2).bitwiseAnd(1)
        aerosol = state.rightShift(6).bitwiseAnd(3)
        cirrus = state.rightShift(8).bitwiseAnd(3)

        bad = cloud_state.eq(1).Or(cloud_shadow.eq(1)).Or(cirrus.eq(3)).Or(aerosol.eq(3))
        bad_frac = bad.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=roi,
            scale=1000,
            maxPixels=1e8,
            bestEffort=True,
        ).get("state_1km")
        return img.set("bad_frac", bad_frac)

    vis_images = []
    labels = []
    dates = []
    bads = []

    for label, start, end in WINDOWS:
        col = base_col.filterDate(start, end).map(add_bad_fraction).sort("bad_frac")
        count = int(col.size().getInfo())
        print(f"{label}: {count} images")
        if count == 0:
            raise RuntimeError(f"No image for {label}")
        img = ee.Image(col.first()).clip(roi)

        time_ms = int(img.get("system:time_start").getInfo())
        date = _date_from_millis(time_ms)
        bad_frac = img.get("bad_frac").getInfo()
        labels.append(label)
        dates.append(date)
        bads.append(float(bad_frac) if bad_frac is not None else None)

        rgb = (
            img.select(["sur_refl_b01", "sur_refl_b04", "sur_refl_b03"])
            .multiply(0.0001)
            .visualize(min=0.0, max=0.35, gamma=1.25)
            .rename([f"{label}_R", f"{label}_G", f"{label}_B"])
        )
        vis_images.append(rgb)
        print(f"  selected {date}, bad_frac={bad_frac}")

    burn = (
        ee.ImageCollection("MODIS/061/MCD64A1")
        .filterDate("2021-07-01", "2021-11-01")
        .filterBounds(roi)
        .select("BurnDate")
        .max()
        .clip(roi)
        .gt(0)
        .unmask(0)
        .visualize(min=0, max=1, palette=["000000", "ff0000"])
        .rename(["burn_R", "burn_G", "burn_B"])
    )
    vis_images.append(burn)

    stack = ee.Image.cat(vis_images)

    metadata = {
        "scenario": "Dixie Fire paper RGB figures",
        "roi": ROI,
        "labels": labels,
        "dates": dates,
        "bad_frac": bads,
        "export_prefix": EXPORT_PREFIX,
        "note": "Bands are triplets: label_R,label_G,label_B, followed by burn_R,burn_G,burn_B.",
    }
    Path(METADATA_JSON).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved local metadata: {METADATA_JSON}")

    task = ee.batch.Export.image.toDrive(
        image=stack,
        description="BITS_Dixie_paper_RGB_EPSG4326",
        folder=EXPORT_FOLDER,
        fileNamePrefix=EXPORT_PREFIX,
        region=roi,
        scale=500,
        crs="EPSG:4326",
        maxPixels=1e9,
    )
    task.start()
    print("Started export task:", task.id)
    print(f"Expected Drive file: {EXPORT_FOLDER}/{EXPORT_PREFIX}.tif")


def convert_rgb_tif(tif_path: str | Path, outdir: str | Path, metadata_json: str | Path = METADATA_JSON) -> None:
    import numpy as np
    import rasterio
    import matplotlib.pyplot as plt

    tif_path = Path(tif_path)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(Path(metadata_json).read_text(encoding="utf-8"))

    labels = metadata["labels"]
    dates = metadata["dates"]
    expected = len(labels) * 3 + 3

    with rasterio.open(tif_path) as src:
        arr = src.read()
        print("GeoTIFF shape:", arr.shape)
        print("CRS:", src.crs)
        print("Transform:", src.transform)

    if arr.shape[0] != expected:
        raise RuntimeError(f"Expected {expected} bands, got {arr.shape[0]}")

    paths = []
    for i, (label, date) in enumerate(zip(labels, dates)):
        rgb = np.moveaxis(arr[i*3:i*3+3], 0, -1)
        path = outdir / f"{i:02d}_{label}_{date}_north_up_rgb.png"
        plt.figure(figsize=(6.2, 5))
        plt.imshow(rgb)
        plt.title(f"{label} {date}")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.06)
        plt.close()
        paths.append(path)

    burn_rgb = np.moveaxis(arr[len(labels)*3:len(labels)*3+3], 0, -1)
    burn_path = outdir / "mcd64a1_burn_mask_north_up.png"
    plt.figure(figsize=(6.2, 5))
    plt.imshow(burn_rgb)
    plt.title("MCD64A1 BurnDate > 0")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(burn_path, dpi=220, bbox_inches="tight", pad_inches=0.06)
    plt.close()
    paths.append(burn_path)

    cols = 4
    rows = int(np.ceil(len(paths) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.2 * rows))
    axes = np.array(axes).reshape(rows, cols)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, path in zip(axes.ravel(), paths):
        img = plt.imread(path)
        ax.imshow(img)
        ax.set_title(path.stem, fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    sheet = outdir / "north_up_rgb_contact_sheet.png"
    fig.savefig(sheet, dpi=180, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

    print("Created PNGs in:", outdir)
    print("Contact sheet:", sheet)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--convert", action="store_true")
    parser.add_argument("--tif", default=f"{EXPORT_PREFIX}.tif")
    parser.add_argument("--out", default="runs/dixie_wildfire/paper_figures/north_up_rgb")
    parser.add_argument("--metadata-json", default=METADATA_JSON)
    args = parser.parse_args()

    if args.export:
        export_paper_rgb()
    if args.convert:
        convert_rgb_tif(args.tif, args.out, args.metadata_json)
    if not args.export and not args.convert:
        parser.print_help()


if __name__ == "__main__":
    main()