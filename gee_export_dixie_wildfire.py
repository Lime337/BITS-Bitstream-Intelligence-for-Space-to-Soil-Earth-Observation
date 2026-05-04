from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

GEE_PROJECT = "rapid-being-484808-t1"
ROI = [-121.65, 39.55, -120.85, 40.40]

EXPORT_FOLDER = "BITS_exports"
EXPORT_PREFIX = "dixie_wildfire_mod09ga_mcd64a1_stack"
METADATA_JSON = "dixie_wildfire_metadata.json"

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
    "sur_refl_b01",  # red
    "sur_refl_b02",  # NIR
    "sur_refl_b03",  # blue
    "sur_refl_b04",  # green
    "sur_refl_b05",  # SWIR
    "sur_refl_b06",  # SWIR
    "sur_refl_b07",  # SWIR
    "state_1km",   # QA state flags
]

RENAMED_BANDS = ["b01", "b02", "b03", "b04", "b05", "b06", "b07", "state_1km"]


def _date_from_millis(ms: int) -> str:
    return dt.datetime.utcfromtimestamp(ms / 1000).strftime("%Y%m%d")


def export_to_drive() -> None:
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

        bad = (
            cloud_state.eq(1)
            .Or(cloud_shadow.eq(1))
            .Or(cirrus.eq(3))
            .Or(aerosol.eq(3))
        )

        bad_frac = bad.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=roi,
            scale=1000,
            maxPixels=1e8,
            bestEffort=True,
        ).get("state_1km")

        return img.set("bad_frac", bad_frac)

    images = []
    selected_dates = []
    selected_bad_fracs = []
    window_labels = []

    for label, start, end in WINDOWS:
        col = (
            base_col
            .filterDate(start, end)
            .map(add_bad_fraction)
            .sort("bad_frac")
        )
        count = int(col.size().getInfo())
        print(f"{label}: found {count} MOD09GA images in {start} to {end}")
        if count == 0:
            raise RuntimeError(f"No MOD09GA image found for window {label}")

        img = ee.Image(col.first()).clip(roi).rename([f"{label}_{b}" for b in RENAMED_BANDS]).toInt32()
        time_ms = int(ee.Image(col.first()).get("system:time_start").getInfo())
        bad_frac = ee.Image(col.first()).get("bad_frac").getInfo()

        images.append(img)
        selected_dates.append(_date_from_millis(time_ms))
        selected_bad_fracs.append(float(bad_frac) if bad_frac is not None else None)
        window_labels.append(label)

        print(f"  selected date: {selected_dates[-1]}, bad_frac={selected_bad_fracs[-1]}")

    mod09_stack = ee.Image.cat(images).toInt32()

    burn = (
        ee.ImageCollection("MODIS/061/MCD64A1")
        .filterDate("2021-07-01", "2021-11-01")
        .filterBounds(roi)
        .select("BurnDate")
        .max()
        .clip(roi)
        .rename("burn_2021")
        .unmask(0)
        .toInt32()
    )

    stack = ee.Image.cat([mod09_stack, burn]).toInt32()

    metadata = {
        "scenario": "Dixie Fire wildfire recovery",
        "roi": ROI,
        "mod09ga_collection": "MODIS/061/MOD09GA",
        "mod09ga_doi": "10.5067/MODIS/MOD09GA.061",
        "mcd64a1_collection": "MODIS/061/MCD64A1",
        "window_labels": window_labels,
        "selected_dates": selected_dates,
        "selected_bad_fracs": selected_bad_fracs,
        "renamed_bands": RENAMED_BANDS,
        "export_band_order": "date-major MOD09GA bands, then final burn_2021 band",
        "windows": WINDOWS,
        "export_prefix": EXPORT_PREFIX,
    }
    Path(METADATA_JSON).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved local metadata: {METADATA_JSON}")

    task = ee.batch.Export.image.toDrive(
        image=stack,
        description="BITS_Dixie_wildfire_MOD09GA_MCD64A1_INT32",
        folder=EXPORT_FOLDER,
        fileNamePrefix=EXPORT_PREFIX,
        region=roi,
        scale=500,
        maxPixels=1e9,
    )
    task.start()

    print("Started export task.")
    print("Task ID:", task.id)
    print(f"Check Google Drive folder: {EXPORT_FOLDER}")
    print(f"Expected file: {EXPORT_PREFIX}.tif")


def convert_tif_to_npz(tif_path: str, npz_path: str, metadata_json: str = METADATA_JSON) -> None:
    import numpy as np
    import rasterio

    tif_path = Path(tif_path)
    npz_path = Path(npz_path)
    metadata_path = Path(metadata_json)

    if not tif_path.exists():
        raise FileNotFoundError(f"GeoTIFF not found: {tif_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata JSON not found: {metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    T = len(metadata["window_labels"])
    n_bands_per_time = len(RENAMED_BANDS)
    expected_bands = T * n_bands_per_time + 1

    with rasterio.open(tif_path) as src:
        arr = src.read()
        print("GeoTIFF shape:", arr.shape)
        print("Band count:", src.count)
        print("Dtypes:", src.dtypes)
        print("CRS:", src.crs)
        print("Transform:", src.transform)

    if arr.shape[0] != expected_bands:
        raise RuntimeError(
            f"Expected {expected_bands} bands = {T} windows x {n_bands_per_time} MOD09GA bands + burn_2021, "
            f"but got {arr.shape[0]} bands."
        )

    mod09 = arr[:T * n_bands_per_time].reshape(T, n_bands_per_time, arr.shape[1], arr.shape[2])
    burn = arr[-1]

    data = {}
    for i, name in enumerate(RENAMED_BANDS):
        data[name] = mod09[:, i, :, :]

    data["burn_2021"] = burn
    data["dates"] = np.array(metadata["selected_dates"], dtype="<U16")
    data["window_labels"] = np.array(metadata["window_labels"], dtype="<U32")
    data["bad_frac"] = np.array(metadata["selected_bad_fracs"], dtype=np.float32)
    data["roi"] = np.array(metadata["roi"], dtype=np.float32)

    np.savez(npz_path, **data)

    print(f"Saved wildfire NPZ: {npz_path}")
    print("Windows:", metadata["window_labels"])
    print("Dates:", metadata["selected_dates"])
    print("Now run:")
    print(
        f"python bits_sid_nbai_demo_v5_wildfire.py "
        f"--mode wildfire_npz --npz {npz_path} --out runs/dixie_wildfire --save-all-images"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", action="store_true", help="Export Dixie Fire MOD09GA + MCD64A1 stack to Google Drive.")
    parser.add_argument("--convert", action="store_true", help="Convert downloaded GeoTIFF to NPZ.")
    parser.add_argument("--tif", type=str, default=f"{EXPORT_PREFIX}.tif", help="Downloaded GeoTIFF path.")
    parser.add_argument("--npz", type=str, default="dixie_wildfire_bits.npz", help="Output NPZ path.")
    parser.add_argument("--metadata-json", type=str, default=METADATA_JSON, help="Metadata JSON created during export.")
    args = parser.parse_args()

    if args.export:
        export_to_drive()

    if args.convert:
        convert_tif_to_npz(args.tif, args.npz, args.metadata_json)

    if not args.export and not args.convert:
        parser.print_help()


if __name__ == "__main__":
    main()