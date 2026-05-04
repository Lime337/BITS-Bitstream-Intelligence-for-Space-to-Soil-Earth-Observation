from __future__ import annotations

import argparse
import json
from pathlib import Path

GEE_PROJECT = "rapid-being-484808-t1"

ROI = [-92.8, 30.1, -92.3, 30.5]

START_DATE = "2022-06-01"
END_DATE = "2022-06-10"

EXPORT_FOLDER = "BITS_exports"
EXPORT_PREFIX = "mod09ga_sequence_stack"
METADATA_JSON = "mod09ga_sequence_metadata.json"

BANDS = [
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


def export_sequence_to_drive() -> None:
    import ee

    ee.Initialize(project=GEE_PROJECT)

    roi = ee.Geometry.Rectangle(ROI)

    collection = (
        ee.ImageCollection("MODIS/061/MOD09GA")
        .filterDate(START_DATE, END_DATE)
        .filterBounds(roi)
        .select(BANDS)
        .sort("system:time_start")
    )

    count = int(collection.size().getInfo())
    print(f"Found {count} MOD09GA images between {START_DATE} and {END_DATE} for ROI={ROI}")

    if count == 0:
        raise RuntimeError("No MOD09GA images found. Try a wider date range or different ROI.")

    time_list = collection.aggregate_array("system:time_start").getInfo()
    dates = []
    for t in time_list:
        import datetime as _dt
        dates.append(_dt.datetime.utcfromtimestamp(t / 1000).strftime("%Y%m%d"))

    metadata = {
        "collection": "MODIS/061/MOD09GA",
        "doi": "10.5067/MODIS/MOD09GA.061",
        "roi": ROI,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "dates": dates,
        "renamed_bands": RENAMED_BANDS,
        "band_order": "For each date, bands are ordered as b01,b02,b03,b04,b05,b06,b07,state_1km",
        "export_prefix": EXPORT_PREFIX,
    }
    Path(METADATA_JSON).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved local metadata: {METADATA_JSON}")
    print("Dates:", dates)

    def prep(img):
        return img.clip(roi).rename(RENAMED_BANDS).toInt32()

    stack = collection.map(prep).toBands().toInt32()

    task = ee.batch.Export.image.toDrive(
        image=stack,
        description="BITS_MOD09GA_sequence_INT32",
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
    print("You can also check task status in the Earth Engine Code Editor Tasks tab.")


def convert_tif_to_npz(tif_path: str, npz_path: str, metadata_json: str = METADATA_JSON) -> None:
    import numpy as np
    import rasterio

    tif_path = Path(tif_path)
    npz_path = Path(npz_path)
    metadata_path = Path(metadata_json)

    if not tif_path.exists():
        raise FileNotFoundError(f"GeoTIFF not found: {tif_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata JSON not found: {metadata_path}. "
            "Run --export first, or pass --metadata-json."
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    dates = metadata["dates"]
    T = len(dates)
    n_bands_per_time = len(RENAMED_BANDS)
    expected_bands = T * n_bands_per_time

    with rasterio.open(tif_path) as src:
        arr = src.read()
        print("GeoTIFF shape:", arr.shape)
        print("Band count:", src.count)
        print("Dtypes:", src.dtypes)
        print("CRS:", src.crs)
        print("Transform:", src.transform)

    if arr.shape[0] != expected_bands:
        raise RuntimeError(
            f"Expected {expected_bands} bands from {T} dates x {n_bands_per_time} bands/date, "
            f"but GeoTIFF has {arr.shape[0]} bands."
        )

    stack = arr.reshape(T, n_bands_per_time, arr.shape[1], arr.shape[2])

    data = {}
    for i, name in enumerate(RENAMED_BANDS):
        data[name] = stack[:, i, :, :]

    data["dates"] = np.array(dates, dtype="<U16")
    data["roi"] = np.array(metadata["roi"], dtype=np.float32)

    np.savez(npz_path, **data)

    print(f"Saved sequence NPZ: {npz_path}")
    print("Dates:", dates)
    print("Now run:")
    print(
        f"python bits_sid_nbai_demo_v3_sequence.py "
        f"--mode npz_sequence --npz {npz_path} --out runs/modis_sequence_01"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", action="store_true", help="Export MOD09GA sequence to Google Drive.")
    parser.add_argument("--convert", action="store_true", help="Convert downloaded sequence GeoTIFF to NPZ.")
    parser.add_argument("--tif", type=str, default=f"{EXPORT_PREFIX}.tif", help="Downloaded GeoTIFF path.")
    parser.add_argument("--npz", type=str, default="mod09ga_sequence.npz", help="Output NPZ path.")
    parser.add_argument("--metadata-json", type=str, default=METADATA_JSON, help="Metadata JSON created during export.")
    args = parser.parse_args()

    if args.export:
        export_sequence_to_drive()

    if args.convert:
        convert_tif_to_npz(args.tif, args.npz, args.metadata_json)

    if not args.export and not args.convert:
        parser.print_help()


if __name__ == "__main__":
    main()