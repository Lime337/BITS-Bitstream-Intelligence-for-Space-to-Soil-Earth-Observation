import argparse
from pathlib import Path

GEE_PROJECT = "rapid-being-484808-t1"

ROI = [-92.8, 30.1, -92.3, 30.5]

START_DATE = "2022-06-01"
END_DATE = "2022-06-10"

EXPORT_FOLDER = "BITS_exports"
EXPORT_PREFIX = "mod09ga_bits_tile"

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


def export_to_drive() -> None:
    import ee

    ee.Initialize(project=GEE_PROJECT)

    roi = ee.Geometry.Rectangle(ROI)

    collection = (
        ee.ImageCollection("MODIS/061/MOD09GA")
        .filterDate(START_DATE, END_DATE)
        .filterBounds(roi)
        .select(BANDS)
    )

    count = collection.size().getInfo()
    print(f"Found {count} MOD09GA images between {START_DATE} and {END_DATE} for ROI={ROI}")

    if count == 0:
        raise RuntimeError("No MOD09GA images found. Try a wider date range or different ROI.")

    img = collection.first().clip(roi).rename(RENAMED_BANDS)
    img = img.toInt32()

    task = ee.batch.Export.image.toDrive(
        image=img,
        description="BITS_MOD09GA_tile",
        folder=EXPORT_FOLDER,
        fileNamePrefix=EXPORT_PREFIX,
        region=roi,
        scale=500,
        maxPixels=1e8,
    )
    task.start()

    print("Started export task.")
    print("Task ID:", task.id)
    print(f"Check Google Drive folder: {EXPORT_FOLDER}")
    print("You can also check task status in the Earth Engine Code Editor Tasks tab.")


def convert_tif_to_npz(tif_path: str, npz_path: str) -> None:
    import numpy as np
    import rasterio

    tif_path = Path(tif_path)
    npz_path = Path(npz_path)

    if not tif_path.exists():
        raise FileNotFoundError(f"GeoTIFF not found: {tif_path}")

    with rasterio.open(tif_path) as src:
        arr = src.read()
        print("GeoTIFF shape:", arr.shape)
        print("Band count:", src.count)
        print("Dtypes:", src.dtypes)
        print("CRS:", src.crs)
        print("Transform:", src.transform)

    if arr.shape[0] != len(RENAMED_BANDS):
        raise RuntimeError(
            f"Expected {len(RENAMED_BANDS)} bands, got {arr.shape[0]}. "
            "Check exported bands."
        )

    data = {name: arr[i] for i, name in enumerate(RENAMED_BANDS)}
    np.savez(npz_path, **data)

    print(f"Saved NPZ: {npz_path}")
    print("Now run:")
    print(
        f"python bits_sid_nbai_demo_v2_calibrated.py "
        f"--mode npz --npz {npz_path} --out runs/modis_real_01"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", action="store_true", help="Export MOD09GA tile to Google Drive.")
    parser.add_argument("--convert", action="store_true", help="Convert downloaded GeoTIFF to NPZ.")
    parser.add_argument("--tif", type=str, default=f"{EXPORT_PREFIX}.tif", help="Downloaded GeoTIFF path.")
    parser.add_argument("--npz", type=str, default=f"{EXPORT_PREFIX}.npz", help="Output NPZ path.")
    args = parser.parse_args()

    if args.export:
        export_to_drive()

    if args.convert:
        convert_tif_to_npz(args.tif, args.npz)

    if not args.export and not args.convert:
        parser.print_help()


if __name__ == "__main__":
    main()