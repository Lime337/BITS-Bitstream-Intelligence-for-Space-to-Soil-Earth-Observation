# BITS: Bitstream Intelligence for Space-to-Soil Earth Observation

**BITS** is an end-to-end prototype for quality-aware remote-sensing image conditioning, compact hyperdimensional land-signature generation, and onboard-style decision support for soil, vegetation, and land-resilience monitoring.

The current demonstration uses the **2021 Dixie Fire** region in Northern California to show how BITS can track pre-fire baseline conditions, wildfire-driven peak disturbance, burned-area separability, post-fire vegetation loss, and long-term recovery.

---

## 1. Project Overview

![Overall Methodology](./assets/methodology.png)

BITS combines two processing layers.

### Layer 1 — SID: Stochastic Image Conditioning

**SID** is a stochastic/bitstream-inspired image-conditioning front end. In this prototype, it is used as a quality-aware conditioning layer for degraded remote-sensing observations.

SID is not intended to hallucinate missing land information under opaque clouds. Instead, it is designed to:

- stabilize recoverable haze, thin-cloud, or noise-affected observations,
- improve local contrast and sharpness,
- preserve key spectral indices within a moderate perturbation range,
- prevent degraded observations from corrupting downstream HDC memory.

### Layer 2 — N-BAI: Neuro-symbolic Bit-level AI

**N-BAI** encodes SID-conditioned MODIS observations into compact binary hyperdimensional computing/vector-symbolic architecture signatures.

Each tile/scene signature captures land-condition cues such as:

- vegetation state,
- exposed-soil and surface brightness,
- NIR/Red relationship,
- SWIR and burn-sensitive response,
- texture and spatial structure.

N-BAI compares current signatures against pre-fire baseline signatures using Hamming distance.

```text
MOD09GA b01-b07 + state_1km QA
        |
        v
SID image conditioning
        |
        v
N-BAI / HDC binary land-condition signatures
        |
        v
Change score + recovery tracking + adaptive downlink/re-observation decision
```

---

## 2. NASA Datasets

### Primary remote-sensing input

- **Dataset:** MODIS/Terra Surface Reflectance Daily L2G Global 1 km and 500 m SIN Grid, Version 6.1
- **Short name:** `MOD09GA.061`
- **Earth Engine ID:** `MODIS/061/MOD09GA`
- **DOI:** `10.5067/MODIS/MOD09GA.061`

Used bands:

```text
b01: red
b02: NIR
b03: blue
b04: green
b05-b07: SWIR bands
state_1km: QA / cloud-state flags
```

### Burned-area proxy label

- **Dataset:** MODIS Burned Area Monthly Global 500 m, Version 6.1
- **Short name:** `MCD64A1.061`
- **Earth Engine ID:** `MODIS/061/MCD64A1`
- **Band used:** `BurnDate`

The `BurnDate > 0` mask is used as a proxy burned-area label for evaluating HDC tile-level burned/unburned separability.

---

## 3. Case Study: Dixie Fire, California

The prototype is evaluated on a dense 16-window time sequence around the 2021 Dixie Fire.

### Temporal windows

```text
pre_202104
pre_202105
pre_202106
fire_202107
fire_202108a
fire_202108b
post_202109
post_202110
rec_202204
rec_202206
rec_202304
rec_202306
rec_202404
rec_202406
rec_202504
rec_202506
```

The final presentation metrics use `fire_202108b` as the peak/later-fire evaluation window and exclude the invalid/fill-heavy `post_202109` row from recovery headline metrics.

---

## 4. Software Results

### 4.1 Dense Dixie Fire validation

```text
Number of observations: 16
Image size: 190 × 395
Bands used: 7 reflectance bands
Spatial tiles: 300
Burned tiles: 44
Peak evaluation window: fire_202108b
Peak evaluation date: 2021-08-30
```

### 4.2 Burned/unburned tile detection

Using MCD64A1 `BurnDate` as a proxy label, N-BAI/HDC tile-change scores separated burned and unburned tiles as follows:

| Metric | Value |
|---|---:|
| Accuracy | 97.33% |
| F1-score | 0.907 |
| True Positives | 39 |
| True Negatives | 253 |
| False Positives | 3 |
| False Negatives | 5 |
| HDC threshold | 0.03493 |

Interpretation:

> N-BAI captures wildfire-driven land-condition shifts by comparing binary tile signatures against a pre-fire baseline. The strong agreement with MCD64A1 BurnDate indicates that compact HDC signatures can recover burned-area structure at tile level.

### 4.3 Recovery monitoring

The vegetation recovery ratio is computed over burned pixels relative to the pre-fire NDVI baseline.

| Recovery Metric | Value |
|---|---:|
| Pre-fire burned-pixel NDVI baseline | 0.6780 |
| Post-fire reference window | post_202110 |
| Post-fire reference date | 2021-10-14 |
| Post-fire recovery ratio | 0.5123 |
| Latest June recovery window | rec_202506 |
| Latest June recovery date | 2025-06-17 |
| Latest June recovery ratio | 0.7036 |

Interpretation:

> Burned-region NDVI dropped to about 51% of the pre-fire baseline after the fire and recovered to about 70% of the pre-fire baseline by June 2025.

### 4.4 HDC separation

| HDC Metric | Value |
|---|---:|
| Max burned-tile HDC change | 0.08688 |
| Max unburned-tile HDC change | 0.02014 |
| Burned/unburned HDC peak ratio | 4.31× |

Interpretation:

> Burned tiles show a substantially larger HDC signature shift than unburned tiles, supporting the use of binary land-condition signatures for disturbance detection.

### 4.5 Memory, compression, and downlink

| Metric | Value |
|---|---:|
| Raw scene size estimate | 1026.1 KB |
| Raw 16-scene sequence estimate | 16.4 MB |
| Scene signature | 4096 bits = 512 bytes |
| Tile signature set | 153.6 KB |
| Scene + tile payload | 154.1 KB |
| Scene-level compression ratio | 2052× |
| Scene + tile payload compression ratio | 6.82× |
| Estimated downlink reduction | 18.08× |

Decision counts over the 16-observation sequence:

| Decision | Count |
|---|---:|
| `skip_or_low_priority` | 10 |
| `downlink_features_only` | 6 |
| `priority_downlink_full_or_features` | 0 |
| `request_reobserve` | 0 |

Interpretation:

> The current BITS policy keeps stable scenes at low priority and transmits compact feature/signature payloads for informative scenes, producing an estimated 18.08× reduction compared with downlinking all raw observations.

---

## 5. SID No-Reference Conditioning Metrics

Because paired cloud-free ground truth is not available for real MODIS scenes, SID is not evaluated with PSNR/SSIM in the real case study. Instead, the prototype uses no-reference and downstream HDC reliability metrics.

Selected examples:

```text
fire_202108a
fire_202108b
rec_202304
rec_202404
```

### 5.1 Image-conditioning metrics

Across selected hazy/cloud-affected scenes, SID increased:

- entropy,
- local contrast,
- sharpness / Laplacian variance.

Example: `rec_202304`

| Metric | Before SID | After SID |
|---|---:|---:|
| Local contrast | 0.0485 | 0.0731 |
| Sharpness / Laplacian variance | 0.0430 | 0.1326 |
| Entropy | 5.7944 | 5.8979 |

### 5.2 Spectral-index perturbation

SID also reports mean absolute changes in vegetation/burn-sensitive indices.

```text
Mean |ΔNDVI| ≈ 0.038–0.055
Mean |ΔNBR|  ≈ 0.044–0.060
```

Interpretation:

> SID improves local structure and visibility while keeping NDVI/NBR perturbations moderate.

### 5.3 Downstream HDC effect

| Metric | Before SID | After SID |
|---|---:|---:|
| Burned/unburned F1-score | 0.892 | 0.907 |
| Accuracy | 97.0% | 97.33% |
| True Positives | 37 | 39 |
| False Negatives | 7 | 5 |
| Unburned temporal std | 0.01767 | 0.01612 |

Interpretation:

> SID slightly improves downstream burned-tile detection F1, reduces missed burned tiles, and improves temporal stability over unburned regions.

---

## 6. Hardware / FPGA Results

The current hardware result corresponds to the **SC-HDC Dehazer** implementation.

### Vivado synthesis/implementation summary

| Design | LUTs | FFs | clk (MHz) | Est. Fmax (MHz) | CPD (ns) | Total Power (mW) | Area = LUT + FF | ADP = Area × CPD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SC-HDC Dehazer | 8,323 | 16,770 | 100 | 331.6 | 3.016 | 292 | 25,093 | 75,680 |

Interpretation:

> The SC-HDC Dehazer achieves an estimated maximum frequency of 331.6 MHz with 25,093 total LUT+FF resources and 292 mW estimated total power. These results provide a first hardware feasibility point for FPGA deployment of the SID/HDC front end.

### Hardware design components

The repository includes FPGA-oriented Verilog/SystemVerilog building blocks for the N-BAI/HDC layer:
![Hardware Schematic](./assets/hardware.png)

```text
hdc_xor_bind.sv
hdc_popcount.sv
hdc_hamming.sv
hdc_majority_bundle.sv
hdc_threshold_decision.sv
bits_hdc_top_demo.sv
sid3x3_condition_core.sv
```

The HDC hardware maps naturally to:

- XOR binding,
- majority bundling,
- popcount-based Hamming distance,
- threshold decision logic,
- compact binary memory.

A Verilog wrapper generator is also included:

```bash
python hardware/scripts/generate_verilog.py --D 4096 --K 8 --out hardware/rtl/generated
```

---

## 7. Repository Structure

```text
src/
  bits_sid_nbai_demo_v6_wildfire_fixed.py

scripts/
  gee_export_dixie_wildfire_dense.py
  gee_export_dixie_paper_rgb_dense.py
  bits_wildfire_report_tools.py
  bits_finalize_dense_wildfire_summary.py
  measure_sid_no_reference_metrics_fixed.py
  make_sid_north_up_figures_fixed.py

hardware/
  rtl/
    hdc_xor_bind.sv
    hdc_popcount.sv
    hdc_hamming.sv
    hdc_majority_bundle.sv
    hdc_threshold_decision.sv
    bits_hdc_top_demo.sv
    sid3x3_condition_core.sv
  tb/
    tb_bits_hdc_top_demo.sv
  scripts/
    generate_verilog.py

docs/
  methodology.md
  metrics.md

data/
  README.md
```

---

## 8. How to Reproduce

### 8.1 Install dependencies

```bash
conda create -n bits python=3.11 numpy pandas matplotlib rasterio -y
conda activate bits
pip install earthengine-api
earthengine authenticate
```

Set the Google Earth Engine project in the export scripts or by using Earth Engine CLI.

### 8.2 Export dense MODIS/MCD64A1 sequence

```bash
python scripts/gee_export_dixie_wildfire_dense.py --export
```

Download from Google Drive:

```text
BITS_exports/dixie_wildfire_dense_stack.tif
```

### 8.3 Convert GeoTIFF to NPZ

```bash
python scripts/gee_export_dixie_wildfire_dense.py   --convert   --tif dixie_wildfire_dense_stack.tif   --npz dixie_wildfire_dense.npz
```

### 8.4 Run SID + N-BAI

```bash
python src/bits_sid_nbai_demo_v6_wildfire_fixed.py   --mode wildfire_npz   --npz dixie_wildfire_dense.npz   --out runs/dixie_wildfire_dense_aug   --save-all-images   --eval-label-contains 202108b
```

### 8.5 Generate reports

```bash
python scripts/bits_wildfire_report_tools.py   --npz dixie_wildfire_dense.npz   --run-dir runs/dixie_wildfire_dense_aug   --out runs/dixie_wildfire_dense_aug

python scripts/bits_finalize_dense_wildfire_summary.py   --run-dir runs/dixie_wildfire_dense_aug
```

### 8.6 Generate SID no-reference metrics

```bash
python scripts/measure_sid_no_reference_metrics_fixed.py   --npz dixie_wildfire_dense.npz   --out runs/sid_no_reference_metrics   --eval-label fire_202108b
```

### 8.7 Generate north-up SID figures

```bash
python scripts/make_sid_north_up_figures_fixed.py   --npz dixie_wildfire_dense.npz   --tif dixie_wildfire_dense_stack.tif   --out runs/dixie_wildfire_dense_aug/paper_figures/sid_north_up
```

---

## 9. Notes and Limitations

- The burned-area labels are proxy labels from MCD64A1 `BurnDate`, not manually annotated ground truth.
- The HDC threshold values are currently selected using exploratory threshold sweeps.
- PSNR/SSIM are not used for real MODIS SID evaluation because no paired cloud-free ground truth is available.
- The hardware RTL is a starting point for FPGA synthesis and integration. The generic 4096-bit popcount should be pipelined or chunked for timing closure in a production FPGA design.
- Large GeoTIFF/NPZ files should not be committed to GitHub.

---

## 10. Recommended Dataset Citation Text

```text
MODIS/Terra Surface Reflectance Daily L2G Global 1 km and 500 m SIN Grid, Version 6.1,
MOD09GA.061, DOI: 10.5067/MODIS/MOD09GA.061.
```

For burned-area proxy labeling:

```text
MODIS Burned Area Monthly Global 500 m, Version 6.1, MCD64A1.061, BurnDate band.
```
