# BITS: Bitstream Intelligence for Space-to-Soil Earth Observation

BITS is a hardware-aware framework that combines **Stochastic Computing (SC)** and **Hyperdimensional Computing (HDC)** for low-power, real-time Earth observation. The system is designed for NASA-style space-to-soil sensing, enabling efficient image enhancement and classification on edge hardware.

---

## Overview

BITS integrates:
- **Stochastic image enhancement** (bitstream-based dehazing)
- **Hyperdimensional encoding** (4096-D hypervectors)
- **Hardware-friendly logic** (MUX, XOR/XNOR, popcount)

The pipeline avoids floating-point operations and is optimized for **FPGA deployment**.

---

## Repository Structure

### Hardware (Verilog)
- `top_hdc_sc.v` – Top-level SC + HDC system  
- `sc_dehaze.v` – Stochastic image enhancement  
- `hdc_core.v` – HDC encoding and classification  
- `hamming_sim.v` – Similarity computation  
- `popcount.v` – Pipelined popcount  
- `clk.xdc` – Clock constraint  

### Python (Reference)
- `run_satehaze_sc.py` – Main evaluation script  
- `sc_cloud_filter.py` – SC image processing pipeline  
- `sc_cloud_config.py` – Config parameters  
- `sc_cloud_utils.py` – Metrics and utilities  

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/BITS
cd BITS
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
pip install numpy opencv-python scipy scikit-image
