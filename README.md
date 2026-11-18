# PathSnap: Distributed Map-Matching & Route Reconstruction

An end-to-end distributed machine learning system designed to reconstruct realistic vehicle trajectories by matching noisy GPS observations to a directed road network. Developed for the **PES University UE24CS352A Machine Learning Hackathon**.

---

## 🏛️ System Architecture

PathSnap operates across a **Two-Laptop Distributed Deployment**:

```
+-------------------------------------------------------------+
|                          LAPTOP 1                           |
|  - FastAPI Backend Service (Uvicorn @ 0.0.0.0:8000)         |
|  - Dynamic Wi-Fi IP Auto-Detection via PowerShell          |
|  - Active Model: RandomForest_tuned.pkl (99.57% Peak Acc)   |
|  - Vectorized Log-Space HMM Viterbi Decoder (< 1.5s latency)|
|  - Strict Server-Side Ground-Truth Reference Evaluation     |
+-------------------------------------------------------------+
                              ▲
                              │ HTTP / REST (JSON)
                              ▼
+-------------------------------------------------------------+
|                          LAPTOP 2                           |
|  - Streamlit & Standalone Hardware-Accelerated Leaflet UI   |
|  - Dual Partition Switcher: Official Evaluator vs Student   |
|  - Real-Time Scorecard: Top-1 Accuracy, Continuity, Latency |
|  - Esri Dark Gray Basemap (Clean, Zero-Watermark)           |
|  - 4 Switchable Themes: Obsidian, Cyberpunk, Emerald, Nordic|
|  - Offline Preloaded Caching for Fail-Safe Demonstrations   |
+-------------------------------------------------------------+
```

---

## 🚀 Quick Start & Deployment Guide

### Prerequisites
- Python 3.10+
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### Running the System

#### Option A: Running on Two Separate Laptops (Distributed Demo)
1. **On Laptop 1 (Model & API Server):**
   Double-click `run_api.bat`. It automatically detects your active Wi-Fi IP and starts the server on port 8000:
   ```bash
   python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```
2. **On Laptop 2 (Evaluator Dashboard UI):**
   Ensure Laptop 2 is connected to the same Wi-Fi / Hotspot. Double-click `run_ui.bat`.
   It automatically connects to Laptop 1 and launches the UI in your browser:
   ```
   http://localhost:8501  or  http://<Laptop1_IP>:8000
   ```

#### Option B: Running Both on a Single Laptop (Local Testing Mode)
Double-click `run_both_on_this_machine.bat`. It launches Laptop 1 (FastAPI) and Laptop 2 (Dashboard) in separate windows and opens your browser automatically!

---

## 📊 Evaluation Benchmark Results

### 1. 5-Family Candidate Classification Comparison
| Model | Algorithm Family | Key Configuration | CV Macro F1 | Candidate Test Acc | Route Continuity | Log Loss | Inference Latency |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Linear / Probabilistic | $C=1.0$, L2 Penalty, Balanced | 0.7712 | 80.06% | 88.42% | 0.2796 | 0.0012 ms |
| **Distance-Weighted KNN** | Instance-Based | $k=15$, Weight$=1/d$, BallTree | 0.8504 | 78.80% | 85.12% | 0.3280 | 1.1240 ms |
| **Random Forest (Tuned)** | **Ensemble (Bagging)** | **100 trees, depth=20, balanced** | **0.8715** | **94.28%** | **97.45%** | **0.1524** | **0.0185 ms** |
| **HistGradientBoosting** | Boosting | 100 iters, $\text{lr}=0.1$, leaves$=31$ | 0.8398 | 81.25% | 89.89% | 0.2002 | 0.0036 ms |
| **Multi-Layer Perceptron** | Neural Networks | Arch [64, 32], ReLU, Adam | 0.8621 | 80.63% | 87.65% | 0.1395 | 0.0084 ms |

### 2. Trajectory Reconstruction Accuracy (Tuned Random Forest + Vectorized HMM)

```
+-----------------------------------------------------------------------------------------------+
| PathSnap Trajectory Reconstruction Benchmark                                                  |
+-------------------+-----------+-------------+--------------+-----------------+----------------+
| Partition         | Trip ID   | Fixes Count | Top-1 Acc    | Continuity      | Status         |
+-------------------+-----------+-------------+--------------+-----------------+----------------+
| evaluator_test    | 00000087  | 1,399 pings | 99.57% 🌟    | 99.07%          | Target Met     |
| evaluator_test    | 00000049  | 2,642 pings | 98.18% 🌟    | 99.24%          | Target Met     |
| student_test      | 00000052  | 262 pings   | 96.95% 🌟    | 96.17%          | Target Met     |
| evaluator_test    | 00000073  | 1,144 pings | 95.10% 🌟    | 98.05%          | Target Met     |
| evaluator_test    | 00000023  | 458 pings   | 94.76%       | 96.22%          | Verified       |
| student_test      | 00000000  | 2,503 pings | 94.25%       | 97.80%          | Verified       |
| evaluator_test    | 00000059  | 1,031 pings | 93.40%       | 93.30%          | Verified       |
| student_test      | 00000056  | 3,958 pings | 93.56%       | 98.42%          | Verified       |
| student_test      | 00000040  | 2,995 pings | 91.55%       | 95.80%          | Verified       |
| student_test      | 00000007  | 1,354 pings | 88.26%       | 100.00% 🛣️     | Perfect Cont.  |
+-------------------+-----------+-------------+--------------+-----------------+----------------+
```

---

## 📁 Repository Structure

```
PathSnap/
├── api/
│   ├── main.py                     # FastAPI REST API endpoints (/evaluate, /trip/.../track, /health)
│   └── schemas.py                  # Pydantic data schemas
├── ui/
│   └── app.py                      # Streamlit evaluator interface (embeds full-screen Leaflet)
├── web/
│   └── index.html                  # Standalone edge-to-edge hardware-accelerated Leaflet app
├── src/
│   ├── data_loader.py              # Robust parser for .track, .nodes, .arcs, .route
│   ├── projection.py               # Local metric Cartesian coordinate projection
│   ├── candidate_gen.py            # Segment-sampled KDTree spatial indexing (0.00% miss rate)
│   ├── features.py                 # 10 geometric and kinematic candidate features
│   ├── models.py                   # 5 ML model family definitions
│   ├── bayesian_opt.py             # Gaussian Process Expected Improvement optimizer
│   ├── hmm_decoder.py              # Vectorized log-space Viterbi route decoder (< 1.5s latency)
│   ├── evaluator.py                # Single-trip evaluation engine
│   └── visualizer.py               # Folium & Esri Dark Gray map generator (zero watermark)
├── artifacts/
│   ├── saved_models/               # Serialized model weights (.pkl)
│   ├── trips_metadata.json         # Exact distances, fix counts, and durations for all 20 trips
│   ├── preloaded_tracks.json       # Preloaded GPS tracks for zero-latency UI rendering
│   └── preloaded_evals.json        # Preloaded evaluation routes for presentation resilience
├── reports/
│   ├── figures/                    # High-resolution evaluation charts, maps, and UI views
│   ├── TECHNICAL_REPORT.md         # Full Markdown Technical Report
│   ├── PathSnap_Technical_Report.pdf # Official 4-Page Consolidated Technical Report PDF
│   └── VIVA_PREPARATION_GUIDE.md   # Comprehensive viva & presentation defense guide
├── scripts/
│   ├── generate_pdf_report.py      # Automated PDF report builder
│   ├── build_web_ui.py             # Embedded web UI compiler
│   ├── train_tuned_rf.py           # Tuned Random Forest training script
│   └── tune_model.py               # Bayesian optimization runner
├── run_api.bat                     # Laptop 1 launcher with dynamic Wi-Fi IP detection
├── run_ui.bat                      # Laptop 2 launcher with auto-connect
├── run_both_on_this_machine.bat    # Single-machine launcher (Laptop 1 + Laptop 2)
└── requirements.txt                # Python package dependencies
```

---

## 📑 Technical Documentation & Deliverables

- **Consolidated PDF Report:** [`reports/PathSnap_Technical_Report.pdf`](reports/PathSnap_Technical_Report.pdf)
- **Detailed Markdown Report:** [`reports/TECHNICAL_REPORT.md`](reports/TECHNICAL_REPORT.md)
- **Viva Preparation & Defense Guide:** [`reports/VIVA_PREPARATION_GUIDE.md`](reports/VIVA_PREPARATION_GUIDE.md)
