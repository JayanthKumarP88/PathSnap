# PathSnap: Distributed Map-Matching & Route Reconstruction

An end-to-end distributed machine learning system designed to reconstruct realistic vehicle trajectories by matching noisy GPS observations to a directed road network. Developed for the **PES University UE24CS352A Machine Learning Hackathon**.

---

## 🏛️ System Architecture

PathSnap operates across a **Two-Laptop Distributed Deployment**:

```
+-------------------------------------------------------------+
|                          LAPTOP 1                           |
|  - FastAPI Backend Service (Uvicorn @ 0.0.0.0:8000)         |
|  - Trained Model Artifacts (HistGradientBoosting Tuned)     |
|  - Road Network Topology Graph (Nodes & Arcs)               |
|  - HMM Viterbi Decoder (Topological Sequence Optimizer)     |
|  - Server-Side Reference Evaluation Engine                  |
+-------------------------------------------------------------+
                              ▲
                              │ HTTP / JSON
                              ▼
+-------------------------------------------------------------+
|                          LAPTOP 2                           |
|  - Streamlit Evaluator Dashboard                            |
|  - Reserved Evaluation Trip Selector (10 Trips)             |
|  - KPI Scorecards (Top-1 Accuracy, Route Continuity)        |
|  - Interactive Embedded Folium Map Visualizations           |
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

#### 1. On Laptop 1 (Model & API Server)
Double-click `run_api.bat` or run:
```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```
*Note: Laptop 1 binds to `0.0.0.0` so it is accessible to Laptop 2 across the local Wi-Fi / LAN network.*

#### 2. On Laptop 2 (Evaluator User Interface)
Double-click `run_ui.bat` or run:
```bash
python -m streamlit run ui/app.py --server.port 8501 --server.address 0.0.0.0
```
- In the sidebar, set **Laptop 1 API Base URL** to Laptop 1's LAN IP (e.g. `http://192.168.1.50:8000`).
- Select any of the 10 reserved evaluation trips (`00000000`, `00000007`, `00000009`, `00000040`, `00000052`, `00000056`, `00000065`, `00000076`, `00000077`, `00000091`).
- Click **"Run Reconstruction & Evaluate"** to trigger the real-time inference pipeline and view the interactive Folium map.

---

## 📁 Repository Structure

```
PathSnap/
├── api/
│   ├── main.py                     # FastAPI REST API endpoints
│   └── schemas.py                  # Pydantic data schemas
├── ui/
│   └── app.py                      # Streamlit evaluator interface
├── src/
│   ├── data_loader.py              # Robust parser for .track, .nodes, .arcs, .route
│   ├── projection.py               # Local metric Cartesian coordinate projection
│   ├── candidate_gen.py            # Segment-sampled KDTree spatial indexing
│   ├── features.py                 # 10 geometric and kinematic features
│   ├── models.py                   # 5 ML model family definitions
│   ├── bayesian_opt.py             # Gaussian Process Expected Improvement optimizer
│   ├── hmm_decoder.py              # Log-space Viterbi sequential route decoder
│   ├── evaluator.py                # Single-trip evaluation engine
│   └── visualizer.py               # Folium interactive multi-layer map builder
├── artifacts/
│   ├── saved_models/               # Serialized model weights (.pkl)
│   ├── dataset_inventory.csv       # Summary statistics of all partitions
│   ├── model_comparison.csv        # 5-family benchmark results
│   ├── bayesian_optimization_trials.csv # 20 GP-EI trial records
│   ├── bayesian_tuned_comparison.csv    # Untuned vs. tuned comparison
│   └── hmm_vs_classifier_comparison.csv # HMM vs classifier results across 10 trips
├── maps/                           # Generated interactive HTML maps
├── reports/
│   ├── figures/                    # High-resolution evaluation charts & confusion matrices
│   └── PathSnap_Technical_Report.pdf # Consolidated Technical Report PDF
├── scripts/
│   ├── preprocess_and_eda.py       # Preprocessing & EDA runner
│   ├── train_and_compare_models.py # 5-family training & Group CV runner
│   ├── tune_model.py               # Bayesian optimization runner
│   └── reconstruct_and_compare_hmm.py # HMM evaluation runner
├── run_api.bat                     # Laptop 1 one-click launcher
├── run_ui.bat                      # Laptop 2 one-click launcher
├── requirements.txt                # Python package dependencies
└── README.md                       # Project documentation
```

---

## 📊 Experimental Results Summary

### 1. 5-Family Model Comparison
| Model | Algorithm Family | Key Configuration | CV Macro F1 | Top-1 Accuracy | Route Continuity | Student-Test Macro F1 | Log Loss | Training Time (s) | Inference Latency (ms/sample) |
|---|---|---|---|---|---|---|---|---|---|
| **LogisticRegression** | Linear & Probabilistic | C=1.0, L2, balanced | 0.7567 | 80.06% | 88.93% | 0.7902 | 0.2796 | 0.78s | 0.0002 ms |
| **DistanceKNN** | Instance-based | k=15, distance-weighted, BallTree | 0.8046 | 78.80% | 88.60% | 0.8629 | 0.3280 | 0.09s | 1.1215 ms |
| **RandomForest** | Ensemble | 100 trees, max_depth=16, balanced | **0.8277** | 81.14% | **90.01%** | 0.8665 | 0.1674 | 15.06s | 0.0028 ms |
| **HistGradientBoosting** | Boosting | 100 iters, lr=0.1, max_leaves=31 | 0.8119 | **81.25%** | 89.89% | 0.8464 | 0.2002 | **2.03s** | **0.0036 ms** |
| **MLPClassifier** | Neural Networks | Arch [64, 32], ReLU, Adam | 0.8195 | 80.63% | 89.87% | **0.8794** | **0.1395** | 30.07s | 0.0019 ms |

### 2. Bayesian Hyperparameter Optimization (GP-EI)
- Selected Model: **HistGradientBoosting**
- Search Space: `learning_rate` [0.02, 0.25], `max_leaf_nodes` [15, 63], `min_samples_leaf` [10, 80], `l2_regularization` [0.001, 5.0]
- Optimization Result:
  - **Student-Test Macro F1**: Improved from $0.8464 \to \mathbf{0.8496}$
  - **Log Loss**: Improved from $0.2002 \to \mathbf{0.1875}$
  - **Optimal Hyperparameters**: `learning_rate=0.2439`, `max_leaf_nodes=63`, `min_samples_leaf=47`, `l2_regularization=1.1413`

### 3. HMM & Viterbi Sequential Route Reconstruction (10 Student-Test Trips)
- **Average Top-1 Accuracy**: Improved from $82.30\% \to \mathbf{88.96\%}$ ($\mathbf{+6.66\%}$ gain)
  - Peak accuracy on Trip 00000052: **$96.95\%$**
  - Peak accuracy on Trip 00000000: **$94.85\%$**
  - Peak accuracy on Trip 00000056: **$93.68\%$**
- **Average Route Continuity**: Improved from $89.58\% \to \mathbf{98.20\%}$ ($\mathbf{+8.63\%}$ gain)
  - Trip 00000007 reached **$100.00\%$** complete topological continuity!
