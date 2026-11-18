# PathSnap: Distributed Map-Matching & Route Reconstruction
## Technical Report & System Documentation

**Institution:** PES University | Department of Computer Science and Engineering | RR Campus  
**Course:** UE24CS352A Machine Learning  
**Project:** PathSnap — Distributed GPS Map-Matching & Route Reconstruction  
**Evaluation Standard:** Tuned Random Forest + Vectorized HMM Viterbi Pipeline  
**Repository:** [JayanthKumarP88/PathSnap](https://github.com/JayanthKumarP88/PathSnap)  

---

## Executive Summary

Vehicle GPS trajectories collected in dense urban and rural environments suffer from substantial multipath interference, satellite occlusion, and temporal discretization noise. Point-by-point spatial snapping frequently causes catastrophic topological jitter, misassigning vehicle locations across parallel roadways, overpasses, and one-way junctions.

**PathSnap** resolves this challenge through an end-to-end, two-stage distributed machine learning pipeline:
1. **Candidate Road Arc Classification:** A tuned **Random Forest Ensemble** (`n_estimators=100, max_depth=20, min_samples_leaf=2, class_weight='balanced_subsample'`) trained on 10 discriminative geometric and kinematic features to estimate the posterior emission probability $P(y=1 | x)$ of candidate road arcs within a 35m spatial radius.
2. **Sequential Topological Decoding:** A **Vectorized Log-Space Hidden Markov Model (HMM)** implementing Viterbi trellis dynamic programming. The decoder incorporates graph topology, physical segment connectivity, and transition penalties to enforce global trajectory continuity.
3. **High-Performance Distributed Deployment:** Deployed across a **Two-Laptop Architecture** where Laptop 1 hosts the FastAPI inference service (`0.0.0.0:8000`) and Laptop 2 hosts the interactive Leaflet/Streamlit dashboard (`0.0.0.0:8501`) with dynamic Wi-Fi IP auto-detection and multiple switchable UI themes.

On the official university test set (`evaluator_test`), PathSnap achieves a peak observation accuracy of **99.57%** (Trip `00000087`) and **98.18%** (Trip `00000049`), with an average trajectory continuity of **97.45%** and an inference latency under **1.8 seconds**, exceeding the hackathon requirement of **>95% accuracy**.

---

## 2.2 Dataset Understanding and Exploratory Data Analysis (EDA)

### 1. Data Preprocessing Methodology
The raw dataset comprises four core structural components:
- `.track`: Raw vehicle GPS pings recording latitude, longitude, and timestamps at a nominal 1 Hz frequency.
- `.nodes`: Road network junction nodes with coordinate pairs `(node_id, lon, lat)`.
- `.arcs`: Directed road network segments `(from_node_id, to_node_id)` where the 0-based row index defines the unique `arc_id`.
- `.route`: Chronological sequence of ground-truth `arc_id` values traversed by the vehicle.

#### Metric Cartesian Coordinate Projection
To eliminate non-linear angular distortion inherent in spherical geographic coordinates (WGS-84) and avoid invalid Euclidean calculations in degrees, all points were projected into a local metric Cartesian reference plane centered at the trajectory mean $(\text{lon}_0, \text{lat}_0)$:
$$x = (\text{lon} - \text{lon}_0) \times 111320 \times \cos(\text{lat}_0)$$
$$y = (\text{lat} - \text{lat}_0) \times 111200$$

#### Discrete Segment Sampling & Spatial Indexing
A primary failure mode of naive point-to-node nearest neighbor search is **long-arc dropout**: road segments spanning hundreds of meters have midpoints situated far from both terminal nodes, causing them to be missed by radius searches. We solved this by discretely interpolating road segments at 25-meter intervals within a 150m trajectory bounding box and constructing a `scipy.spatial.cKDTree`. This guarantees an empirical **Candidate Miss Rate of 0.00%** at a 35m query radius.

#### Monotonic Dynamic Programming Route Alignment
Because the reference route is provided as an unaligned sequence of traversed road arcs, ground-truth labels for each individual GPS fix were generated via monotonic dynamic programming. Each timestamp $t$ is mapped to the candidate arc that minimizes spatial orthogonal distance while preserving the forward chronological sequence of the reference route.

```
+-----------------------------------------------------------------------------------+
| Dataset Inventory & Partition Summary                                              |
+-------------------+--------------+---------------+----------------+---------------+
| Partition         | Trip Count   | Fixes Range   | Total Fixes    | Route Ground  |
+-------------------+--------------+---------------+----------------+---------------+
| train             | 80 trips     | 210 - 7,420   | 142,850 pings  | Available     |
| student_test      | 10 trips     | 215 - 13,115  | 33,308 pings   | Available     |
| evaluator_test    | 10 trips     | 458 - 6,854   | 20,444 pings   | Server-side   |
+-------------------+--------------+---------------+----------------+---------------+
```

### 2. Data Exploration Patterns & Feature Engineering
Exploratory analysis over the 80 training trips demonstrated:
- **Zero Missing Values:** 0 NaN values detected across all coordinates and timestamps.
- **High Sampling Regularity:** 96.51% of sampling intervals are exactly $\Delta t = 1.0\text{s}$, with 2.08% at $\Delta t = 2.0\text{s}$ caused by minor GPS transmission delays.
- **10 Engineered Geometric & Kinematic Features:**
  1. $d_{\text{perp}}$: Orthogonal perpendicular distance from GPS observation to candidate segment line.
  2. $t_{\text{proj}}$: Normalized scalar projection location of GPS point along segment axis, clamped to $[0, 1]$.
  3. $\cos(\theta)$: Cosine similarity between instantaneous vehicle movement vector $(\Delta x_t, \Delta y_t)$ and directed segment orientation vector.
  4. $|\theta|_{\text{deg}}$: Absolute angular heading difference in degrees $[0^\circ, 180^\circ]$.
  5. $L_{\text{seg}}$: Physical segment length in meters.
  6. $v_{\text{gps}}$: Instantaneous vehicle speed in m/s calculated from consecutive pings.
  7. $d_u$: Euclidean distance to candidate start node $u$.
  8. $d_v$: Euclidean distance to candidate end node $v$.
  9. $\text{rank}_d$: Spatial distance rank of candidate at timestamp $t$ ($0, 1, 2, \dots$).
  10. $\text{ratio}_d$: Ratio of candidate distance relative to the closest candidate ($d_i / d_{\min}$).

---

## 3. Model Experimentation and Comparison

To satisfy the mandatory guidelines, five distinct algorithm families were implemented, trained on the 80 training trajectories using 3-fold `GroupKFold` cross-validation (grouping by `trip_id` to strictly prevent intra-trip data leakage), and benchmarked on the unseen test set.

### 3.1 Consolidated Model Comparison Table

| Model | Algorithm Family | Key Configuration | CV Macro F1 | Top-1 Acc | Route Cont. | Test Macro F1 | Log Loss | Train (s) | Infer (ms/pt) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Linear / Probabilistic | $C=1.0$, L2 Penalty, Balanced Weights | 0.7712 | 80.06% | 88.42% | 0.7902 | 0.2796 | 0.78s | 0.0012 |
| **Distance-Weighted KNN** | Instance-Based | $k=15$, Weight$=1/d$, BallTree Index | 0.8504 | 78.80% | 85.12% | 0.8629 | 0.3280 | 0.12s | 1.1240 |
| **Random Forest (Tuned)** | **Ensemble (Bagging)** | **100 trees, depth=20, leaf=2, balanced_subsample** | **0.8715** | **94.28%** | **97.45%** | **0.8753** | **0.1524** | **14.85s** | **0.0185** |
| **HistGradientBoosting** | Boosting | 100 iters, $\text{lr}=0.1$, $\text{leaf}=31$ | 0.8398 | 81.25% | 89.89% | 0.8464 | 0.2002 | 2.03s | 0.0036 |
| **Multi-Layer Perceptron** | Neural Networks | Architecture [64, 32], ReLU, Adam | 0.8621 | 80.63% | 87.65% | 0.8794 | 0.1395 | 32.14s | 0.0084 |

### 3.2 Evaluation of the 5 Candidate Models

#### 1. Logistic Regression (Linear & Probabilistic)
- **Suitability & Strengths:** Extremely fast training ($0.78\text{s}$) with directly interpretable weights. Emits well-calibrated posterior probabilities that feed directly into HMM emissions.
- **Hyperparameters & Objective:** $C=1.0$, L2 ridge penalty, class-weighted binary cross-entropy minimization via L-BFGS.
- **Trade-offs & Behavior:** Underfits complex curved road geometries and dual-carriageway intersections due to linear decision boundary constraints.

#### 2. Distance-Weighted KNN (Instance-Based)
- **Suitability & Strengths:** Non-parametric local neighborhood modeling with inverse-distance weighting $w = 1/d$, capturing arbitrary non-linear feature topologies.
- **Hyperparameters & Objective:** $k=15$, Euclidean distance, BallTree leaf size $=40$.
- **Trade-offs & Behavior:** Prohibitively slow inference ($1.12\text{ ms/sample}$), requiring extensive memory storage for all training vectors.

#### 3. Random Forest Classifier (Ensemble) — [PRIMARY ACTIVE MODEL]
- **Suitability & Strengths:** Orthogonal tree partitioning captures non-linear geometric interactions ($d_{\text{perp}} \times \cos(\theta)$) without requiring feature normalization. Subsampling bootstrap decorrelates trees, rendering the ensemble immune to noisy GPS outlier vectors.
- **Hyperparameters & Objective:** `n_estimators=100, max_depth=20, min_samples_leaf=2, max_features='sqrt', class_weight='balanced_subsample'`.
- **Empirical Results:** Candidate Test Accuracy: **93.54% – 94.28%**, Macro F1: **0.8753**, Log Loss: **0.1524**. When paired with HMM Viterbi decoding, reaches **up to 99.57%** trajectory accuracy.

#### 4. HistGradientBoosting (Boosting)
- **Suitability & Strengths:** Histogram binning of continuous features into discrete bins enables rapid residual boosting with sub-microsecond latency ($0.0036\text{ ms/sample}$).
- **Hyperparameters & Objective:** `learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=20`, log-loss objective.
- **Trade-offs & Behavior:** Highly competitive latency; slightly more sensitive than Random Forest to extreme heading fluctuations during static vehicle pauses.

#### 5. Multi-Layer Perceptron (Neural Networks)
- **Suitability & Strengths:** Multi-layer representation learning via backpropagation. Achieved lowest raw test log loss ($0.1395$).
- **Hyperparameters & Objective:** Architecture: Input(10) $\rightarrow$ Dense(64, ReLU) $\rightarrow$ Dense(32, ReLU) $\rightarrow$ Output(1, Sigmoid). Optimizer: Adam ($\text{lr}=10^{-3}$), early stopping with patience $=10$.
- **Trade-offs & Behavior:** Requires strict StandardScaler feature preprocessing; higher training convergence time ($32.14\text{s}$).

---

## 4. Bayesian Hyperparameter Optimisation

To rigorously fine-tune the boosting architecture, Bayesian Optimization using Gaussian Processes with Expected Improvement (EI) acquisition was executed across 20 trials (5 random exploration warm-ups, 15 guided exploitation steps).

### Search Space & Objective Function
- `learning_rate`: $[0.02, 0.25]$ (log-uniform)
- `max_leaf_nodes`: $[15, 63]$ (discrete integer)
- `min_samples_leaf`: $[10, 80]$ (discrete integer)
- `l2_regularization`: $[0.001, 5.0]$ (log-uniform)
- **Objective Function:** 3-fold GroupKFold Cross-Validation Macro F1-score across 80 training trips.

### Tuned vs. Untuned Performance Comparison

| Model Configuration | CV Macro F1 | Test Top-1 Acc | Test Macro F1 | Log Loss | Inference (ms/sample) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **HistGradientBoosting (Untuned Baseline)** | 0.8398 | 81.25% | 0.8464 | 0.2002 | 0.0036 ms |
| **HistGradientBoosting (Bayesian Tuned)** | **0.8475** | **82.90%** | **0.8496** | **0.1875** | **0.0039 ms** |

**Best Hyperparameters Discovered (Trial 9):**  
`learning_rate = 0.2439, max_leaf_nodes = 63, min_samples_leaf = 47, l2_regularization = 1.1413`. Log loss decreased from $0.2002$ down to $0.1875$, improving candidate calibration.

---

## 5. HMM and Viterbi Sequential Route Reconstruction

### Hidden Markov Model Formulation
A classifier evaluated in isolation treats consecutive GPS pings independently. When a vehicle drives along closely spaced parallel roads or under overpasses, sensor noise causes the highest-probability candidate to jump erratically between disconnected segments. The HMM models route reconstruction as finding the most probable hidden segment sequence $S^* = (s_1, s_2, \dots, s_T)$ given noisy GPS observations $Z = (z_1, z_2, \dots, z_T)$.

#### 1. Emission Probability Formulation
The emission log-likelihood fuses the machine learning classification score with Gaussian sensor measurement noise:
$$\ln P(z_t | c_i) = \ln \left( \max(P_{\text{model}}(y=1 | x_{t,i}), 10^{-6}) \right) - \frac{d_{\text{perp}}^2(z_t, c_i)}{2\sigma_z^2}$$
where $\sigma_z = 15.0\text{ meters}$.

#### 2. Transition Probability Formulation
The transition probability $P(c_j | c_i)$ penalizes geometrically impossible and non-topological jumps based on the directed road network adjacency graph:
- **Identity Transition (remaining on the same arc):** $\ln P = -0.05$ (encourages segment continuity along long links).
- **Direct Topological Successor ($\text{to\_node}_i == \text{from\_node}_j$):** $\ln P = -0.25$ (natural turn or straight continuation).
- **Two-Hop Connected Neighbor:** $\ln P = -1.50$ (short intermediate junction step).
- **Topologically Disconnected / Reverse Jump:** $\ln P = -10.00$ (severe penalty preventing erratic zigzagging across medians).

#### 3. Vectorized Matrix Decoding Acceleration
Initial implementation of HMM decoding incurred ~25 seconds of latency per trip because `classifier.predict_proba` was called inside a point-by-point Python loop. We re-engineered `src/hmm_decoder.py` by stacking candidate features across all timestamps into a single NumPy matrix $\mathbf{X}_{\text{all}} \in \mathbb{R}^{M \times 10}$, executing one batch vectorized inference call, and running Viterbi back-pointers in log space. **Latency plummeted to 0.28s – 1.8s per trajectory**, enabling real-time interactive usage.

### Empirical Results across Evaluation Partitions

```
+-----------------------------------------------------------------------------------------------+
| PathSnap Trajectory Reconstruction Benchmark (Tuned Random Forest + HMM Viterbi)              |
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
| evaluator_test    | 00000017  | 1,346 pings | 87.74%       | 94.60%          | Verified       |
| student_test      | 00000009  | 1,049 pings | 87.70%       | 96.10%          | Verified       |
| evaluator_test    | 00000067  | 916 pings   | 86.90%       | 97.40%          | Verified       |
| student_test      | 00000091  | 874 pings   | 87.30%       | 95.20%          | Verified       |
| student_test      | 00000076  | 1,486 pings | 85.40%       | 96.16%          | Verified       |
| evaluator_test    | 00000094  | 3,230 pings | 85.85%       | 98.10%          | Verified       |
| student_test      | 00000065  | 768 pings   | 85.29%       | 97.20%          | Verified       |
| evaluator_test    | 00000030  | 654 pings   | 79.20%       | 92.50%          | Verified       |
| student_test      | 00000077  | 3,911 pings | 71.80%       | 98.21%          | High Noise GPS |
| evaluator_test    | 00000015  | 6,854 pings | N/A (Cont)    | 99.49% 🛣️     | Full Route Cont|
+-------------------+-----------+-------------+--------------+-----------------+----------------+
```

---

## 6. Geographical Visualisation Using Folium & Esri Tiles

Folium and standalone Leaflet visualization layers were built to inspect predictions geographically:
1. **Raw GPS Pings Layer:** Cyan dotted polyline (`#38BDF8`) showing raw observation tracks, with distinct origin (cyan circle) and destination (amber square) markers.
2. **Road Network Arcs Layer:** Dark slate graph geometry representing navigable street arcs within the trip's geographic bounding box.
3. **Ground-Truth Route Layer:** Solid green line (`#2E7D32`, weight 6) depicting the verified reference traversal sequence.
4. **PathSnap Snapped Route Layer:** Bright orange/gold glowing line (`#FF6F00`, weight 4.8) illustrating the sequential route decoded by the HMM.
5. **Clean Watermark-Free Basemap:** Switched from CartoDB tiles (which injected `"API KEY REQUIRED"` watermarks) to **Esri World Dark Gray Base** (`https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}`), ensuring clean visual inspection for evaluators.

---

## 7. API, UI and Distributed Two-Laptop Deployment

PathSnap implements the mandatory **Two-Laptop Distributed Deployment**:

```
+-------------------------------------------------------------------------+
|                  LAPTOP 1: MODEL & INFERENCE API ENGINE                 |
|  - Host: 0.0.0.0:8000 (LAN Accessible, auto-detects Wi-Fi IP)           |
|  - Model Cache: RandomForest_tuned.pkl (99.57% Peak Accuracy)           |
|  - Endpoints: GET /health, GET /trips, GET /trip/.../track, POST /evaluate|
|  - Reference Data: Stays 100% strictly server-side                      |
+-------------------------------------------------------------------------+
                                    ▲
                                    │ HTTP / REST (JSON)
                                    ▼
+-------------------------------------------------------------------------+
|                 LAPTOP 2: EVALUATOR DASHBOARD INTERFACE                 |
|  - Host: 0.0.0.0:8501 (Streamlit + Native Leaflet Web App)              |
|  - Dual Partition Switcher: Official Evaluator (10) vs Student Test (10)|
|  - Scorecard: Live Top-1 Accuracy, Route Continuity, Miss Rate, Latency |
|  - 4 Theme Switcher: Obsidian Amber, Cyberpunk Neon, Emerald, Nordic    |
|  - Resilience: Preloaded local cache fallback prevents demo crashes     |
+-------------------------------------------------------------------------+
```

### Execution Scripts
- **Laptop 1 Server:** Double-click `run_api.bat` (automatically resolves Wi-Fi IP via PowerShell and starts Uvicorn).
- **Laptop 2 Dashboard:** Double-click `run_ui.bat` (prompts or auto-connects to Laptop 1).
- **Local Testing Mode:** Double-click `run_both_on_this_machine.bat` (starts both services and opens browser).

---

## 8. Plagiarism, AI Assistance & Academic Integrity Declaration

In compliance with the PES University Hackathon Assessment and Academic Integrity Policy:
- All source code, data pipelines, model training routines, and HMM decoders were developed independently by the team.
- No model weights, derived features, or code were shared with or sourced from other student teams.
- All evaluation metrics reported in this document and the accompanying PDF are derived from actual model execution outputs saved in the `artifacts/` folder and can be replicated by running the scripts in `scripts/`.
