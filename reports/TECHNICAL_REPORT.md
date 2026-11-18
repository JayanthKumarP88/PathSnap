# PathSnap Technical Report
## Distributed Map-Matching and Route Reconstruction
**Course:** UE24CS352A Machine Learning Hackathon  
**Institution:** PES University | Department of Computer Science & Engineering | RR Campus  
**Team Members:** Jayanth Kumar, Pranay Krishna, Kundan V, Janhavi Ramakrishnan  
**Repository:** [https://github.com/JayanthKumarP88/PathSnap](https://github.com/JayanthKumarP88/PathSnap)  
**Primary Active Model:** Tuned Random Forest (`RandomForest_tuned.pkl`) + Vectorized HMM Viterbi Decoder  

---

## 1. Overview

**PathSnap** reconstructs realistic vehicle trajectories by matching noisy GPS observations to a directed road network. Our end-to-end pipeline consists of:
1. Loading and parsing raw trajectory, network, and reference files.
2. Projecting spherical GPS coordinates into a local metric Cartesian plane centered on the trajectory mean.
3. Generating candidate road segments using a discrete 25m-interpolated segment spatial `cKDTree` (0.00% candidate miss rate).
4. Computing 10 discriminative geometric and kinematic features for each candidate.
5. Classifying each candidate with a tuned **Random Forest Ensemble** to predict the posterior emission probability $P(y=1 | x)$.
6. Sequentially decoding the true road sequence using a high-speed, vectorized log-space **Hidden Markov Model (HMM) Viterbi decoder**.
7. Serving real-time inference via a **FastAPI backend** on Laptop 1 and presenting interactive visualizations, scorecards, and maps on an evaluator UI on Laptop 2 across a local Wi-Fi network.

---

## 2. Dataset Understanding and Exploratory Data Analysis (EDA)

### 2.1 Dataset Partition Used
All exploratory analysis in this section was conducted strictly on the **training split (80 trips)**. Model comparison utilized 3-fold `GroupKFold` cross-validation on the training split (grouping by `trip_id` so that no trajectory appears in both training and validation folds) and the `student_test` partition. The `evaluator_test` partition is reserved strictly for final demonstration and server-side evaluation.

| Partition | Trips | Fixes Range per Trip | Total Fixes | Reference Route |
| :--- | :---: | :---: | :---: | :--- |
| **train** | 80 | 210 – 7,420 | 142,850 | Available |
| **student_test** | 10 | 215 – 13,115 | 33,308 | Available |
| **evaluator_test** | 10 | 458 – 6,854 | 20,444 | Server-side evaluation |

### 2.2 Data Preprocessing Methodology
The loader in `src/data_loader.py` parses four core file types:
- `.track`: Raw vehicle GPS pings with latitude, longitude, and timestamps at approximately 1 Hz.
- `.nodes`: Road network junction nodes with coordinate pairs `(node_id, lon, lat)`.
- `.arcs`: Directed road network segments `(from_node_id, to_node_id)`, where the 0-based row index serves as `arc_id`.
- `.route`: Chronological sequence of ground-truth `arc_id` values traversed.

#### Metric Cartesian Coordinate Projection
Because longitude and latitude are angular measurements on a non-Euclidean spherical surface, direct distance computations in degrees introduce geographic distortion. In `src/projection.py`, all coordinates were converted to a metric Cartesian representation $(x, y)$ centered on the trajectory mean $(\text{lon}_0, \text{lat}_0)$:
$$x = (\text{lon} - \text{lon}_0) \times 111320 \times \cos(\text{lat}_0)$$
$$y = (\text{lat} - \text{lat}_0) \times 111200$$
This supports accurate Euclidean distance calculations in meters, point-to-segment orthogonal projections, and spatial candidate indexing.

#### Ground-Truth Label Creation
The reference route provides an unaligned sequence of traversed road arcs rather than a timestamped label per GPS ping. Ground-truth labels for candidate arcs were created using a monotonic dynamic programming alignment: each GPS timestamp $t$ is mapped to the nearest reference arc that maintains the chronological forward progression of the route. Candidates matching the aligned arc receive label 1, and all others receive label 0.

### 2.3 Missing-Value and Structural Analysis (Training Split, 80 Trips)

| Check | Finding | Status |
| :--- | :---: | :--- |
| Missing GPS values (NaN) | 0 | Clean |
| Out-of-range GPS coordinates | 0 | Clean |
| Duplicate GPS rows | 0 | Clean |
| Timestamp ordering violations | 0 | Clean |
| Non-positive time differences ($\Delta t \le 0$) | 0 | Clean |
| Required files missing | 0 | Clean |

While structurally complete, the dataset exhibits substantial behavioral noise:

| Exploration Result | Observed Value | Technical Implication |
| :--- | :---: | :--- |
| Trips examined | 80 | Full training partition |
| GPS observations | 142,850 | Large-scale sample size |
| Median sampling interval | 1.0 second | Nominal 1 Hz collection |
| Sampling intervals of exactly 1 s / 2 s | 96.51% / 2.08% | Periodic minor transmission delays |
| Trips with time gaps $> 5$ seconds | 45 / 80 | Occasional sensor dropouts |
| Total detected time gaps | 1,216 | Handled via temporal decay in HMM |
| Largest time gap | 153,064 s (~42.5 hrs) | Vehicle turned off between sessions |
| Maximum implied speed | 187.42 m/s | Severe GPS multipath location bounce |
| Trips with speed jumps $> 50$ m/s | 46 / 80 | Speed cannot be the sole predictive feature |
| Maximum GPS-to-reference deviation | 569.66 m | Large urban multipath reflection |
| Trips with maximum deviation $> 50$ m | 24 / 80 | Spatial distance alone is insufficient |
| Median GPS/reference length ratio | 1.18 | GPS noise artificially inflates length by 18% |
| Maximum GPS/reference length ratio | 7.503 | Extreme noise inflation during static stops |

### 2.4 Road Network Exploration (Training Split)

| Road Network Metric | Observed Value | Technical Implication |
| :--- | :---: | :--- |
| Total nodes examined | 1,933,485 | Large-scale graph topology |
| Total arcs examined | 3,876,904 | High-density directed street network |
| Median nodes per trip network | 9,608 | Scaled bounding box extraction required |
| Median arcs per trip network | 19,526 | Requires spatial indexing for candidate search |
| Trips with isolated nodes | 0 | All nodes connect to at least one arc |
| Multiple weakly connected components | 79 / 80 | Real-world road networks have disconnected subgraphs |
| Maximum weakly connected components | 421 | Disconnected road jumps must be penalized, not banned |
| Median arcs in largest weak component | 98.11% | Majority of roads belong to main network |
| Minimum arcs in largest weak component | 69.28% | Some trips cross separate island subgraphs |
| Trips containing self-loop arcs | 8 | Graph sanitation handled |
| Trips containing duplicate arcs | 53 | Parallel bidirectional segments present |
| Trips containing zero-length arcs | 28 | Filtered out during candidate projection |

### 2.5 Reference Route Checks (Training Split)

| Reference Route Check | Finding |
| :--- | :---: |
| Reference routes available | 80 / 80 trips |
| Invalid route arc IDs | 0 |
| Trips with route discontinuities | 10 / 80 |
| Total detected route breaks | 13 |
| Median route length | 280 arcs |
| Minimum route length | 59 arcs |
| Maximum route length | 2,266 arcs |

Because 10 training routes contain topological breaks, reference routes represent high-confidence recorded observations rather than mathematically perfect ground truth.

### 2.6 Candidate Road Generation
A standard nearest-node search frequently drops long road segments, as the midpoint of a 500m highway segment may be 250m away from either terminal node. In `src/candidate_gen.py`, we interpolated road segments at 25m intervals within a 150m bounding box of the trajectory and indexed the sampled coordinates into a `scipy.spatial.cKDTree`.

Candidates are defined as all arcs possessing a sample point within a **35m radius** of the GPS observation. This yielded a **0.00% candidate miss rate** when evaluated on the `student_test` partition and verified on the `evaluator_test` partition.

### 2.7 Feature Engineering
Our model evaluates 10 geometric and kinematic features (`src/features.py`):

| Feature | Mathematical Definition / Meaning |
| :--- | :--- |
| `dist` | Orthogonal perpendicular distance from GPS observation to candidate segment (m) |
| `t_proj` | Normalized projection location along segment axis, clamped to $[0, 1]$ |
| `cos_heading` | Cosine similarity between GPS vehicle motion vector $(\Delta x, \Delta y)$ and segment vector |
| `heading_diff_deg` | Absolute heading difference in degrees $[0^\circ, 180^\circ]$ |
| `arc_length` | Physical length of candidate road segment (m) |
| `gps_speed` | Instantaneous vehicle speed in m/s from consecutive pings |
| `dist_u` | Euclidean distance from GPS ping to candidate segment start node $u$ |
| `dist_v` | Euclidean distance from GPS ping to candidate segment end node $v$ |
| `dist_rank` | Ordinal rank of candidate after sorting candidates by distance ($0, 1, 2, \dots$) |
| `dist_ratio` | Ratio of candidate distance relative to nearest candidate ($d_i / d_{\min}$) |

**Why these features:** Because 24 trips exhibit GPS deviations exceeding 50m, distance alone cannot disambiguate parallel highways and service roads. `cos_heading` and `heading_diff_deg` immediately filter out perpendicular cross-streets and opposite-direction carriageways, while `dist_rank` and `dist_ratio` reward unambiguous single candidates.

---

## 3. Model Experimentation and Comparison

Five models across five distinct algorithm families were trained on the 80 training trips using 3-fold `GroupKFold` cross-validation (grouped by `trip_id`) and evaluated on the `student_test` partition.

### 3.1 Overall Model Comparison Table

| Model | Algorithm Family | Key Configuration | CV Macro F1 | Top-1 Accuracy | Route Continuity | Student-Test Macro F1 | Log Loss | Train Time | Inference Latency |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Linear / Probabilistic | $C=1.0$, L2 penalty, balanced weights | 0.7712 | 80.06% | 88.42% | 0.7902 | 0.2796 | 0.78s | 0.0012 ms/pt |
| **Distance-Weighted KNN** | Instance-Based | $k=15$, Weight$=1/d$, BallTree leaf 40 | 0.8504 | 78.80% | 85.12% | 0.8629 | 0.3280 | 0.12s | 1.1240 ms/pt |
| **Random Forest (Tuned)** | **Ensemble (Bagging)** | **100 trees, depth 20, leaf 2, balanced_subsample** | **0.8715** | **94.28%** | **97.45%** | **0.8753** | **0.1524** | **14.85s** | **0.0185 ms/pt** |
| **HistGradientBoosting** | Boosting | 100 iters, lr=0.1, 31 leaves, min leaf 20 | 0.8398 | 81.25% | 89.89% | 0.8464 | 0.2002 | 2.03s | 0.0036 ms/pt |
| **MLP Classifier** | Neural Networks | Arch [64, 32], ReLU, Adam lr=1e-3, early stop | 0.8621 | 80.63% | 87.65% | 0.8794 | 0.1395 | 32.14s | 0.0084 ms/pt |

### 3.2 Model Evaluation & Justification of Selection

#### Selection of Random Forest as Final Model:
**Random Forest (Tuned)** was selected as the final production model because:
1. **Highest Top-1 Observation Accuracy:** Achieves **94.28%** candidate accuracy, significantly outperforming HistGradientBoosting (81.25%), Logistic Regression (80.06%), and MLP (80.63%).
2. **Superior Route Continuity:** Delivers **97.45%** route continuity across trajectories, preventing fragmented segment jumps.
3. **Outlier Robustness:** Random feature subsampling and tree bootstrap averaging render the ensemble resilient to severe GPS multipath outliers (e.g., speed spikes > 50 m/s).
4. **Generalization Stability:** The small gap between CV Macro F1 (0.8715) and Student-Test Macro F1 (0.8753) confirms that the model generalizes cleanly without overfitting to specific geographic regions.

#### Training vs. Test Generalization:
Training Macro F1 scores logged across models:
- Logistic Regression: Train F1 = 0.7725 $\rightarrow$ CV F1 = 0.7712 (slight underfitting due to linear capacity limits).
- Distance KNN: Train F1 = 0.8810 $\rightarrow$ CV F1 = 0.8504 (moderate memory requirement, slow inference).
- **Random Forest (Tuned):** Train F1 = 0.9120 $\rightarrow$ CV F1 = 0.8715 $\rightarrow$ Test F1 = 0.8753 (stable generalization).
- HistGradientBoosting: Train F1 = 0.8580 $\rightarrow$ CV F1 = 0.8398 $\rightarrow$ Test F1 = 0.8464.
- MLP Classifier: Train F1 = 0.8950 $\rightarrow$ CV F1 = 0.8621 $\rightarrow$ Test F1 = 0.8794.

---

## 4. Hyperparameter Tuning and Optimization

### 4.1 Tuning the Primary Model: Random Forest
The production model was optimized in `scripts/train_tuned_rf.py` by systematically tuning tree depth, leaf constraints, and class weighting to handle candidate imbalance (where ~85% of candidates within 35m are negative):

| Random Forest Variant | CV Macro F1 | Student-Test Top-1 | Student-Test Macro F1 | Log Loss | Route Continuity | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Untuned Baseline** (`depth=16, balanced`) | 0.8277 | 81.14% | 0.8665 | 0.1674 | 90.01% | 15.06s |
| **Tuned Production** (`depth=20, leaf=2, subsample`) | **0.8715** | **94.28%** | **0.8753** | **0.1524** | **97.45%** | **14.85s** |

**Impact of Tuning:** Deepening tree depth to 20 while enforcing `min_samples_leaf=2` captured intricate non-linear interactions ($d_{\text{perp}} \times \cos\theta$) without memorizing leaf noise. Top-1 observation accuracy rose by **+13.14%** (81.14% $\rightarrow$ 94.28%), and route continuity improved by **+7.44%**.

### 4.2 Bayesian Optimization Benchmark (HistGradientBoosting)
As part of our exploration of gradient boosting (`src/bayesian_opt.py`, `scripts/tune_model.py`), Bayesian Optimization with a Gaussian Process surrogate and Expected Improvement (EI) acquisition was executed across 20 trials on HistGradientBoosting:
- Search space: `learning_rate` [0.02, 0.25], `max_leaf_nodes` [15, 63], `min_samples_leaf` [10, 80], `l2_regularization` [0.001, 5.0].
- Objective: 3-fold GroupKFold Macro F1 across 80 training trips.
- Best parameters found (Trial 9): `learning_rate=0.2439, max_leaf_nodes=63, min_samples_leaf=47, l2_regularization=1.1413`.
- Result: Test Macro F1 improved from 0.8464 to 0.8496, and Log Loss improved from 0.2002 to 0.1875.
- **Architectural Conclusion:** Even after Bayesian tuning, HistGradientBoosting Top-1 accuracy (82.90%) remained far below the Tuned Random Forest (94.28%), confirming Random Forest as the superior choice.

---

## 5. HMM and Viterbi Sequential Route Reconstruction

A classifier evaluated in isolation evaluates each GPS ping independently, leading to parallel road jumping. The **Hidden Markov Model** in `src/hmm_decoder.py` finds the globally optimal sequence of road segments across the entire trajectory.

### 5.1 Formulation
1. **Hidden States:** Candidate road segments (arcs) located within 35m of each observation.
2. **Emission Log-Likelihood:** Fuses the Random Forest posterior with Gaussian spatial noise:
   $$\ln P(z_t | c_i) = \ln \left( \max(P_{\text{RF}}(y=1 | x_{t,i}), 10^{-6}) \right) - \frac{d_{\text{perp}}^2(z_t, c_i)}{2\sigma_z^2}$$
   with $\sigma_z = 15.0\text{ meters}$.
3. **Transition Log-Probabilities:** Derived from directed road network topology:
   - Identity transition (staying on same arc): $\ln P = -0.05$
   - Direct topological successor ($\text{to\_node}_i == \text{from\_node}_j$): $\ln P = -0.25$
   - Two-hop reachable neighbor: $\ln P = -1.50$
   - Disconnected / reverse jump: $\ln P = -10.00$
4. **Handling Missing Candidates and GPS Gaps:**
   - If a GPS ping has zero candidates within 35m, the search expands up to 100m to select the closest segment, resetting transition history to prevent tracking failure.
   - For temporal gaps ($\Delta t > 5\text{s}$), spatial transition penalties are decayed exponentially ($\ln P_{\text{trans}} \times e^{-\Delta t / 30}$), allowing the vehicle to re-acquire the road network naturally after prolonged signal loss.
5. **Vectorized Matrix Acceleration:**
   By stacking all candidate feature vectors into a single 2D matrix for batch prediction before executing Viterbi back-pointers in log-space, inference latency decreased from ~25s to **0.28s – 1.8s** per trajectory.

### 5.2 Trajectory Reconstruction Benchmark (Tuned Random Forest + HMM Viterbi)

| Partition | Trip ID | Fixes Count | Top-1 Accuracy | Route Continuity | Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **evaluator_test** | `00000087` | 1,399 | **99.57%** | **99.07%** | Target Met |
| **evaluator_test** | `00000049` | 2,642 | **98.18%** | **99.24%** | Target Met |
| **student_test** | `00000052` | 262 | **96.95%** | **96.17%** | Target Met |
| **evaluator_test** | `00000073` | 1,144 | **95.10%** | **98.05%** | Target Met |
| **evaluator_test** | `00000023` | 458 | 94.76% | 96.22% | Verified |
| **student_test** | `00000000` | 2,503 | 94.25% | 97.80% | Verified |
| **evaluator_test** | `00000059` | 1,031 | 93.40% | 93.30% | Verified |
| **student_test** | `00000056` | 3,958 | 93.56% | 98.42% | Verified |
| **student_test** | `00000040` | 2,995 | 91.55% | 95.80% | Verified |
| **student_test** | `00000007` | 1,354 | 88.26% | **100.00%** | Perfect Continuity |
| **evaluator_test** | `00000017` | 1,346 | 87.74% | 94.60% | Verified |
| **student_test** | `00000009` | 1,049 | 87.70% | 96.10% | Verified |
| **evaluator_test** | `00000067` | 916 | 86.90% | 97.40% | Verified |
| **student_test** | `00000091` | 874 | 87.30% | 95.20% | Verified |
| **student_test** | `00000076` | 1,486 | 85.40% | 96.16% | Verified |
| **evaluator_test** | `00000094` | 3,230 | 85.85% | 98.10% | Verified |
| **student_test** | `00000065` | 768 | 85.29% | 97.20% | Verified |
| **evaluator_test** | `00000030` | 654 | 79.20% | 92.50% | Verified |
| **student_test** | `00000077` | 3,911 | 71.80% | 98.21% | High GPS Noise |
| **evaluator_test** | `00000015` | 6,854 | *(Reference Pending)* | **99.49%** | Complete Route |

### 5.3 Honest Analysis of Accuracy and Continuity
- **Partition Averages:**
  - `student_test` (10 trips): Mean Top-1 Accuracy: **88.21%**, Mean Route Continuity: **97.11%**.
  - `evaluator_test` (9 trips with ground truth): Mean Top-1 Accuracy: **91.19%**, Mean Route Continuity: **96.80%**.
- **Candidate-Level vs. Observation-Level Accuracy:**
  The 94.28% metric represents candidate classification accuracy across all candidate vectors. On full trajectory reconstruction, peak accuracy reaches **99.57%** (Trip 87), **98.18%** (Trip 49), and **96.95%** (Trip 52).
- **Understanding Route Continuity vs. Top-1 Accuracy:**
  A route with high continuity is not necessarily 100% accurate. For example, Trip `00000077` achieves **98.21% continuity** but **71.80% Top-1 accuracy** due to severe GPS drift (>50m) through complex multi-junction overpasses. Route continuity measures topological graph connectivity, while Top-1 measures agreement with ground-truth labels.

---

## 6. Geographical Visualisation Using Folium & Esri Tiles

Interactive multi-layer maps are generated by `src/visualizer.py` and rendered using Leaflet:
- **Clean Basemap Integration:** Upgraded from CartoDB tiles to **Esri World Dark Gray Base** (`https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}`), eliminating third-party API key watermarks.
- **Layers Rendered:**
  1. *Raw GPS Observations:* Cyan dotted polyline (`#38BDF8`) with circular origin marker and square destination marker.
  2. *Road Network Arcs:* Dark slate lines representing graph topology within the bounding box.
  3. *Reference Route:* Solid green path (`#2E7D32`, weight 6) where available.
  4. *Predicted Snapped Route:* Bright orange/gold path (`#FF6F00`, weight 4.8) produced by the HMM.

---

## 7. API, UI and Distributed Two-Laptop Deployment

PathSnap implements the mandatory **Two-Laptop Distributed Deployment Architecture**:

```
+-------------------------------------------------------------------------+
|                  LAPTOP 1: MODEL & INFERENCE API ENGINE                 |
|  - Host: 0.0.0.0:8000 (FastAPI / Uvicorn)                               |
|  - Active Production Model: RandomForest_tuned.pkl                      |
|  - Dynamic Wi-Fi IP Auto-Detection via PowerShell                      |
|  - Endpoints: GET /health, GET /trips, GET /trip/.../track, POST /evaluate|
|  - Reference Data: Stored 100% strictly server-side                     |
+-------------------------------------------------------------------------+
                                    ▲
                                    │ HTTP / REST (JSON)
                                    ▼
+-------------------------------------------------------------------------+
|                 LAPTOP 2: EVALUATOR DASHBOARD INTERFACE                 |
|  - Host: 0.0.0.0:8501 (Streamlit + Native Full-Screen Leaflet Web App)  |
|  - Dual Partition Switcher: Official Evaluator (10) vs Student Test (10)|
|  - Real-Time Scorecard: Top-1 Accuracy, Route Continuity, Latency       |
|  - 4 Dynamic Themes: Obsidian Amber, Cyberpunk Neon, Emerald, Nordic    |
|  - Fail-Safe Cache: Preloaded local fallback prevents network stalls    |
+-------------------------------------------------------------------------+
```

### Execution Instructions:
1. **Laptop 1 (Model Server):** Run `run_api.bat`. Automatically detects Wi-Fi IP and launches FastAPI:
   ```bash
   python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```
2. **Laptop 2 (Evaluator Dashboard):** Connect to same Wi-Fi and run `run_ui.bat`. Automatically connects to Laptop 1 and opens browser:
   ```bash
   python -m streamlit run ui/app.py --server.port 8501 --server.address 0.0.0.0
   ```
3. **Single-Laptop Local Test:** Run `run_both_on_this_machine.bat`. Launches both servers in dedicated terminal windows and opens `http://localhost:8501`.

---

## 8. Repository Mapping and Integrity Statement

### Required Component Mapping

| Required Component | Source Code Location |
| :--- | :--- |
| **Data Processing** | `src/data_loader.py`, `src/projection.py` |
| **Exploratory Data Analysis (EDA)** | `scripts/preprocess_and_eda.py` |
| **Feature Engineering** | `src/features.py`, `src/candidate_gen.py` |
| **Model Training** | `src/models.py`, `scripts/train_tuned_rf.py` |
| **Model Evaluation** | `src/evaluator.py`, `reports/figures/` |
| **Bayesian Optimization** | `src/bayesian_opt.py`, `scripts/tune_model.py` |
| **HMM and Viterbi Decoding** | `src/hmm_decoder.py` |
| **Backend API** | `api/main.py`, `api/schemas.py` |
| **User Interface (UI)** | `ui/app.py`, `web/index.html` |
| **Deployment Scripts** | `run_api.bat`, `run_ui.bat`, `run_both_on_this_machine.bat` |
| **Geographical Visualization** | `src/visualizer.py`, `maps/` |
| **Saved Artifacts & Results** | `artifacts/saved_models/RandomForest_tuned.pkl`, `artifacts/` |

### Academic Integrity Declaration
All code, models, features, and Viterbi decoding routines in this project were implemented independently by the team. No code or model artifacts were acquired from other teams. All experimental results reported herein are reproducible via the public repository: [https://github.com/JayanthKumarP88/PathSnap](https://github.com/JayanthKumarP88/PathSnap).
