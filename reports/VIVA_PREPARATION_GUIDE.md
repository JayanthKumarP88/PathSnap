# PathSnap: Presentation & Viva Defense Guide
## PES University | UE24CS352A Machine Learning Hackathon

This guide prepares the entire team for the **viva, technical questioning, and live evaluation demonstration**. Every team member should be familiar with these core concepts and questions.

---

## 👥 Suggested Team Role Distribution for Viva

| Team Member | Primary Viva Focus & Demonstration Ownership |
|---|---|
| **Pranay Krishna** | **Dataset & Preprocessing**: Spatial projection (angular to metric), sampling regularity, missing-value analysis, and candidate generation. |
| **Janhavi Ramakrishnan** | **Feature Engineering & Candidate Indexing**: Segment-sampled KDTree, 10 kinematic/geometric features, and Folium map visualizer. |
| **Kundan** | **5-Family Model Benchmarking & Bayesian Tuning**: Model trade-offs, GroupKFold CV, Gaussian Process surrogate model with Expected Improvement. |
| **Jayanth Kumar** | **HMM/Viterbi Reconstruction & Two-Laptop Architecture**: Emission/transition modeling, Viterbi recurrence, FastAPI backend, and Evaluator UI demo. |

---

## 🎯 1. Dataset Understanding & Preprocessing

### Q1: Why can't we use standard Euclidean distance on (Longitude, Latitude) coordinates?
> **Answer**: Longitude and latitude are angular coordinates on an oblate spheroid. At the dataset's latitude ($\approx 49.8^\circ\text{N}$ in Germany):
> - $1^\circ$ of Latitude $\approx 111.2\text{ km}$
> - $1^\circ$ of Longitude $\approx 111.32 \cdot \cos(49.8^\circ) \approx 71.8\text{ km}$
>
> A degree of longitude is **$35.4\%$ shorter** than a degree of latitude. Computing Euclidean distance directly in degrees treats longitude and latitude symmetrically, severely warping spatial geometry and distance rankings.  
> **Our Solution**: We implemented a local metric Cartesian projection (`LocalCoordinateProjector`) centered at the trajectory mean:
> $$\Delta x = (\text{lon} - \text{lon}_0) \cdot 111320 \cdot \cos(\text{lat}_0)$$
> $$\Delta y = (\text{lat} - \text{lat}_0) \cdot 111200$$
> This yields flat $(x, y)$ coordinates in meters accurate to millimeters across municipal scales.

### Q2: How did you solve the "Long Segment Dropout" problem in candidate generation?
> **Answer**: Standard spatial indexing (e.g., querying only road arc midpoints or nodes) fails when road segments are long ($100\text{–}300\text{ m}$ on straight roads or highways). A vehicle driving in the center of a $200\text{ m}$ arc is $100\text{ m}$ away from both endpoints, causing naive radius queries ($35\text{ m}$) to miss the true road segment.  
> **Our Solution**: We implemented **Segment-Sampled KDTree Indexing** (`CandidateGenerator`). We discretely placed sample points every $25\text{ m}$ along all candidate arcs. Consequently, no point on any road segment is ever more than $12.5\text{ m}$ from a sampled node. This achieved a **$0.00\%$ Candidate Miss Rate** at a $35\text{ m}$ query radius with an average of only $\sim 7$ candidates per observation.

---

## 🔬 2. Feature Engineering

### Q3: What features did you extract and why?
We engineered **10 discriminatory kinematic and geometric features**:
1. **$d_{\text{perp}}$ (Perpendicular Distance)**: Shortest distance from GPS ping to the road line segment.
2. **$t_{\text{proj}}$ (Projection Fraction)**: Position along the arc ($0.0 \le t \le 1.0$), identifying whether the car is near the start, middle, or end.
3. **$\cos(\theta)$ (Heading Cosine Similarity)**: Dot product between the GPS velocity vector ($\vec{v}_{\text{gps}}$) and road segment orientation ($\vec{v}_{\text{seg}}$). Crucial for disambiguating parallel two-way streets or divided carriageways.
4. **$|\Delta \theta|$ (Angular Difference in Degrees)**: Non-linear angular deviation ($0^\circ\text{–}180^\circ$).
5. **$L_{\text{seg}}$ (Segment Length)**: Segment length in meters.
6. **$v_{\text{gps}}$ (Instantaneous Speed)**: $\|z_t - z_{t-1}\| / \Delta t$ in m/s.
7. **$d_u, d_v$ (Distances to Endpoints)**: Distances to start node $u$ and end node $v$.
8. **$\text{Rank}_{\text{dist}}$ (Candidate Distance Rank)**: Sort rank of candidate ($0, 1, 2, \dots$) at timestamp $t$.
9. **$R_{\text{dist}}$ (Distance Ratio)**: $d_{\text{perp}} / (d_{\text{min}} + 10^{-3})$. Relativizes competition among candidates.

---

## 🤖 3. Model Comparison & Selection

### Q4: Which 5 model families did you evaluate, and how did they compare?
We benchmarked 5 distinct algorithm families using **5-Fold GroupKFold Cross-Validation** (grouped by `trip_id` across 80 training trips to avoid spatial/temporal leakage):

| Model | Family | CV Macro F1 | Observation Top-1 Acc | Route Continuity | Training Time | Latency |
|---|---|---|---|---|---|---|
| **Logistic Regression** | Linear & Probabilistic | 0.7567 | 80.06% | 88.93% | **0.78s** | **0.0002 ms** |
| **Distance-Weighted KNN** | Instance-Based ($k=15$) | 0.8046 | 78.80% | 88.60% | 0.09s | 1.1215 ms |
| **Random Forest** | Ensemble (100 trees) | **0.8277** | 81.14% | **90.01%** | 15.06s | 0.0028 ms |
| **HistGradientBoosting** | Boosting (100 iters) | 0.8119 | **81.25%** | 89.89% | 2.03s | 0.0036 ms |
| **MLP Classifier** | Neural Networks ([64, 32]) | 0.8195 | 80.63% | 89.87% | 30.07s | 0.0019 ms |

### Q5: Why did you select HistGradientBoosting over Random Forest and MLP?
> **Answer**: 
> 1. **Highest Raw Top-1 Accuracy**: Achieved $81.25\%$ point-level accuracy before sequential decoding.
> 2. **Computational Efficiency**: Trained in **$2.03\text{ seconds}$** (vs $15.06\text{s}$ for RF and $30.07\text{s}$ for MLP).
> 3. **Production Latency**: Inference latency of **$0.0036\text{ ms/sample}$**, enabling real-time streaming inference on Laptop 1.
> 4. **Histogram Discretization**: Automatically bins continuous features into 256 integer bins, making it resilient to outliers in GPS speed and distance jitter.

---

## 📈 4. Bayesian Hyperparameter Optimization

### Q6: How did you implement Bayesian Optimization?
> **Answer**: We built a Gaussian Process optimizer (`GaussianProcessBayesianOptimizer`) with a **Matérn-5/2 covariance kernel** and an **Expected Improvement (EI)** acquisition function:
> $$\text{EI}(\mathbf{\theta}) = (\mu(\mathbf{\theta}) - f^* - \xi)\Phi(Z) + \sigma(\mathbf{\theta})\phi(Z), \quad Z = \frac{\mu(\mathbf{\theta}) - f^* - \xi}{\sigma(\mathbf{\theta})}$$
> - Evaluated across **20 trials** (5 random exploration warm-up trials followed by 15 GP-EI exploitation/exploration trials).
> - Evaluated using **3-fold GroupKFold CV** on the 80 training trips.
> - **Search Space**: `learning_rate` [0.02, 0.25] (log-uniform), `max_leaf_nodes` [15, 63] (integer), `min_samples_leaf` [10, 80] (integer), `l2_regularization` [0.001, 5.0] (log-uniform).
> - **Outcome**: Log loss decreased from $0.2002 \to \mathbf{0.1875}$, and Student-Test Macro F1 improved to $\mathbf{0.8496}$.

---

## 🔗 5. HMM & Viterbi Sequential Route Reconstruction

### Q7: Why is a classifier alone insufficient for map matching?
> **Answer**: Pointwise classifiers evaluate each GPS ping in isolation. Due to GPS noise, consecutive pings can alternate between two parallel streets or jump backwards on a one-way road. An HMM models temporal transitions and road-network topology, enforcing continuous physical movement.

### Q8: What are your Hidden States, Emissions, and Transitions?
- **Hidden States**: The set of candidate road segments $C_t = \{c_{t, 1}, \dots, c_{t, M_t}\}$ within the search radius at time $t$.
- **Emission Probabilities**: Combines the calibrated classifier probability with a spatial Gaussian decay:
  $$\ln P(z_t \mid c_i) = \ln P_{\text{model}}(y = 1) - \frac{d_{\text{perp}}^2}{2\sigma_z^2}$$
  where $\sigma_z = 8.0\text{ m}$.
- **Transition Probabilities**:
  - *Same Arc* ($c_{t-1} = c_t$): $\ln A = -0.05$ (vehicle staying on current road).
  - *Direct Topological Successor* ($v_{t-1} = u_t$): $\ln A = -0.25$ (natural progression through intersection).
  - *2-Hop Graph Neighbor*: $\ln A = -1.5$ (short connecting arc).
  - *Disconnected / Reverse Jump*: $\ln A = -10.0$ (heavy penalty).
- **Viterbi Recurrence**: Run in log-space to prevent underflow:
  $$\delta_t(j) = \max_i [\delta_{t-1}(i) + \ln A(i, j)] + \ln b_t(j)$$
  followed by dynamic programming traceback.

### Q9: Did HMM decoding improve the final route?
> **Answer: YES, substantially across all 10 student-test trips!**
> - **Top-1 Observation Accuracy**: Rose from $82.30\% \to \mathbf{88.96\%}$ ($\mathbf{+6.66\%}$ gain), reaching **$96.95\%$** on Trip 00000052 and **$94.85\%$** on Trip 00000000.
> - **Route Continuity**: Rose from $89.58\% \to \mathbf{98.20\%}$ ($\mathbf{+8.63\%}$ gain), reaching **$100.00\%$ complete topological continuity** on Trip 00000007.

---

## 💻 6. Two-Laptop Architecture Demonstration

### Demonstration Steps During Evaluation
1. **Laptop 1 (API Server)**:
   - Run `run_api.bat`.
   - Point out that FastAPI binds to `0.0.0.0:8000` on your LAN IP (e.g. `192.168.x.x`).
   - Ground-truth routes remain strictly on Laptop 1 to eliminate data leakage.
2. **Laptop 2 (Evaluator UI)**:
   - Run `run_ui.bat`.
   - Enter Laptop 1's IP address.
   - Show the 10 evaluation trip dropdown options.
   - Click **"Run Reconstruction & Evaluate"**.
   - Show the live KPI cards: **Top-1 Accuracy, Route Continuity (98%+), Candidate Miss Rate, and Latency**.
   - Show the interactive **Folium map** showing the original GPS points (blue), road network (gray), reference route (green), and reconstructed route (orange dashed line).
3. **Show Server Logs**:
   - Switch back to Laptop 1 CMD window to show the evaluator the real-time inference log containing the incoming client IP, processing stages, and elapsed time.
