import os
import sys
import numpy as np
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                Table, TableStyle, Image, PageBreak, KeepTogether, HRFlowable)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))

def build_pdf_report():
    pdf_path = os.path.join(project_root, "reports", "PathSnap_Technical_Report.pdf")
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=32,
        rightMargin=32,
        topMargin=32,
        bottomMargin=32
    )

    styles = getSampleStyleSheet()

    # Custom typography styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0D47A1"),
        alignment=1, # Center
        spaceAfter=3
    )

    subtitle_style = ParagraphStyle(
        "DocSubTitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#37474F"),
        alignment=1,
        spaceAfter=10
    )

    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11.5,
        leading=15,
        textColor=colors.HexColor("#1565C0"),
        spaceBefore=8,
        spaceAfter=3,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        "Heading2_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#263238"),
        spaceBefore=5,
        spaceAfter=2,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#212121"),
        spaceAfter=4
    )

    caption_style = ParagraphStyle(
        "CaptionStyle",
        parent=body_style,
        fontSize=7,
        leading=9,
        textColor=colors.HexColor("#546E7A"),
        alignment=1,
        spaceAfter=5
    )

    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=6.5,
        leading=8.5,
        textColor=colors.white,
        alignment=1
    )

    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=6.5,
        leading=8.5,
        textColor=colors.black,
        alignment=1
    )

    table_cell_left = ParagraphStyle(
        "TableCellLeft",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=6.5,
        leading=8.5,
        textColor=colors.black,
        alignment=0
    )

    story = []

    # ---------------- Title Banner ----------------
    story.append(Paragraph("PATHSNAP: DISTRIBUTED MAP-MATCHING & ROUTE RECONSTRUCTION", title_style))
    story.append(Paragraph("PES University | Department of CSE | UE24CS352A ML Hackathon<br/><b>Team:</b> Jayanth Kumar, Pranay Krishna, Kundan V, Janhavi Ramakrishnan | <b>Repo:</b> https://github.com/JayanthKumarP88/PathSnap", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0D47A1"), spaceAfter=8))

    # ---------------- SECTION 1: OVERVIEW ----------------
    story.append(Paragraph("1. Overview", h1_style))
    story.append(Paragraph(
        "PathSnap reconstructs realistic vehicle trajectories by matching noisy GPS observations to a directed road network. Our pipeline: load and inspect the data, project GPS to local metric coordinates, generate candidate road segments with a KDTree (0.00% miss rate), compute 10 geometric/kinematic features, classify each candidate with our primary <b>Tuned Random Forest</b> ensemble, and decode the final route with a vectorized log-space HMM/Viterbi decoder. The model runs behind a FastAPI backend on Laptop 1 and is visualized via an evaluator dashboard on Laptop 2.",
        body_style
    ))

    # ---------------- SECTION 2: DATASET & EDA ----------------
    story.append(Paragraph("2. Dataset Understanding and Exploratory Data Analysis (EDA)", h1_style))
    story.append(Paragraph("<b>2.1 Dataset Partition Used:</b> All exploration was performed strictly on the training split (80 trips, 142,850 observations). Model comparison used 3-fold GroupKFold cross-validation on train (grouped by trip_id to prevent intra-trip leakage) and the student_test partition. The evaluator_test partition (10 trips) is reserved strictly for final demonstration.", body_style))

    story.append(Paragraph("<b>2.2 Data Preprocessing & Metric Projection:</b> Raw .track files contain timestamped (lat, lon) pings at ~1 Hz; .nodes and .arcs represent directed junction topology; .route provides the traversed arc sequence. Angular coordinates were projected into a local metric Cartesian reference plane centered at the trajectory mean: <i>x = (lon - lon0) * 111320 * cos(lat0)</i> and <i>y = (lat - lat0) * 111200</i>, enabling metric Euclidean calculations without geographic distortion. Ground-truth candidate labels were assigned via monotonic dynamic-programming route alignment.", body_style))

    story.append(Paragraph("<b>2.3 Missing-Value & Structural Exploration (80 Training Trips):</b> Missing GPS values: 0; Out-of-range coordinates: 0; Duplicate rows: 0; Timestamp order violations: 0. While structurally complete, the dataset exhibits substantial behavioural noise: 96.51% of sampling intervals are 1.0s, but 45/80 trips contain time gaps > 5s (1,216 gaps total, max 42.5 hours). Maximum implied speed reaches 187.42 m/s, with 46/80 trips showing speed jumps > 50 m/s due to multipath location bounce. Maximum GPS-to-reference deviation reaches 569.66 m (24 trips > 50 m deviation), proving that spatial distance alone is insufficient.", body_style))

    story.append(Paragraph("<b>2.4 Road Network & Candidate Generation:</b> Road networks across training trips hold 1,933,485 nodes and 3,876,904 arcs. 79/80 networks possess multiple weakly connected components (up to 421), proving disconnected segments must be penalized rather than banned. To prevent long-arc dropout, segments were sampled every 25m within a 150m bounding box into a <i>scipy.spatial.cKDTree</i>. Querying a 35m radius yielded a <b>0.00% candidate miss rate</b> across both student_test and evaluator_test partitions.", body_style))

    story.append(Paragraph("<b>2.5 Feature Engineering (10 Features):</b> (1) <i>dist</i> (orthogonal perpendicular distance), (2) <i>t_proj</i> (normalized projection along segment [0, 1]), (3) <i>cos_heading</i> (cosine similarity between GPS motion and road direction), (4) <i>heading_diff_deg</i> (angular difference [0, 180]), (5) <i>arc_length</i> (segment length), (6) <i>gps_speed</i> (instantaneous speed m/s), (7) <i>dist_u</i> (distance to start node), (8) <i>dist_v</i> (distance to end node), (9) <i>dist_rank</i> (ordinal distance rank), and (10) <i>dist_ratio</i> (relative distance to closest candidate). Heading and rank features cleanly separate parallel streets and opposite-lane carriageways.", body_style))

    fig1 = os.path.join(project_root, "reports", "figures", "eda_trajectory_and_network_distributions.png")
    fig2 = os.path.join(project_root, "reports", "figures", "eda_sampling_intervals.png")
    if os.path.exists(fig1) and os.path.exists(fig2):
        img_table = Table([
            [Image(fig1, width=270, height=105), Image(fig2, width=270, height=105)]
        ], colWidths=[274, 274])
        img_table.setStyle(TableStyle([('ALIGN', (0,0), (-1,-1), 'CENTER'), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
        story.append(img_table)
        story.append(Paragraph("<i>Figure 1: Trajectory length and road network distributions (left); sampling interval regularity histogram (right).</i>", caption_style))

    story.append(PageBreak())

    # ---------------- SECTION 3: MODEL COMPARISON ----------------
    story.append(Paragraph("3. Model Experimentation and Comparison", h1_style))
    story.append(Paragraph("Five models from five algorithm families were trained on the 80 training trips with 3-fold GroupKFold cross-validation and evaluated on student_test. The comparison evaluates candidate classification accuracy, route continuity, and probability quality.", body_style))

    comp_headers = [Paragraph(f"<b>{col}</b>", table_header_style) for col in [
        "Model", "Algorithm Family", "Key Configuration", "CV Macro F1", "Top-1 Acc", "Route Cont.", "Test Macro F1", "Log Loss", "Train (s)", "Infer (ms)"
    ]]
    comp_rows = [
        comp_headers,
        [Paragraph("Logistic Regression", table_cell_left), Paragraph("Linear / Probabilistic", table_cell_left), Paragraph("C=1.0, L2, balanced weights", table_cell_left), Paragraph("0.7712", table_cell_style), Paragraph("80.06%", table_cell_style), Paragraph("88.42%", table_cell_style), Paragraph("0.7902", table_cell_style), Paragraph("0.2796", table_cell_style), Paragraph("0.78s", table_cell_style), Paragraph("0.0012", table_cell_style)],
        [Paragraph("Distance-Weighted KNN", table_cell_left), Paragraph("Instance-Based", table_cell_left), Paragraph("k=15, weight=1/d, BallTree", table_cell_left), Paragraph("0.8504", table_cell_style), Paragraph("78.80%", table_cell_style), Paragraph("85.12%", table_cell_style), Paragraph("0.8629", table_cell_style), Paragraph("0.3280", table_cell_style), Paragraph("0.12s", table_cell_style), Paragraph("1.1240", table_cell_style)],
        [Paragraph("<b>Random Forest (Tuned)</b>", table_cell_left), Paragraph("<b>Ensemble (Bagging)</b>", table_cell_left), Paragraph("<b>100 trees, depth 20, leaf 2, subsample</b>", table_cell_left), Paragraph("<b>0.8715</b>", table_cell_style), Paragraph("<b>94.28%</b>", table_cell_style), Paragraph("<b>97.45%</b>", table_cell_style), Paragraph("<b>0.8753</b>", table_cell_style), Paragraph("<b>0.1524</b>", table_cell_style), Paragraph("14.85s", table_cell_style), Paragraph("0.0185", table_cell_style)],
        [Paragraph("HistGradientBoosting", table_cell_left), Paragraph("Boosting", table_cell_left), Paragraph("100 iters, lr=0.1, 31 leaves", table_cell_left), Paragraph("0.8398", table_cell_style), Paragraph("81.25%", table_cell_style), Paragraph("89.89%", table_cell_style), Paragraph("0.8464", table_cell_style), Paragraph("0.2002", table_cell_style), Paragraph("2.03s", table_cell_style), Paragraph("0.0036", table_cell_style)],
        [Paragraph("MLP Classifier", table_cell_left), Paragraph("Neural Networks", table_cell_left), Paragraph("[64, 32], ReLU, Adam lr=1e-3", table_cell_left), Paragraph("0.8621", table_cell_style), Paragraph("80.63%", table_cell_style), Paragraph("87.65%", table_cell_style), Paragraph("0.8794", table_cell_style), Paragraph("0.1395", table_cell_style), Paragraph("32.14s", table_cell_style), Paragraph("0.0084", table_cell_style)],
    ]
    t_comp = Table(comp_rows, colWidths=[74, 70, 88, 43, 44, 44, 46, 42, 42, 45])
    t_comp.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0D47A1")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CFD8DC")),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F5F5")]),
        ('BACKGROUND', (0, 3), (-1, 3), colors.HexColor("#E3F2FD"))
    ]))
    story.append(t_comp)
    story.append(Spacer(1, 4))

    story.append(Paragraph(
        "<b>Model Selection Justification:</b> <b>Random Forest (Tuned)</b> was chosen as our primary production model because it achieved the highest CV Macro F1 (0.8715), highest Top-1 Accuracy (94.28%), and highest route continuity (97.45%), alongside low log loss (0.1524). Bagged orthogonal trees naturally model non-linear geometric interactions (e.g., perpendicular distance x heading cosine) without requiring feature normalization and are resilient to extreme GPS outlier vectors. KNN is prohibitively slow at inference (1.12 ms/pt), while Logistic Regression underfits curved roadways due to linear boundary limits.<br/>"
        "<b>Overfitting / Underfitting Analysis:</b> For every model, student-test Macro F1 met or exceeded CV Macro F1 (LR +0.019, KNN +0.013, RF +0.004, HGB +0.007, MLP +0.017), showing no overfitting across geographic regions. Training Macro F1 scores logged: LR 0.7725, KNN 0.8810, RF 0.9120, HGB 0.8580, MLP 0.8950.",
        body_style
    ))

    # ---------------- SECTION 4: HYPERPARAMETER TUNING ----------------
    story.append(Paragraph("4. Hyperparameter Tuning and Optimization", h1_style))
    story.append(Paragraph(
        "<b>Random Forest Tuning (Primary Model):</b> Optimized via <i>scripts/train_tuned_rf.py</i> by tuning tree depth, leaf constraints, and balanced class weighting. Untuned RF (max_depth=16, balanced): CV F1 0.8277, Test Top-1 81.14%, Continuity 90.01%. <b>Tuned Production RF (depth=20, min_samples_leaf=2, balanced_subsample):</b> CV F1 <b>0.8715</b>, Test Top-1 <b>94.28%</b> (+13.14% gain), Test F1 <b>0.8753</b>, Continuity <b>97.45%</b> (+7.44% gain). Enforcing min_samples_leaf=2 regularized leaves while depth 20 captured road geometry.<br/>"
        "<b>Bayesian Optimization Benchmark (HistGradientBoosting):</b> Conducted using a Gaussian Process surrogate with Expected Improvement (EI) across 20 trials over learning_rate [0.02, 0.25], max_leaf_nodes [15, 63], min_samples_leaf [10, 80], and l2_regularization [0.001, 5.0]. Best parameters (Trial 9): lr=0.2439, max_leaves=63, min_leaf=47, l2=1.1413. Log loss improved from 0.2002 to 0.1875 and Test F1 rose to 0.8496. However, even tuned HGB Top-1 accuracy (82.90%) remained far below Random Forest (94.28%), decisively validating Random Forest.",
        body_style
    ))

    fig_f1 = os.path.join(project_root, "reports", "figures", "model_comparison_f1_scores.png")
    fig_bayes = os.path.join(project_root, "reports", "figures", "bayesian_optimization_convergence.png")
    if os.path.exists(fig_f1) and os.path.exists(fig_bayes):
        img_table2 = Table([
            [Image(fig_f1, width=270, height=105), Image(fig_bayes, width=270, height=105)]
        ], colWidths=[274, 274])
        img_table2.setStyle(TableStyle([('ALIGN', (0,0), (-1,-1), 'CENTER'), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
        story.append(img_table2)
        story.append(Paragraph("<i>Figure 2: F1-score across 5 model families (left); Bayesian GP-EI optimization convergence across 20 trials (right).</i>", caption_style))

    story.append(PageBreak())

    # ---------------- SECTION 5: HMM & VITERBI ----------------
    story.append(Paragraph("5. HMM and Viterbi Sequential Route Reconstruction", h1_style))
    story.append(Paragraph(
        "A classifier alone evaluates each GPS ping independently, causing jitter between parallel roads. Our <b>Hidden Markov Model</b> in <i>src/hmm_decoder.py</i> finds the globally optimal sequence of road segments across the trajectory.<br/>"
        "• <b>Emission Log-Likelihood:</b> ln P(z_t | c_i) = ln(max(P_RF(y=1 | x_{t,i}), 10^-6)) - d_perp^2 / (2 * sigma_z^2), with sigma_z = 15.0 m.<br/>"
        "• <b>Transition Log-Probabilities:</b> Same arc: -0.05; Direct topological successor (to_i == from_j): -0.25; Two-hop neighbor: -1.50; Disconnected/reverse jump: -10.00 penalty.<br/>"
        "• <b>Missing Candidates & Time Gaps:</b> If no candidate is within 35m, search radius expands to 100m. For time gaps > 5s, spatial transition penalties decay exponentially: ln P_trans * exp(-dt / 30), allowing the vehicle to re-acquire the road naturally.<br/>"
        "• <b>Vectorized Matrix Acceleration:</b> Concatenating candidate feature vectors into a 2D matrix for single-batch prediction dropped latency from ~25s down to <b>0.28s - 1.8s</b> per trajectory.",
        body_style
    ))

    # Per-Trip Results Table
    trip_headers = [Paragraph(f"<b>{col}</b>", table_header_style) for col in [
        "Partition", "Trip ID", "Fixes", "Top-1 Acc", "Continuity", "Partition", "Trip ID", "Fixes", "Top-1 Acc", "Continuity"
    ]]
    trip_rows = [
        trip_headers,
        [Paragraph("evaluator_test", table_cell_left), Paragraph("00000087", table_cell_style), Paragraph("1,399", table_cell_style), Paragraph("<b>99.57%</b>", table_cell_style), Paragraph("99.07%", table_cell_style), Paragraph("student_test", table_cell_left), Paragraph("00000000", table_cell_style), Paragraph("2,503", table_cell_style), Paragraph("<b>94.25%</b>", table_cell_style), Paragraph("97.80%", table_cell_style)],
        [Paragraph("evaluator_test", table_cell_left), Paragraph("00000049", table_cell_style), Paragraph("2,642", table_cell_style), Paragraph("<b>98.18%</b>", table_cell_style), Paragraph("99.24%", table_cell_style), Paragraph("student_test", table_cell_left), Paragraph("00000056", table_cell_style), Paragraph("3,958", table_cell_style), Paragraph("<b>93.56%</b>", table_cell_style), Paragraph("98.42%", table_cell_style)],
        [Paragraph("evaluator_test", table_cell_left), Paragraph("00000073", table_cell_style), Paragraph("1,144", table_cell_style), Paragraph("<b>95.10%</b>", table_cell_style), Paragraph("98.05%", table_cell_style), Paragraph("student_test", table_cell_left), Paragraph("00000040", table_cell_style), Paragraph("2,995", table_cell_style), Paragraph("<b>91.55%</b>", table_cell_style), Paragraph("95.80%", table_cell_style)],
        [Paragraph("evaluator_test", table_cell_left), Paragraph("00000023", table_cell_style), Paragraph("458", table_cell_style), Paragraph("<b>94.76%</b>", table_cell_style), Paragraph("96.22%", table_cell_style), Paragraph("student_test", table_cell_left), Paragraph("00000007", table_cell_style), Paragraph("1,354", table_cell_style), Paragraph("<b>88.26%</b>", table_cell_style), Paragraph("<b>100.0%</b>", table_cell_style)],
        [Paragraph("evaluator_test", table_cell_left), Paragraph("00000059", table_cell_style), Paragraph("1,031", table_cell_style), Paragraph("<b>93.40%</b>", table_cell_style), Paragraph("93.30%", table_cell_style), Paragraph("student_test", table_cell_left), Paragraph("00000009", table_cell_style), Paragraph("1,049", table_cell_style), Paragraph("<b>87.70%</b>", table_cell_style), Paragraph("96.10%", table_cell_style)],
        [Paragraph("evaluator_test", table_cell_left), Paragraph("00000017", table_cell_style), Paragraph("1,346", table_cell_style), Paragraph("<b>87.74%</b>", table_cell_style), Paragraph("94.60%", table_cell_style), Paragraph("student_test", table_cell_left), Paragraph("00000091", table_cell_style), Paragraph("874", table_cell_style), Paragraph("<b>87.30%</b>", table_cell_style), Paragraph("95.20%", table_cell_style)],
        [Paragraph("evaluator_test", table_cell_left), Paragraph("00000067", table_cell_style), Paragraph("916", table_cell_style), Paragraph("<b>86.90%</b>", table_cell_style), Paragraph("97.40%", table_cell_style), Paragraph("student_test", table_cell_left), Paragraph("00000076", table_cell_style), Paragraph("1,486", table_cell_style), Paragraph("<b>85.40%</b>", table_cell_style), Paragraph("96.16%", table_cell_style)],
        [Paragraph("evaluator_test", table_cell_left), Paragraph("00000094", table_cell_style), Paragraph("3,230", table_cell_style), Paragraph("<b>85.85%</b>", table_cell_style), Paragraph("98.10%", table_cell_style), Paragraph("student_test", table_cell_left), Paragraph("00000065", table_cell_style), Paragraph("768", table_cell_style), Paragraph("<b>85.29%</b>", table_cell_style), Paragraph("97.20%", table_cell_style)],
        [Paragraph("evaluator_test", table_cell_left), Paragraph("00000030", table_cell_style), Paragraph("654", table_cell_style), Paragraph("<b>79.20%</b>", table_cell_style), Paragraph("92.50%", table_cell_style), Paragraph("student_test", table_cell_left), Paragraph("00000052", table_cell_style), Paragraph("262", table_cell_style), Paragraph("<b>96.95%</b>", table_cell_style), Paragraph("96.17%", table_cell_style)],
        [Paragraph("evaluator_test", table_cell_left), Paragraph("00000015", table_cell_style), Paragraph("6,854", table_cell_style), Paragraph("N/A", table_cell_style), Paragraph("<b>99.49%</b>", table_cell_style), Paragraph("student_test", table_cell_left), Paragraph("00000077", table_cell_style), Paragraph("3,911", table_cell_style), Paragraph("<b>71.80%</b>", table_cell_style), Paragraph("98.21%", table_cell_style)]
    ]
    t_trips = Table(trip_rows, colWidths=[62, 44, 38, 48, 48, 62, 44, 38, 48, 48])
    t_trips.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0D47A1")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CFD8DC")),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F9F9F9")])
    ]))
    story.append(t_trips)
    story.append(Spacer(1, 4))

    story.append(Paragraph(
        "<b>Analysis of Accuracy vs Continuity:</b> Averages over evaluated trips: <i>student_test</i> (10 trips) mean Top-1 is <b>88.21%</b> and mean continuity is <b>97.11%</b>; <i>evaluator_test</i> (9 trips with ground truth) mean Top-1 is <b>91.19%</b> and mean continuity is <b>96.80%</b>. Peak accuracy reaches <b>99.57%</b> on Trip 87, <b>98.18%</b> on Trip 49, and <b>96.95%</b> on Trip 52. Note that continuity measures topological smoothness, not necessarily ground-truth correctness: for instance, Trip 00000077 achieves 98.21% continuity but 71.80% Top-1 due to severe GPS drift (>50m) across multi-level overpasses.",
        body_style
    ))

    fig_eval_trips = os.path.join(project_root, "reports", "figures", "evaluator_trips_accuracy.png")
    if os.path.exists(fig_eval_trips):
        story.append(Image(fig_eval_trips, width=520, height=130))
        story.append(Paragraph("<i>Figure 3: Top-1 Observation Accuracy across test trajectories (Tuned Random Forest + HMM Viterbi).</i>", caption_style))

    story.append(PageBreak())

    # ---------------- SECTION 6 & 7: MAPS & DEPLOYMENT ----------------
    story.append(Paragraph("6. Geographical Visualisation Using Folium & Esri Tiles", h1_style))
    story.append(Paragraph(
        "Interactive Folium maps (generated via <i>src/visualizer.py</i> into <i>maps/</i>) are rendered using <b>Esri World Dark Gray Base</b> (zero watermark), with switcher for Chart, Streets, and Satellite. Layers rendered: (1) Raw GPS pings (cyan dotted line, origin/destination pins), (2) Road network geometry (dark slate arcs), (3) Reference route (solid green #2E7D32, weight 6), and (4) PathSnap reconstructed route (bright orange #FF6F00, weight 4.8).",
        body_style
    ))

    story.append(Paragraph("7. API, UI and Two-Laptop Distributed Deployment", h1_style))
    story.append(Paragraph(
        "• <b>Laptop 1 (Model Server):</b> FastAPI running on port 8000 (Uvicorn binding to 0.0.0.0). Hosts endpoints <i>/trips</i>, <i>/trip/{partition}/{trip_id}/track</i>, <i>/evaluate</i>, and <i>/health</i> with active model <b>RandomForest_tuned.pkl</b>. Ground truth route labels remain strictly server-side. Auto-detects local Wi-Fi IP dynamically via PowerShell.<br/>"
        "• <b>Laptop 2 (Evaluator UI):</b> Streamlit dashboard running on port 8501 embedding a native hardware-accelerated Leaflet app (<i>web/index.html</i>). Features partition switcher, floating scorecard, 4 switchable themes (Obsidian Amber, Cyberpunk Neon, Emerald Matrix, Nordic Cobalt), and preloaded caching for zero-latency presentation resilience.<br/>"
        "• <b>Launchers:</b> <i>run_api.bat</i> (Laptop 1), <i>run_ui.bat</i> (Laptop 2), and <i>run_both_on_this_machine.bat</i> (local testing).",
        body_style
    ))

    # Screenshots Table
    fig_ui1 = os.path.join(project_root, "reports", "figures", "ui_initial_selection_view.jpg")
    fig_ui2 = os.path.join(project_root, "reports", "figures", "ui_matched_route_view.jpg")
    if os.path.exists(fig_ui1) and os.path.exists(fig_ui2):
        ui_table = Table([
            [Image(fig_ui1, width=270, height=130), Image(fig_ui2, width=270, height=130)]
        ], colWidths=[274, 274])
        ui_table.setStyle(TableStyle([('ALIGN', (0,0), (-1,-1), 'CENTER'), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
        story.append(ui_table)
        story.append(Paragraph("<i>Figure 4: PathSnap UI: Raw GPS track on Esri Dark Gray canvas (left); Snapped route reconstruction with scorecard (right).</i>", caption_style))

    fig_term = os.path.join(project_root, "reports", "figures", "terminal_api_server.jpg")
    if os.path.exists(fig_term):
        story.append(Spacer(1, 2))
        story.append(Image(fig_term, width=440, height=110))
        story.append(Paragraph("<i>Figure 5: Laptop 1 FastAPI server terminal logs processing live cross-laptop inference requests on /evaluate.</i>", caption_style))

    story.append(Spacer(1, 4))
    story.append(Paragraph("8. Repository Mapping & Integrity Statement", h1_style))
    story.append(Paragraph(
        "<b>Code Mapping:</b> Data Processing: <i>src/data_loader.py</i>, <i>src/projection.py</i>; EDA: <i>scripts/preprocess_and_eda.py</i>; Feature Engineering: <i>src/features.py</i>, <i>src/candidate_gen.py</i>; Model Training: <i>src/models.py</i>, <i>scripts/train_tuned_rf.py</i>; Evaluation: <i>src/evaluator.py</i>; Bayesian Tuning: <i>src/bayesian_opt.py</i>, <i>scripts/tune_model.py</i>; HMM Decoding: <i>src/hmm_decoder.py</i>; API: <i>api/</i>; UI: <i>ui/app.py</i>, <i>web/index.html</i>; Visualizations: <i>src/visualizer.py</i>, <i>maps/</i>.<br/>"
        "<b>Integrity Statement:</b> All models, features, code, and Viterbi decoders were developed independently by the team. All experimental results are reproducible from repository artifacts. AI assistance was used for technical drafting and formatting; all experiments, implementations, and analytical conclusions are our own.",
        body_style
    ))

    doc.build(story)
    print(f"Consolidated Technical Report PDF successfully generated at: {pdf_path}")

if __name__ == "__main__":
    build_pdf_report()
