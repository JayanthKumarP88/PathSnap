import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             accuracy_score, log_loss, confusion_matrix)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models import create_model_suite, MODEL_FAMILIES
from src.data_loader import TripDataLoader
from src.hmm_decoder import HMMRouteDecoder

def run_model_benchmarking():
    print("=" * 60)
    print("PATHSNAP: 5-FAMILY MODEL EXPERIMENTATION & BENCHMARKING")
    print("=" * 60)

    # 1. Load Preprocessed Data
    train_cache = np.load(os.path.join(project_root, "artifacts", "train_features.npz"), allow_pickle=True)
    X_train = train_cache["X"]
    y_train = train_cache["y"]
    groups_train = train_cache["groups"]

    test_cache = np.load(os.path.join(project_root, "artifacts", "student_test_features.npz"), allow_pickle=True)
    X_test = test_cache["X"]
    y_test = test_cache["y"]
    test_meta = test_cache["metadata"]

    print(f"Loaded Training set: {X_train.shape} across {len(np.unique(groups_train))} trip groups")
    print(f"Loaded Student-Test set: {X_test.shape} ({len(test_meta)} observation targets)")

    # 2. Define Model Definitions with Specific Hyperparameters
    model_configs = {
        "LogisticRegression": {
            "family": "Linear and Probabilistic",
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, C=1.0, random_state=42, class_weight="balanced"))
            ]),
            "key_config": "C=1.0, L2 penalty, balanced weights"
        },
        "DistanceKNN": {
            "family": "Instance-based",
            # Subsample for efficient instance query
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=-1, algorithm="ball_tree"))
            ]),
            "key_config": "k=15, distance-weighted, BallTree"
        },
        "RandomForest": {
            "family": "Ensemble",
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", RandomForestClassifier(n_estimators=100, max_depth=16, min_samples_leaf=5,
                                               random_state=42, n_jobs=-1, class_weight="balanced_subsample"))
            ]),
            "key_config": "100 trees, max_depth=16, balanced"
        },
        "HistGradientBoosting": {
            "family": "Boosting",
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=31, min_samples_leaf=20,
                                                       learning_rate=0.1, random_state=42, class_weight="balanced"))
            ]),
            "key_config": "100 iterations, lr=0.1, max_leaves=31"
        },
        "MLPClassifier": {
            "family": "Neural Networks",
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu",
                                      max_iter=100, early_stopping=True, random_state=42))
            ]),
            "key_config": "Arch [64, 32], ReLU, Adam, Early Stopping"
        }
    }

    # 3. 5-Fold Group Cross-Validation
    print("\nExecuting 5-Fold GroupKFold Cross-Validation on Training Trips...")
    gkf = GroupKFold(n_splits=5)
    cv_scores = {}

    for name, cfg in model_configs.items():
        print(f"\n--- Running CV for {name} ({cfg['family']}) ---")
        fold_f1s = []
        fold_accs = []
        
        # Subsample for quick instance-based CV if KNN
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups=groups_train)):
            if name == "DistanceKNN" and len(train_idx) > 40000:
                sub_idx = np.random.RandomState(42).choice(train_idx, size=40000, replace=False)
                X_tr, y_tr = X_train[sub_idx], y_train[sub_idx]
            else:
                X_tr, y_tr = X_train[train_idx], y_train[train_idx]
            
            X_val, y_val = X_train[val_idx], y_train[val_idx]
            
            pipe = cfg["pipeline"]
            pipe.fit(X_tr, y_tr)
            y_val_pred = pipe.predict(X_val)
            fold_f1 = f1_score(y_val, y_val_pred, average="macro")
            fold_acc = accuracy_score(y_val, y_val_pred)
            fold_f1s.append(fold_f1)
            fold_accs.append(fold_acc)

        mean_cv_f1 = np.mean(fold_f1s)
        mean_cv_acc = np.mean(fold_accs)
        cv_scores[name] = {"cv_macro_f1": mean_cv_f1, "cv_acc": mean_cv_acc}
        print(f"  {name} 5-Fold Mean Macro F1: {mean_cv_f1:.4f} (Accuracy: {mean_cv_acc:.4f})")

    # 4. Full Training and Evaluation on Student-Test Trips
    print("\n" + "=" * 60)
    print("FULL MODEL TRAINING & STUDENT-TEST EVALUATION")
    print("=" * 60)

    comparison_records = []
    saved_models_dir = os.path.join(project_root, "artifacts", "saved_models")
    os.makedirs(saved_models_dir, exist_ok=True)
    fig_dir = os.path.join(project_root, "reports", "figures")
    os.makedirs(fig_dir, exist_ok=True)

    mlp_loss_curve = None

    # Load road network for student-test route continuity calculations
    loader = TripDataLoader()
    test_trips = loader.list_trips("student_test")
    test_trip_arcs = {}
    for trip_id in test_trips:
        tdata = loader.load_trip(trip_id, "student_test")
        test_trip_arcs[trip_id] = tdata["arcs"]

    for name, cfg in model_configs.items():
        print(f"\nTraining {name} on full dataset...")
        pipe = cfg["pipeline"]

        # Handle KNN sample size for optimal speed
        if name == "DistanceKNN" and len(X_train) > 60000:
            sample_indices = np.random.RandomState(42).choice(len(X_train), size=60000, replace=False)
            X_fit, y_fit = X_train[sample_indices], y_train[sample_indices]
        else:
            X_fit, y_fit = X_train, y_train

        # Time training
        t0 = time.time()
        pipe.fit(X_fit, y_fit)
        train_time = time.time() - t0

        if name == "MLPClassifier":
            mlp_loss_curve = pipe.named_steps["clf"].loss_curve_

        # Time inference on test set
        t0 = time.time()
        test_preds = pipe.predict(X_test)
        test_probs = pipe.predict_proba(X_test)
        inference_time = time.time() - t0
        latency_ms_per_sample = (inference_time / len(X_test)) * 1000.0

        # Metrics on candidate pairs
        macro_f1 = f1_score(y_test, test_preds, average="macro")
        precision = precision_score(y_test, test_preds, average="macro", zero_division=0)
        recall = recall_score(y_test, test_preds, average="macro", zero_division=0)
        try:
            logloss_val = log_loss(y_test, test_probs)
        except Exception:
            logloss_val = np.nan

        # Observation-level Top-1 Accuracy & Route Continuity
        # Group by observation in student-test metadata
        obs_correct = 0
        total_obs = len(test_meta)
        
        # Track predicted sequence per trip for continuity
        trip_pred_arcs = {t_id: [] for t_id in test_trips}

        for meta_item in test_meta:
            trip_id = meta_item["trip_id"]
            start_r = meta_item["start_row"]
            count = meta_item["count"]
            target = meta_item["target_arc"]
            candidates = meta_item["candidates"]

            if count == 0:
                continue

            c_probs = test_probs[start_r : start_r + count, 1]
            best_cand_idx = np.argmax(c_probs)
            pred_arc = candidates[best_cand_idx]
            trip_pred_arcs[trip_id].append(pred_arc)

            if pred_arc == target:
                obs_correct += 1

        top1_acc = obs_correct / total_obs

        # Calculate Route Continuity across all test trips
        total_transitions = 0
        continuous_transitions = 0
        for trip_id, p_arcs in trip_pred_arcs.items():
            arcs_matrix = test_trip_arcs[trip_id]
            for i in range(len(p_arcs) - 1):
                total_transitions += 1
                c1 = p_arcs[i]
                c2 = p_arcs[i + 1]
                if c1 == c2:
                    continuous_transitions += 1
                else:
                    v1 = arcs_matrix[c1, 1]
                    u2 = arcs_matrix[c2, 0]
                    if v1 == u2:
                        continuous_transitions += 1

        route_continuity = continuous_transitions / total_transitions if total_transitions > 0 else 0.0

        # Save model artifact
        model_path = os.path.join(saved_models_dir, f"{name}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(pipe, f)

        # Confusion matrix plot
        cm = confusion_matrix(y_test, test_preds)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["Candidate Miss", "True Arc"],
                    yticklabels=["Candidate Miss", "True Arc"])
        plt.title(f"Confusion Matrix: {name}", fontsize=11, fontweight="bold")
        plt.ylabel("Actual Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        cm_path = os.path.join(fig_dir, f"confusion_matrix_{name}.png")
        plt.savefig(cm_path, dpi=300)
        plt.close()

        record = {
            "Model": name,
            "Algorithm Family": cfg["family"],
            "Key Configuration": cfg["key_config"],
            "CV Macro F1-score": round(cv_scores[name]["cv_macro_f1"], 4),
            "Top-1 Accuracy": round(top1_acc, 4),
            "Route continuity": round(route_continuity, 4),
            "Student-Test Macro F1-score": round(macro_f1, 4),
            "Log Loss": round(logloss_val, 4) if not np.isnan(logloss_val) else "N/A",
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "Training Time (s)": round(train_time, 2),
            "Inference Time (ms/sample)": round(latency_ms_per_sample, 4)
        }
        comparison_records.append(record)
        print(f"Results for {name}:")
        print(f"  Top-1 Accuracy: {top1_acc*100:.2f}% | Continuity: {route_continuity*100:.2f}% | Macro F1: {macro_f1:.4f}")
        print(f"  Training Time: {train_time:.2f}s | Latency: {latency_ms_per_sample:.4f} ms/sample")

    # 5. Consolidated Model Comparison Table
    comparison_df = pd.DataFrame(comparison_records)
    table_csv = os.path.join(project_root, "artifacts", "model_comparison.csv")
    comparison_df.to_csv(table_csv, index=False)
    print("\n" + "=" * 60)
    print("CONSOLIDATED MODEL COMPARISON TABLE")
    print("=" * 60)
    print(comparison_df.to_string(index=False))

    # 6. Comparative Visualizations
    # Chart 1: CV Macro F1 vs Student-Test Macro F1
    plt.figure(figsize=(9, 5))
    x = np.arange(len(comparison_df))
    width = 0.35
    plt.bar(x - width/2, comparison_df["CV Macro F1-score"], width, label="CV Macro F1-score", color="#1976D2")
    plt.bar(x + width/2, comparison_df["Student-Test Macro F1-score"], width, label="Student-Test Macro F1-score", color="#388E3C")
    plt.xticks(x, comparison_df["Model"], rotation=15, fontweight="bold")
    plt.ylabel("Macro F1-Score", fontweight="bold")
    plt.title("Cross-Validation vs Student-Test Macro F1 Across Algorithm Families", fontsize=12, fontweight="bold")
    plt.legend(frameon=True)
    plt.tight_layout()
    chart1_path = os.path.join(fig_dir, "model_comparison_f1_scores.png")
    plt.savefig(chart1_path, dpi=300)
    plt.close()

    # Chart 2: Top-1 Accuracy and Route Continuity
    plt.figure(figsize=(9, 5))
    plt.bar(x - width/2, comparison_df["Top-1 Accuracy"], width, label="Observation Top-1 Accuracy", color="#F57C00")
    plt.bar(x + width/2, comparison_df["Route continuity"], width, label="Route Continuity (Raw Classifier)", color="#7B1FA2")
    plt.xticks(x, comparison_df["Model"], rotation=15, fontweight="bold")
    plt.ylabel("Accuracy / Continuity Ratio", fontweight="bold")
    plt.title("Observation Top-1 Accuracy & Route Continuity Comparison", fontsize=12, fontweight="bold")
    plt.legend(frameon=True)
    plt.tight_layout()
    chart2_path = os.path.join(fig_dir, "model_comparison_accuracy_continuity.png")
    plt.savefig(chart2_path, dpi=300)
    plt.close()

    # Chart 3: MLP Loss Curve
    if mlp_loss_curve is not None:
        plt.figure(figsize=(7, 4))
        plt.plot(mlp_loss_curve, marker="o", color="#D32F2F", linewidth=2)
        plt.title("MLP Classifier Training Loss Curve (Adam Optimizer)", fontsize=11, fontweight="bold")
        plt.xlabel("Epoch / Iteration")
        plt.ylabel("Cross-Entropy Loss")
        plt.tight_layout()
        mlp_path = os.path.join(fig_dir, "mlp_training_loss_curve.png")
        plt.savefig(mlp_path, dpi=300)
        plt.close()
        print(f"Saved MLP loss curve to: {mlp_path}")

    print(f"\nAll comparison figures and models saved successfully!")

if __name__ == "__main__":
    run_model_benchmarking()
