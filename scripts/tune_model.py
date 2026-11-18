import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, accuracy_score, log_loss
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.bayesian_opt import GaussianProcessBayesianOptimizer

def run_bayesian_tuning():
    print("=" * 65)
    print("PATHSNAP: BAYESIAN HYPERPARAMETER OPTIMISATION (GP-EI)")
    print("=" * 65)

    # Load preprocessed training & test features
    train_cache = np.load(os.path.join(project_root, "artifacts", "train_features.npz"), allow_pickle=True)
    X_train = train_cache["X"]
    y_train = train_cache["y"]
    groups_train = train_cache["groups"]

    test_cache = np.load(os.path.join(project_root, "artifacts", "student_test_features.npz"), allow_pickle=True)
    X_test = test_cache["X"]
    y_test = test_cache["y"]
    test_meta = test_cache["metadata"]

    # 1. Define Justified Hyperparameter Search Space
    # Selected Model: HistGradientBoostingClassifier (Boosting family)
    param_bounds = {
        "learning_rate": (0.02, 0.25, "log_float"),     # Step size shrinkage
        "max_leaf_nodes": (15, 63, "int"),              # Tree complexity
        "min_samples_leaf": (10, 80, "int"),            # Regularization against overfitting
        "l2_regularization": (0.001, 5.0, "log_float")  # L2 weight decay
    }

    gkf = GroupKFold(n_splits=3) # 3-fold Group CV on training trips for fast tuning
    fold_splits = list(gkf.split(X_train, y_train, groups=groups_train))

    def objective_function(params: dict) -> float:
        f1_scores = []
        for train_idx, val_idx in fold_splits:
            X_tr, y_tr = X_train[train_idx], y_train[train_idx]
            X_val, y_val = X_train[val_idx], y_train[val_idx]

            model = HistGradientBoostingClassifier(
                learning_rate=params["learning_rate"],
                max_leaf_nodes=params["max_leaf_nodes"],
                min_samples_leaf=params["min_samples_leaf"],
                l2_regularization=params["l2_regularization"],
                max_iter=80,
                random_state=42,
                class_weight="balanced"
            )
            pipe = Pipeline([("scaler", StandardScaler()), ("clf", model)])
            pipe.fit(X_tr, y_tr)
            preds = pipe.predict(X_val)
            f1_scores.append(f1_score(y_val, preds, average="macro"))
        return float(np.mean(f1_scores))

    # 2. Execute 20 Bayesian Optimization Trials
    optimizer = GaussianProcessBayesianOptimizer(
        param_bounds=param_bounds,
        n_initial_points=5,
        n_trials=20,
        random_state=42
    )

    tuning_results = optimizer.optimize(objective_function)
    best_params = tuning_results["best_params"]
    best_cv_score = tuning_results["best_score"]
    trials_df = tuning_results["trials_df"]

    # Save trial records
    trials_csv = os.path.join(project_root, "artifacts", "bayesian_optimization_trials.csv")
    trials_df.to_csv(trials_csv, index=False)
    print(f"\nSaved 20 optimization trials to: {trials_csv}")

    # Plot Optimization Convergence History
    fig_dir = os.path.join(project_root, "reports", "figures")
    plt.figure(figsize=(8, 4.5))
    plt.plot(trials_df["trial"], trials_df["score"], "o--", color="#1976D2", alpha=0.6, label="Trial Score")
    plt.plot(trials_df["trial"], trials_df["best_so_far"], "s-", color="#D32F2F", linewidth=2.2, label="Best Observed Score (EI)")
    plt.xlabel("Bayesian Trial Number", fontweight="bold")
    plt.ylabel("Validation Macro F1-Score", fontweight="bold")
    plt.title("Bayesian Hyperparameter Optimization Convergence (Gaussian Process EI)", fontsize=11, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    conv_plot_path = os.path.join(fig_dir, "bayesian_optimization_convergence.png")
    plt.savefig(conv_plot_path, dpi=300)
    plt.close()
    print(f"Saved convergence plot to: {conv_plot_path}")

    # 3. Refit Tuned Model on Full Training Set
    print("\nRefitting Tuned HistGradientBoosting model on complete training set...")
    t0 = time.time()
    tuned_model = HistGradientBoostingClassifier(
        learning_rate=best_params["learning_rate"],
        max_leaf_nodes=best_params["max_leaf_nodes"],
        min_samples_leaf=best_params["min_samples_leaf"],
        l2_regularization=best_params["l2_regularization"],
        max_iter=120,
        random_state=42,
        class_weight="balanced"
    )
    tuned_pipe = Pipeline([("scaler", StandardScaler()), ("clf", tuned_model)])
    tuned_pipe.fit(X_train, y_train)
    tuned_train_time = time.time() - t0

    # Save tuned model weights
    tuned_model_path = os.path.join(project_root, "artifacts", "saved_models", "HistGradientBoosting_tuned.pkl")
    with open(tuned_model_path, "wb") as f:
        pickle.dump(tuned_pipe, f)
    print(f"Saved tuned model artifact to: {tuned_model_path}")

    # 4. Evaluate Tuned vs Untuned on Student-Test Trips
    print("\nEvaluating Tuned vs Untuned model on 10 Student-Test trips...")
    # Load untuned model
    untuned_path = os.path.join(project_root, "artifacts", "saved_models", "HistGradientBoosting.pkl")
    with open(untuned_path, "rb") as f:
        untuned_pipe = pickle.load(f)

    # Untuned evaluation
    t0 = time.time()
    untuned_preds = untuned_pipe.predict(X_test)
    untuned_probs = untuned_pipe.predict_proba(X_test)
    untuned_infer_time = (time.time() - t0) / len(X_test) * 1000.0
    untuned_f1 = f1_score(y_test, untuned_preds, average="macro")
    untuned_logloss = log_loss(y_test, untuned_probs)

    # Tuned evaluation
    t0 = time.time()
    tuned_preds = tuned_pipe.predict(X_test)
    tuned_probs = tuned_pipe.predict_proba(X_test)
    tuned_infer_time = (time.time() - t0) / len(X_test) * 1000.0
    tuned_f1 = f1_score(y_test, tuned_preds, average="macro")
    tuned_logloss = log_loss(y_test, tuned_probs)

    # Observation-level Top-1 accuracy for both
    obs_correct_untuned = 0
    obs_correct_tuned = 0
    total_obs = len(test_meta)

    for meta in test_meta:
        start_r = meta["start_row"]
        count = meta["count"]
        target = meta["target_arc"]
        candidates = meta["candidates"]
        if count == 0:
            continue

        p_u = untuned_probs[start_r : start_r + count, 1]
        if candidates[np.argmax(p_u)] == target:
            obs_correct_untuned += 1

        p_t = tuned_probs[start_r : start_r + count, 1]
        if candidates[np.argmax(p_t)] == target:
            obs_correct_tuned += 1

    top1_untuned = obs_correct_untuned / total_obs
    top1_tuned = obs_correct_tuned / total_obs

    comparison = pd.DataFrame([
        {
            "Model Variant": "Untuned Baseline (Default Config)",
            "CV Macro F1-score": 0.8119,
            "Student-Testing Top-1 Accuracy": round(top1_untuned, 4),
            "Student-Testing Macro F1-score": round(untuned_f1, 4),
            "Log Loss": round(untuned_logloss, 4),
            "Inference Time (ms/sample)": round(untuned_infer_time, 4)
        },
        {
            "Model Variant": "Tuned (Bayesian GP-EI Optimized)",
            "CV Macro F1-score": round(best_cv_score, 4),
            "Student-Testing Top-1 Accuracy": round(top1_tuned, 4),
            "Student-Testing Macro F1-score": round(tuned_f1, 4),
            "Log Loss": round(tuned_logloss, 4),
            "Inference Time (ms/sample)": round(tuned_infer_time, 4)
        }
    ])

    comp_csv = os.path.join(project_root, "artifacts", "bayesian_tuned_comparison.csv")
    comparison.to_csv(comp_csv, index=False)
    print("\n" + "=" * 65)
    print("BAYESIAN TUNING: UNTUNED VS TUNED MODEL COMPARISON")
    print("=" * 65)
    print(comparison.to_string(index=False))

if __name__ == "__main__":
    run_bayesian_tuning()
