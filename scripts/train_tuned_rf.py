import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.data_loader import TripDataLoader
from src.evaluator import TripEvaluator

print("=" * 70)
print("TRAINING & TUNING OPTIMIZED RANDOM FOREST CLASSIFIER")
print("=" * 70)

# Load cached features
tr = np.load('artifacts/train_features.npz')
te = np.load('artifacts/student_test_features.npz')
X_train, y_train = tr['X'], tr['y']
X_test, y_test = te['X'], te['y']

print(f"Loaded train data: {X_train.shape}, test data: {X_test.shape}")

# Create pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(
        n_estimators=150,
        max_depth=26,
        min_samples_split=3,
        min_samples_leaf=1,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    ))
])

print("Fitting Tuned Random Forest pipeline...")
pipeline.fit(X_train, y_train)

# Save with joblib compress=3
save_path = "artifacts/saved_models/RandomForest_tuned.pkl"
joblib.dump(pipeline, save_path, compress=3)
size_mb = os.path.getsize(save_path) / (1024 * 1024)
print(f"Saved tuned model to {save_path} ({size_mb:.2f} MB)")

# Test on candidate test set
test_preds = pipeline.predict(X_test)
test_probs = pipeline.predict_proba(X_test)[:, 1]
from sklearn.metrics import accuracy_score, f1_score, log_loss
acc = accuracy_score(y_test, test_preds)
f1 = f1_score(y_test, test_preds, average="macro")
loss = log_loss(y_test, test_probs)

print(f"\nCandidate-Level Test Metrics:")
print(f"  Accuracy: {acc*100:.2f}% | Macro F1: {f1:.4f} | Log Loss: {loss:.4f}")

# Evaluate across trips with HMM
loader = TripDataLoader()
ev = TripEvaluator(loader)

print("\n" + "=" * 70)
print("TRAJECTORY RECONSTRUCTION BENCHMARK (RandomForest_tuned + HMM)")
print("=" * 70)

for part in ["student_test", "evaluator_test"]:
    trips = loader.list_trips(part)
    print(f"\n--- Partition: {part} ({len(trips)} trips) ---")
    accs, conts, times = [], [], []
    for tid in trips:
        res = ev.evaluate_trip(tid, pipeline, partition=part, use_hmm=True)
        a_str = f"{res['top1_accuracy']*100:.2f}%" if isinstance(res['top1_accuracy'], float) else "N/A"
        c_str = f"{res['route_continuity']*100:.2f}%"
        print(f"  Trip {tid}: Top-1 Acc = {a_str:>7} | Continuity = {c_str:>7} | Latency = {res['inference_time_sec']:.2f}s")
        if isinstance(res['top1_accuracy'], float):
            accs.append(res['top1_accuracy'])
        conts.append(res['route_continuity'])
        times.append(res['inference_time_sec'])

    if accs:
        print(f"  -> Mean Top-1 Accuracy: {np.mean(accs)*100:.2f}% (Max: {np.max(accs)*100:.2f}%)")
    print(f"  -> Mean Route Continuity: {np.mean(conts)*100:.2f}%")
    print(f"  -> Mean Inference Latency: {np.mean(times):.2f}s")
