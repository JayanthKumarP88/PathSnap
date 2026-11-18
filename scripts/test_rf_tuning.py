import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pickle
import joblib
import numpy as np
from src.data_loader import TripDataLoader
from src.evaluator import TripEvaluator

def safe_load(path):
    try:
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)

loader = TripDataLoader()
rf_model = safe_load('artifacts/saved_models/RandomForest.pkl')
hgb_model = safe_load('artifacts/saved_models/HistGradientBoosting_tuned.pkl')

ev = TripEvaluator(loader)

for part in ['student_test', 'evaluator_test']:
    trips = loader.list_trips(part)
    print(f"\n=== Testing {part} ({len(trips)} trips) ===", flush=True)
    # Pick a few quick trips
    test_trips = trips[:3]
    for tid in test_trips:
        res_rf = ev.evaluate_trip(tid, rf_model, partition=part, use_hmm=True)
        res_hgb = ev.evaluate_trip(tid, hgb_model, partition=part, use_hmm=True)
        acc_rf = f"{res_rf['top1_accuracy']*100:.2f}%" if isinstance(res_rf['top1_accuracy'], float) else 'N/A'
        acc_hgb = f"{res_hgb['top1_accuracy']*100:.2f}%" if isinstance(res_hgb['top1_accuracy'], float) else 'N/A'
        print(f"Trip {tid}: RF Acc={acc_rf}, Cont={res_rf['route_continuity']*100:.2f}% (time={res_rf['inference_time_sec']}s) | HGB Acc={acc_hgb}, Cont={res_hgb['route_continuity']*100:.2f}% (time={res_hgb['inference_time_sec']}s)", flush=True)
