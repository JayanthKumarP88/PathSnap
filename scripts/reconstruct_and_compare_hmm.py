import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_loader import TripDataLoader
from src.evaluator import TripEvaluator
from src.visualizer import RouteMapVisualizer

def run_hmm_comparison():
    print("=" * 65)
    print("PATHSNAP: HMM & VITERBI ROUTE RECONSTRUCTION EVALUATION")
    print("=" * 65)

    loader = TripDataLoader()
    test_trips = loader.list_trips("student_test")
    evaluator = TripEvaluator(loader)

    # Load tuned model
    model_path = os.path.join(project_root, "artifacts", "saved_models", "HistGradientBoosting_tuned.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    trip_results = []
    print(f"Evaluating 10 Student-Test trips with Classifier-Only vs Classifier + HMM Viterbi...")

    for trip_idx, trip_id in enumerate(test_trips):
        # 1. Classifier-Only evaluation
        res_clf = evaluator.evaluate_trip(trip_id=trip_id, model=model, partition="student_test", use_hmm=False)

        # 2. Classifier + HMM Viterbi evaluation
        res_hmm = evaluator.evaluate_trip(trip_id=trip_id, model=model, partition="student_test", use_hmm=True)

        record = {
            "Trip ID": trip_id,
            "Observations": res_hmm["observations_count"],
            "Classifier-Only Top-1 Acc": round(res_clf["top1_accuracy"] * 100, 2),
            "HMM-Viterbi Top-1 Acc": round(res_hmm["top1_accuracy"] * 100, 2),
            "Accuracy Delta (%)": round((res_hmm["top1_accuracy"] - res_clf["top1_accuracy"]) * 100, 2),
            "Classifier Route Continuity": round(res_clf["route_continuity"] * 100, 2),
            "HMM Route Continuity": round(res_hmm["route_continuity"] * 100, 2),
            "Continuity Gain (%)": round((res_hmm["route_continuity"] - res_clf["route_continuity"]) * 100, 2),
            "HMM Latency (s)": round(res_hmm["inference_time_sec"], 2)
        }
        trip_results.append(record)
        print(f"Trip {trip_id}: Accuracy {record['Classifier-Only Top-1 Acc']}% -> {record['HMM-Viterbi Top-1 Acc']}% | Continuity {record['Classifier Route Continuity']}% -> {record['HMM Route Continuity']}%")

        # Generate Folium map for the first test trip
        if trip_idx == 0:
            vis = RouteMapVisualizer()
            fmap = vis.build_map(
                track_lonlat=np.array(res_hmm["track_lonlat"]),
                nodes_lonlat=np.array(res_hmm["nodes_lonlat"]),
                arcs=np.array(res_hmm["arcs"]),
                network_arc_ids=res_hmm["bbox_arc_ids"],
                predicted_arc_ids=np.array(res_hmm["reconstructed_arcs"]),
                reference_arc_ids=np.array(res_hmm["reference_arcs"]),
                trip_title=f"PathSnap Reconstruction - Student-Test Trip {trip_id}"
            )
            map_out = os.path.join(project_root, "maps", f"student_test_trip_{trip_id}.html")
            vis.save_map(fmap, map_out)

    results_df = pd.DataFrame(trip_results)
    table_path = os.path.join(project_root, "artifacts", "hmm_vs_classifier_comparison.csv")
    results_df.to_csv(table_path, index=False)

    print("\n" + "=" * 65)
    print("CONSOLIDATED HMM VS CLASSIFIER-ONLY COMPARISON (10 TRIPS)")
    print("=" * 65)
    print(results_df.to_string(index=False))

    mean_clf_acc = results_df["Classifier-Only Top-1 Acc"].mean()
    mean_hmm_acc = results_df["HMM-Viterbi Top-1 Acc"].mean()
    mean_clf_cont = results_df["Classifier Route Continuity"].mean()
    mean_hmm_cont = results_df["HMM Route Continuity"].mean()

    print(f"\nAverage Top-1 Accuracy:  Classifier: {mean_clf_acc:.2f}% | HMM-Viterbi: {mean_hmm_acc:.2f}% (Delta: {mean_hmm_acc - mean_clf_acc:+.2f}%)")
    print(f"Average Route Continuity: Classifier: {mean_clf_cont:.2f}% | HMM-Viterbi: {mean_hmm_cont:.2f}% (Delta: {mean_hmm_cont - mean_clf_cont:+.2f}%)")

    # Plot HMM vs Classifier comparison bar chart
    fig_dir = os.path.join(project_root, "reports", "figures")
    plt.figure(figsize=(10, 5))
    x = np.arange(len(results_df))
    width = 0.35
    plt.bar(x - width/2, results_df["Classifier Route Continuity"], width, label="Classifier-Only Continuity", color="#E53935")
    plt.bar(x + width/2, results_df["HMM Route Continuity"], width, label="HMM-Viterbi Continuity", color="#2E7D32")
    plt.xticks(x, [f"T{tid[-4:]}" for tid in results_df["Trip ID"]], rotation=0, fontweight="bold")
    plt.xlabel("Student-Test Trip ID", fontweight="bold")
    plt.ylabel("Route Continuity (%)", fontweight="bold")
    plt.title("HMM Viterbi Sequential Decoding vs Classifier-Only Route Continuity", fontsize=12, fontweight="bold")
    plt.legend()
    plt.tight_layout()
    chart_path = os.path.join(fig_dir, "hmm_route_continuity_improvement.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"Saved continuity improvement chart to: {chart_path}")

if __name__ == "__main__":
    run_hmm_comparison()
