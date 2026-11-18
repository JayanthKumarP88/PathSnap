import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure PathSnap root is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_loader import TripDataLoader
from src.projection import LocalCoordinateProjector
from src.candidate_gen import CandidateGenerator, align_route_to_track
from src.features import FeatureExtractor, FEATURE_NAMES
from src.visualizer import RouteMapVisualizer

def run_preprocessing_and_eda():
    print("=" * 60)
    print("PATHSNAP: PREPROCESSING & EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 60)

    loader = TripDataLoader()
    train_trips = loader.list_trips("train")
    test_trips = loader.list_trips("student_test")

    print(f"Discovered {len(train_trips)} Training trips and {len(test_trips)} Student-Test trips.")

    # 1. Dataset Inventory & Missing Value Analysis
    inventory = []
    dt_frequencies = {}
    total_missing = 0
    all_traj_lengths = []
    all_network_arcs = []
    all_network_nodes = []

    print("\nScanning dataset partitions for missing values, sampling rates, and network statistics...")
    for trip_id in train_trips:
        data = loader.load_trip(trip_id, "train")
        track, nodes, arcs, route = data["track"], data["nodes"], data["arcs"], data["route"]

        # Check NaNs
        nans = np.isnan(track).sum() + np.isnan(nodes).sum() + np.isnan(arcs).sum()
        total_missing += nans

        # Sampling rates
        dts = np.round(np.diff(track[:, 2]), 1)
        for dt_val in dts:
            dt_frequencies[dt_val] = dt_frequencies.get(dt_val, 0) + 1

        all_traj_lengths.append(len(track))
        all_network_nodes.append(len(nodes))
        all_network_arcs.append(len(arcs))

        inventory.append({
            "trip_id": trip_id,
            "partition": "train",
            "track_observations": len(track),
            "nodes_count": len(nodes),
            "arcs_count": len(arcs),
            "route_arcs_count": len(route) if route is not None else 0,
            "nan_count": int(nans)
        })

    inventory_df = pd.DataFrame(inventory)
    os.makedirs(os.path.join(project_root, "artifacts"), exist_ok=True)
    inventory_df.to_csv(os.path.join(project_root, "artifacts", "dataset_inventory.csv"), index=False)
    print(f"Saved dataset inventory to artifacts/dataset_inventory.csv")
    print(f"Total NaN / missing values detected: {total_missing}")

    # Top dt frequencies
    total_intervals = sum(dt_frequencies.values())
    print("\nSampling Interval Distribution:")
    for dt_k in sorted(dt_frequencies.keys())[:5]:
        cnt = dt_frequencies[dt_k]
        print(f"  dt = {dt_k}s: {cnt} instances ({cnt / total_intervals * 100:.2f}%)")

    # 2. Generate EDA Visualizations
    fig_dir = os.path.join(project_root, "reports", "figures")
    os.makedirs(fig_dir, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    # Figure 1: Trajectory Length and Network Arcs Distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(all_traj_lengths, bins=25, kde=True, ax=axes[0], color="#1976D2")
    axes[0].set_title("GPS Trajectory Length Distribution (Observations/Trip)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Observation Count")
    axes[0].set_ylabel("Number of Trips")

    sns.histplot(all_network_arcs, bins=25, kde=True, ax=axes[1], color="#388E3C")
    axes[1].set_title("Local Road Network Complexity (Arcs/Trip)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Number of Arcs")
    axes[1].set_ylabel("Number of Trips")

    plt.tight_layout()
    fig1_path = os.path.join(fig_dir, "eda_trajectory_and_network_distributions.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"Saved EDA distribution figure to: {fig1_path}")

    # Figure 2: Sampling Interval Distribution Pie/Bar
    fig, ax = plt.subplots(figsize=(7, 4))
    common_dts = {f"{k}s": v for k, v in sorted(dt_frequencies.items(), key=lambda x: x[1], reverse=True)[:4]}
    common_dts["Other"] = sum(v for k, v in dt_frequencies.items() if f"{k}s" not in common_dts)
    ax.bar(common_dts.keys(), common_dts.values(), color="#F57C00", edgecolor="black", alpha=0.85)
    ax.set_title("GPS Sampling Frequency Regularity (Nominal 1 Hz)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time Delta (dt)")
    ax.set_ylabel("Occurrence Count")
    for i, (k, v) in enumerate(common_dts.items()):
        ax.text(i, v + max(common_dts.values()) * 0.02, f"{v / total_intervals * 100:.1f}%", ha="center", fontweight="bold")
    plt.tight_layout()
    fig2_path = os.path.join(fig_dir, "eda_sampling_intervals.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"Saved EDA sampling interval figure to: {fig2_path}")

    # 3. Interactive Folium Map for a Sample Trip
    sample_trip_id = train_trips[0]
    print(f"\nBuilding interactive Folium map for sample trip '{sample_trip_id}'...")
    sample_data = loader.load_trip(sample_trip_id, "train")
    track, nodes, arcs, route = sample_data["track"], sample_data["nodes"], sample_data["arcs"], sample_data["route"]

    projector = LocalCoordinateProjector(np.mean(track[:, 0]), np.mean(track[:, 1]))
    track_xy = projector.to_xy(track[:, 0], track[:, 1])
    nodes_xy = projector.to_xy(nodes[:, 0], nodes[:, 1])

    cand_gen = CandidateGenerator(nodes_xy, arcs, track_xy, sample_spacing=25.0)
    visualizer = RouteMapVisualizer()
    fmap = visualizer.build_map(
        track_lonlat=track[:, :2],
        nodes_lonlat=nodes,
        arcs=arcs,
        network_arc_ids=cand_gen.bbox_arcs.tolist(),
        predicted_arc_ids=route, # In EDA, show ground-truth route
        reference_arc_ids=route,
        trip_title=f"PathSnap EDA - Trip {sample_trip_id}"
    )
    map_path = os.path.join(project_root, "maps", f"eda_trip_{sample_trip_id}.html")
    visualizer.save_map(fmap, map_path)

    # 4. Feature Extraction on Training Subset (20 Diverse Trips)
    print("\n" + "=" * 60)
    print("FEATURE EXTRACTION & CANDIDATE GENERATION PIPELINE")
    print("=" * 60)

    selected_train_trips = train_trips[:20] # 20 trips for rich, diverse feature set
    print(f"Extracting candidate feature pairs across {len(selected_train_trips)} training trips...")

    train_X_list = []
    train_y_list = []
    train_groups = []
    obs_count_total = 0
    cand_misses = 0

    t0 = time.time()
    for trip_idx, trip_id in enumerate(selected_train_trips):
        t_start = time.time()
        tdata = loader.load_trip(trip_id, "train")
        t_track, t_nodes, t_arcs, t_route = tdata["track"], tdata["nodes"], tdata["arcs"], tdata["route"]
        if t_route is None:
            continue

        proj = LocalCoordinateProjector(np.mean(t_track[:, 0]), np.mean(t_track[:, 1]))
        t_track_xy = proj.to_xy(t_track[:, 0], t_track[:, 1])
        t_nodes_xy = proj.to_xy(t_nodes[:, 0], t_nodes[:, 1])

        # Align ground truth route to track
        true_arcs = align_route_to_track(t_track_xy, t_nodes_xy, t_arcs, t_route)

        # Candidate generator
        c_gen = CandidateGenerator(t_nodes_xy, t_arcs, t_track_xy, sample_spacing=25.0)
        f_ext = FeatureExtractor(t_nodes_xy, t_arcs)

        # Velocities
        gps_vecs = np.zeros_like(t_track_xy)
        gps_vecs[1:] = t_track_xy[1:] - t_track_xy[:-1]
        gps_vecs[0] = gps_vecs[1]
        dt_arr = np.diff(t_track[:, 2])
        dt_arr = np.concatenate([[1.0], np.maximum(dt_arr, 0.1)])
        gps_speeds = np.linalg.norm(gps_vecs, axis=1) / dt_arr

        for t in range(len(t_track_xy)):
            obs_count_total += 1
            target_arc = true_arcs[t]
            cands = c_gen.get_candidates(t_track_xy[t], radius=35.0, fallback_k=5)

            if target_arc not in cands:
                cand_misses += 1

            feats = f_ext.extract_observation_features(
                P=t_track_xy[t],
                v_gps=gps_vecs[t],
                gps_speed=gps_speeds[t],
                candidate_arcs=cands
            )

            for i, c in enumerate(cands):
                train_X_list.append(feats[i])
                train_y_list.append(1 if c == target_arc else 0)
                train_groups.append(trip_idx)

        print(f"[{trip_idx+1:02d}/{len(selected_train_trips)}] Trip {trip_id}: {len(t_track)} obs processed in {time.time()-t_start:.1f}s")

    train_X = np.array(train_X_list, dtype=np.float32)
    train_y = np.array(train_y_list, dtype=np.int32)
    train_groups = np.array(train_groups, dtype=np.int32)

    print(f"\nFeature extraction complete in {time.time()-t0:.1f}s!")
    print(f"Total observations: {obs_count_total}")
    print(f"Candidate Miss Rate: {cand_misses / obs_count_total * 100:.3f}% ({cand_misses}/{obs_count_total})")
    print(f"Total candidate rows: {len(train_X)}")
    print(f"Positive labels: {np.sum(train_y == 1)} ({np.mean(train_y)*100:.2f}%)")
    print(f"Negative labels: {np.sum(train_y == 0)}")

    # Save preprocessed features to artifacts
    train_cache_path = os.path.join(project_root, "artifacts", "train_features.npz")
    np.savez_compressed(
        train_cache_path,
        X=train_X,
        y=train_y,
        groups=train_groups,
        feature_names=np.array(FEATURE_NAMES)
    )
    print(f"Saved preprocessed training features to: {train_cache_path}")

    # Also prepare Student-Test features for rapid evaluation
    print("\nPreparing Student-Test dataset features...")
    test_X_list = []
    test_y_list = []
    test_obs_metadata = [] # stores (trip_id, obs_idx, target_arc, candidate_arcs, cand_start_row, cand_count)

    row_cursor = 0
    for trip_idx, trip_id in enumerate(test_trips):
        tdata = loader.load_trip(trip_id, "student_test")
        t_track, t_nodes, t_arcs, t_route = tdata["track"], tdata["nodes"], tdata["arcs"], tdata["route"]

        proj = LocalCoordinateProjector(np.mean(t_track[:, 0]), np.mean(t_track[:, 1]))
        t_track_xy = proj.to_xy(t_track[:, 0], t_track[:, 1])
        t_nodes_xy = proj.to_xy(t_nodes[:, 0], t_nodes[:, 1])

        true_arcs = align_route_to_track(t_track_xy, t_nodes_xy, t_arcs, t_route) if t_route is not None else np.zeros(len(t_track), dtype=int)
        c_gen = CandidateGenerator(t_nodes_xy, t_arcs, t_track_xy, sample_spacing=25.0)
        f_ext = FeatureExtractor(t_nodes_xy, t_arcs)

        gps_vecs = np.zeros_like(t_track_xy)
        gps_vecs[1:] = t_track_xy[1:] - t_track_xy[:-1]
        gps_vecs[0] = gps_vecs[1]
        dt_arr = np.diff(t_track[:, 2])
        dt_arr = np.concatenate([[1.0], np.maximum(dt_arr, 0.1)])
        gps_speeds = np.linalg.norm(gps_vecs, axis=1) / dt_arr

        for t in range(len(t_track_xy)):
            cands = c_gen.get_candidates(t_track_xy[t], radius=35.0, fallback_k=5)
            feats = f_ext.extract_observation_features(
                P=t_track_xy[t],
                v_gps=gps_vecs[t],
                gps_speed=gps_speeds[t],
                candidate_arcs=cands
            )
            cand_count = len(cands)
            target = true_arcs[t]

            for i, c in enumerate(cands):
                test_X_list.append(feats[i])
                test_y_list.append(1 if c == target else 0)

            test_obs_metadata.append({
                "trip_id": trip_id,
                "obs_idx": t,
                "target_arc": int(target),
                "candidates": cands,
                "start_row": row_cursor,
                "count": cand_count
            })
            row_cursor += cand_count

    test_X = np.array(test_X_list, dtype=np.float32)
    test_y = np.array(test_y_list, dtype=np.int32)
    test_cache_path = os.path.join(project_root, "artifacts", "student_test_features.npz")
    np.savez_compressed(
        test_cache_path,
        X=test_X,
        y=test_y,
        metadata=pd.DataFrame(test_obs_metadata).to_dict(orient="records")
    )
    print(f"Saved Student-Test features to: {test_cache_path} ({len(test_X)} rows)")
    print("\nPREPROCESSING & EDA COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    run_preprocessing_and_eda()
