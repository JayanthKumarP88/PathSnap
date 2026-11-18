import time
import numpy as np
from typing import Dict, Any, List, Optional
from src.data_loader import TripDataLoader
from src.projection import LocalCoordinateProjector
from src.candidate_gen import CandidateGenerator, align_route_to_track
from src.features import FeatureExtractor
from src.hmm_decoder import HMMRouteDecoder

class TripEvaluator:
    """
    End-to-end evaluation and inference engine for a single trip.
    Integrates spatial projection, candidate generation, model scoring,
    and HMM Viterbi sequential route reconstruction.
    """
    def __init__(self, data_loader: Optional[TripDataLoader] = None):
        self.loader = data_loader if data_loader is not None else TripDataLoader()

    def evaluate_trip(self, trip_id: str, model: Any,
                      partition: str = "student_test",
                      use_hmm: bool = True,
                      search_radius: float = 35.0) -> Dict[str, Any]:
        start_time = time.time()
        trip_str = str(trip_id).zfill(8)

        # 1. Load Raw Trip Files
        data = self.loader.load_trip(trip_str, partition)
        track = data["track"]      # (lon, lat, time)
        nodes = data["nodes"]      # (lon, lat)
        arcs = data["arcs"]        # (from_node, to_node)
        route = data["route"]      # reference route arc_ids (may be None if hidden)

        T = len(track)
        ref_lon = float(np.mean(track[:, 0]))
        ref_lat = float(np.mean(track[:, 1]))

        # 2. Local Metric Coordinate Projection
        projector = LocalCoordinateProjector(ref_lon, ref_lat)
        track_xy = projector.to_xy(track[:, 0], track[:, 1])
        nodes_xy = projector.to_xy(nodes[:, 0], nodes[:, 1])

        # 3. Ground Truth Alignment (Server-side only if reference route available)
        true_arcs = None
        if route is not None and len(route) > 0:
            true_arcs = align_route_to_track(track_xy, nodes_xy, arcs, route)

        # 4. Spatial Candidate Generation
        c_gen = CandidateGenerator(nodes_xy, arcs, track_xy, sample_spacing=25.0)
        f_ext = FeatureExtractor(nodes_xy, arcs)

        # Compute Kinematics (Velocities & Headings)
        gps_vecs = np.zeros_like(track_xy)
        gps_vecs[1:] = track_xy[1:] - track_xy[:-1]
        gps_vecs[0] = gps_vecs[1]
        dt_arr = np.diff(track[:, 2])
        dt_arr = np.concatenate([[1.0], np.maximum(dt_arr, 0.1)])
        gps_speeds = np.linalg.norm(gps_vecs, axis=1) / dt_arr

        observation_cands = []
        observation_feats = []

        for t in range(T):
            cands = c_gen.get_candidates(track_xy[t], radius=search_radius, fallback_k=5)
            feats = f_ext.extract_observation_features(
                P=track_xy[t],
                v_gps=gps_vecs[t],
                gps_speed=gps_speeds[t],
                candidate_arcs=cands
            )
            observation_cands.append(cands)
            observation_feats.append(feats)

        # 5. Route Reconstruction (Classifier vs HMM Viterbi)
        decoder = HMMRouteDecoder(arcs, nodes_xy, sigma_z=8.0)
        decoded_arcs, classifier_arcs = decoder.decode(
            observation_candidates=observation_cands,
            observation_features=observation_feats,
            classifier=model,
            track_xy=track_xy
        )

        final_predicted_arcs = decoded_arcs if use_hmm else classifier_arcs

        # 6. Compute Evaluation Metrics
        accuracy = None
        if true_arcs is not None:
            accuracy = float(np.mean(final_predicted_arcs == true_arcs))

        route_continuity = decoder.evaluate_route_continuity(final_predicted_arcs)
        elapsed_time = time.time() - start_time

        # Candidate Miss Rate for this trip
        cand_misses = 0
        if true_arcs is not None:
            for t in range(T):
                if true_arcs[t] not in observation_cands[t]:
                    cand_misses += 1
        cand_miss_rate = float(cand_misses / T) if true_arcs is not None else 0.0

        return {
            "trip_id": trip_str,
            "partition": partition,
            "observations_count": T,
            "use_hmm": use_hmm,
            "top1_accuracy": round(accuracy, 4) if accuracy is not None else "N/A",
            "route_continuity": round(route_continuity, 4),
            "candidate_miss_rate": round(cand_miss_rate, 4),
            "inference_time_sec": round(elapsed_time, 3),
            "reconstructed_arcs": final_predicted_arcs.tolist(),
            "raw_classifier_arcs": classifier_arcs.tolist(),
            "reference_arcs": route.tolist() if route is not None else [],
            "track_lonlat": track[:, :2].tolist(),
            "nodes_lonlat": nodes.tolist(),
            "arcs": arcs.tolist(),
            "bbox_arc_ids": c_gen.bbox_arcs.tolist()
        }
