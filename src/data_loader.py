import os
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple

class TripDataLoader:
    """
    Robust data loader for PathSnap trip records (.track, .nodes, .arcs, .route).
    Handles relative and absolute path resolution across dataset partitions.
    """
    def __init__(self, data_root: Optional[str] = None):
        if data_root is None:
            # Look for standard hackathon directory layout
            candidate_roots = [
                os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data")),
                os.path.join("..", "ML_hackathon_files", "Student folder", "Student folder"),
                os.path.join("ML_hackathon_files", "Student folder", "Student folder"),
                os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "ML_hackathon_files", "Student folder", "Student folder")),
                os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "evaluator_test", "evaluator_test")),
                os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "evaluator_test"))
            ]
            for cr in candidate_roots:
                if os.path.exists(cr):
                    self.data_root = os.path.abspath(cr)
                    break
            else:
                self.data_root = os.path.abspath("ML_hackathon_files")
        else:
            self.data_root = os.path.abspath(data_root)

    def get_trip_path(self, trip_id: str, partition: str = "train") -> str:
        trip_str = str(trip_id).zfill(8)
        search_dirs = [
            os.path.join(self.data_root, partition, trip_str),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", partition, trip_str)),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "evaluator_test", "evaluator_test", trip_str)),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "evaluator_test", trip_str)),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "ML_hackathon_files", "Student folder", "Student folder", partition, trip_str)),
            os.path.join(self.data_root, "evaluator_test", trip_str),
            os.path.join(self.data_root, "student_test", trip_str),
            os.path.join(self.data_root, "train", trip_str),
            os.path.abspath(trip_str)
        ]
        for p in search_dirs:
            if os.path.exists(p) and os.path.isdir(p):
                return p
        raise FileNotFoundError(f"Trip directory not found for ID '{trip_id}' across candidate directories.")

    def load_trip(self, trip_id: str, partition: str = "train") -> Dict[str, np.ndarray]:
        trip_dir = self.get_trip_path(trip_id, partition)
        trip_str = os.path.basename(trip_dir)
        base = os.path.join(trip_dir, trip_str)

        track_file = f"{base}.track"
        nodes_file = f"{base}.nodes"
        arcs_file = f"{base}.arcs"
        route_file = f"{base}.route"

        if not os.path.exists(track_file):
            raise FileNotFoundError(f"Track file missing: {track_file}")
        if not os.path.exists(nodes_file):
            raise FileNotFoundError(f"Nodes file missing: {nodes_file}")
        if not os.path.exists(arcs_file):
            raise FileNotFoundError(f"Arcs file missing: {arcs_file}")

        track_df = pd.read_csv(track_file, sep=r"\s+", header=None, names=["lon", "lat", "time"], dtype=np.float64)
        nodes_df = pd.read_csv(nodes_file, sep=r"\s+", header=None, names=["lon", "lat"], dtype=np.float64)
        arcs_df = pd.read_csv(arcs_file, sep=r"\s+", header=None, names=["from_node", "to_node"], dtype=np.int64)

        route = None
        if os.path.exists(route_file):
            route_df = pd.read_csv(route_file, sep=r"\s+", header=None, names=["arc_id"], dtype=np.int64)
            route = route_df["arc_id"].values

        return {
            "trip_id": trip_str,
            "track": track_df[["lon", "lat", "time"]].values,
            "nodes": nodes_df[["lon", "lat"]].values,
            "arcs": arcs_df[["from_node", "to_node"]].values,
            "route": route
        }

    def list_trips(self, partition: str = "train") -> list:
        search_roots = [
            os.path.join(self.data_root, partition),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", partition)),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "evaluator_test", "evaluator_test")) if partition == "evaluator_test" else None,
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "evaluator_test")) if partition == "evaluator_test" else None,
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "ML_hackathon_files", "Student folder", "Student folder", partition))
        ]
        for sr in search_roots:
            if sr and os.path.exists(sr):
                trips = sorted([d for d in os.listdir(sr) if os.path.isdir(os.path.join(sr, d)) and d.isdigit()])
                if trips:
                    return trips
        return []
