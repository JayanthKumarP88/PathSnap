import numpy as np
from typing import List, Tuple, Dict, Any

FEATURE_NAMES = [
    "dist",             # Perpendicular distance to road segment (m)
    "t_proj",           # Normalized projection location [0.0, 1.0]
    "cos_heading",      # Cosine similarity between GPS heading & road segment
    "heading_diff_deg", # Absolute angular difference in degrees [0, 180]
    "arc_length",       # Physical length of road segment (m)
    "gps_speed",        # Instantaneous vehicle speed (m/s)
    "dist_u",           # Distance to start node (m)
    "dist_v",           # Distance to end node (m)
    "dist_rank",        # Candidate rank sorted by distance (0, 1, 2, ...)
    "dist_ratio"        # Distance ratio relative to closest candidate
]

class FeatureExtractor:
    """
    Extracts 10 highly discriminatory geometric and kinematic features
    for GPS-to-road-candidate association.
    """
    def __init__(self, nodes_xy: np.ndarray, arcs: np.ndarray):
        self.nodes_xy = nodes_xy
        self.arcs = arcs
        u_all = arcs[:, 0]
        v_all = arcs[:, 1]
        self.A_all = nodes_xy[u_all]
        self.B_all = nodes_xy[v_all]
        self.vec_all = self.B_all - self.A_all
        self.len_all = np.linalg.norm(self.vec_all, axis=1)

    def extract_observation_features(self, P: np.ndarray, v_gps: np.ndarray,
                                     gps_speed: float, candidate_arcs: List[int]) -> np.ndarray:
        """
        Extracts feature vectors for all candidate arcs for a single GPS observation.
        Returns: np.ndarray of shape (num_candidates, 10).
        """
        if not candidate_arcs:
            return np.empty((0, len(FEATURE_NAMES)), dtype=np.float64)

        c_indices = np.array(candidate_arcs, dtype=np.int64)
        A = self.A_all[c_indices]
        B = self.B_all[c_indices]
        v_seg = self.vec_all[c_indices]
        L_seg = np.maximum(self.len_all[c_indices], 1e-6)

        # Vectorized projection
        diff = P[None, :] - A
        dot = np.sum(diff * v_seg, axis=1)
        t_proj = np.clip(dot / (L_seg ** 2), 0.0, 1.0)
        proj = A + t_proj[:, None] * v_seg
        dist = np.linalg.norm(P[None, :] - proj, axis=1)

        # Distances to endpoints
        dist_u = np.linalg.norm(diff, axis=1)
        dist_v = np.linalg.norm(P[None, :] - B, axis=1)

        # Heading alignment
        v_gps_len = float(gps_speed) + 1e-6
        cos_heading = np.sum(v_gps[None, :] * v_seg, axis=1) / (v_gps_len * L_seg)
        cos_heading = np.clip(cos_heading, -1.0, 1.0)
        heading_diff_deg = np.degrees(np.arccos(cos_heading))

        # Relative ranking and ratio
        min_dist = np.min(dist)
        dist_rank = np.argsort(np.argsort(dist)).astype(np.float64)
        dist_ratio = dist / (min_dist + 1e-3)

        speed_arr = np.full(len(c_indices), gps_speed, dtype=np.float64)

        feature_matrix = np.column_stack([
            dist,
            t_proj,
            cos_heading,
            heading_diff_deg,
            L_seg,
            speed_arr,
            dist_u,
            dist_v,
            dist_rank,
            dist_ratio
        ])

        return feature_matrix
