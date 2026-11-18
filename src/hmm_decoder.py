import numpy as np
import networkx as nx
from typing import List, Tuple, Dict, Any, Optional

class HMMRouteDecoder:
    """
    Hidden Markov Model and Viterbi decoder for sequential map-matching.
    Combines classifier emission scores with road-network topological
    transition constraints to resolve sensor noise and ensure route continuity.
    """
    def __init__(self, arcs: np.ndarray, nodes_xy: np.ndarray,
                 sigma_z: float = 8.0, beta: float = 5.0):
        self.arcs = arcs
        self.nodes_xy = nodes_xy
        self.sigma_z = sigma_z
        self.beta = beta

        # Build directed adjacency lookup for O(1) connectivity queries
        self.arc_endpoints = {}
        self.outgoing_arcs = {}
        for arc_id, (u, v) in enumerate(arcs):
            u_int, v_int = int(u), int(v)
            self.arc_endpoints[arc_id] = (u_int, v_int)
            if u_int not in self.outgoing_arcs:
                self.outgoing_arcs[u_int] = []
            self.outgoing_arcs[u_int].append(arc_id)

    def compute_transition_log_prob(self, c_prev: int, c_curr: int,
                                    euc_dist: float) -> float:
        """
        Calculates log transition probability between candidate road segments.
        Direct topological connections are strongly rewarded; disconnected
        transitions incur heavy penalties.
        """
        if c_prev == c_curr:
            # Vehicle remains on the exact same road segment
            return -0.05

        u_prev, v_prev = self.arc_endpoints.get(c_prev, (-1, -1))
        u_curr, v_curr = self.arc_endpoints.get(c_curr, (-2, -2))

        # Direct topological successor: end node of prev matches start node of curr
        if v_prev == u_curr:
            return -0.25

        # 2-hop topological neighbor: check outgoing arcs from v_prev
        if v_prev in self.outgoing_arcs:
            for intermediate_arc in self.outgoing_arcs[v_prev]:
                _, mid_v = self.arc_endpoints[intermediate_arc]
                if mid_v == u_curr:
                    return -1.5

        # Disconnected or reverse transition penalty
        return -10.0

    def decode(self, observation_candidates: List[List[int]],
               observation_features: List[np.ndarray],
               classifier: Any,
               track_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Executes log-space Viterbi decoding over all trajectory observations.
        Returns:
            decoded_arcs: np.ndarray of shape (T,) - sequential HMM decoded arc IDs
            classifier_arcs: np.ndarray of shape (T,) - pointwise top-1 classifier arc IDs
        """
        T = len(observation_candidates)
        if T == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

        # Precompute classifier emission probabilities
        emissions_log = []
        classifier_arcs = np.zeros(T, dtype=np.int64)

        for t in range(T):
            cands = observation_candidates[t]
            feats = observation_features[t]

            # Classifier confidence (P(y=1))
            probs = classifier.predict_proba(feats)[:, 1]
            probs = np.clip(probs, 1e-6, 1.0 - 1e-6)

            # Pointwise top-1 candidate
            best_idx = np.argmax(probs)
            classifier_arcs[t] = cands[best_idx]

            # Combine classifier score with physical distance Gaussian likelihood
            dists = feats[:, 0]  # feature 0 is perpendicular distance
            spatial_penalty = -(dists ** 2) / (2.0 * (self.sigma_z ** 2))
            log_emissions = np.log(probs) + spatial_penalty
            emissions_log.append(log_emissions)

        # Initialize Viterbi structures
        viterbi_trellis = []
        backpointers = []

        # Step 0: Initial state log-probabilities
        viterbi_trellis.append(emissions_log[0])
        backpointers.append(np.zeros(len(observation_candidates[0]), dtype=int))

        # Dynamic programming forward pass
        for t in range(1, T):
            prev_cands = observation_candidates[t - 1]
            curr_cands = observation_candidates[t]
            prev_log_probs = viterbi_trellis[t - 1]
            curr_log_emissions = emissions_log[t]

            euc_dist = float(np.linalg.norm(track_xy[t] - track_xy[t - 1]))

            curr_probs = np.zeros(len(curr_cands), dtype=np.float64)
            curr_bp = np.zeros(len(curr_cands), dtype=int)

            for j, c_j in enumerate(curr_cands):
                # Vectorized/fast transition scoring
                trans_scores = np.array([
                    self.compute_transition_log_prob(c_i, c_j, euc_dist)
                    for c_i in prev_cands
                ], dtype=np.float64)

                total_scores = prev_log_probs + trans_scores
                best_i = int(np.argmax(total_scores))
                curr_probs[j] = total_scores[best_i] + curr_log_emissions[j]
                curr_bp[j] = best_i

            viterbi_trellis.append(curr_probs)
            backpointers.append(curr_bp)

        # Viterbi traceback
        decoded_indices = np.zeros(T, dtype=int)
        decoded_indices[-1] = int(np.argmax(viterbi_trellis[-1]))

        for t in range(T - 2, -1, -1):
            decoded_indices[t] = backpointers[t + 1][decoded_indices[t + 1]]

        decoded_arcs = np.array([
            observation_candidates[t][decoded_indices[t]] for t in range(T)
        ], dtype=np.int64)

        return decoded_arcs, classifier_arcs

    def evaluate_route_continuity(self, arc_sequence: np.ndarray) -> float:
        """
        Calculates the topological continuity percentage of an arc sequence.
        Returns: float in range [0.0, 1.0].
        """
        if len(arc_sequence) <= 1:
            return 1.0

        continuous_steps = 0
        total_transitions = len(arc_sequence) - 1

        for t in range(total_transitions):
            c_prev = int(arc_sequence[t])
            c_curr = int(arc_sequence[t + 1])

            if c_prev == c_curr:
                continuous_steps += 1
                continue

            u_prev, v_prev = self.arc_endpoints.get(c_prev, (-1, -1))
            u_curr, v_curr = self.arc_endpoints.get(c_curr, (-2, -2))

            if v_prev == u_curr:
                continuous_steps += 1

        return float(continuous_steps / total_transitions)
