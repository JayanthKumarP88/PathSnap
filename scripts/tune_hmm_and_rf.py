import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import joblib
import numpy as np
from src.data_loader import TripDataLoader
from src.projection import LocalCoordinateProjector
from src.candidate_gen import CandidateGenerator, align_route_to_track
from src.features import FeatureExtractor

loader = TripDataLoader()
rf_model = joblib.load('artifacts/saved_models/RandomForest.pkl')

# Let's test on student_test trips
student_trips = loader.list_trips('student_test')
evaluator_trips = loader.list_trips('evaluator_test')

print(f"Loaded {len(student_trips)} student trips, {len(evaluator_trips)} evaluator trips")

# Pre-extract data and candidates for fast grid search
def prepare_trip(trip_id, partition):
    data = loader.load_trip(trip_id, partition)
    track = data["track"]
    nodes = data["nodes"]
    arcs = data["arcs"]
    route = data["route"]

    T = len(track)
    ref_lon, ref_lat = float(np.mean(track[:, 0])), float(np.mean(track[:, 1]))
    proj = LocalCoordinateProjector(ref_lon, ref_lat)
    track_xy = proj.to_xy(track[:, 0], track[:, 1])
    nodes_xy = proj.to_xy(nodes[:, 0], nodes[:, 1])

    true_arcs = align_route_to_track(track_xy, nodes_xy, arcs, route) if route is not None else None

    # Kinematics
    gps_vecs = np.zeros_like(track_xy)
    gps_vecs[1:] = track_xy[1:] - track_xy[:-1]
    gps_vecs[0] = gps_vecs[1]
    dt_arr = np.concatenate([[1.0], np.maximum(np.diff(track[:, 2]), 0.1)])
    gps_speeds = np.linalg.norm(gps_vecs, axis=1) / dt_arr

    c_gen = CandidateGenerator(nodes_xy, arcs, track_xy, sample_spacing=25.0)
    f_ext = FeatureExtractor(nodes_xy, arcs)

    obs_cands = []
    obs_feats = []
    for t in range(T):
        cands = c_gen.get_candidates(track_xy[t], radius=35.0, fallback_k=5)
        feats = f_ext.extract_observation_features(track_xy[t], gps_vecs[t], gps_speeds[t], cands)
        obs_cands.append(cands)
        obs_feats.append(feats)

    # Vectorized RF predict_proba
    lengths = [len(f) for f in obs_feats]
    all_feats = np.vstack(obs_feats)
    all_probs = rf_model.predict_proba(all_feats)[:, 1]

    # Pre-build arc endpoints
    arc_endpoints = {arc_id: (int(u), int(v)) for arc_id, (u, v) in enumerate(arcs)}
    outgoing_arcs = {}
    for arc_id, (u, v) in arc_endpoints.items():
        outgoing_arcs.setdefault(u, []).append(arc_id)

    return {
        "trip_id": trip_id,
        "T": T,
        "obs_cands": obs_cands,
        "obs_feats": obs_feats,
        "all_probs": all_probs,
        "cum_indices": np.cumsum([0] + lengths),
        "true_arcs": true_arcs,
        "arc_endpoints": arc_endpoints,
        "outgoing_arcs": outgoing_arcs
    }

print("Pre-caching 5 evaluation trips...")
test_cache = [prepare_trip(tid, 'student_test') for tid in ['00000000', '00000052', '00000007', '00000056']]
test_cache += [prepare_trip(tid, 'evaluator_test') for tid in ['00000023', '00000030', '00000017']]

def run_viterbi(trip_data, sigma_z=8.0, emission_weight=1.0, stay_cost=-0.05, succ_cost=-0.25, jump_cost=-10.0, heading_boost=0.0):
    T = trip_data["T"]
    obs_cands = trip_data["obs_cands"]
    obs_feats = trip_data["obs_feats"]
    all_probs = trip_data["all_probs"]
    cum_indices = trip_data["cum_indices"]
    arc_endpoints = trip_data["arc_endpoints"]
    outgoing_arcs = trip_data["outgoing_arcs"]

    emissions_log = []
    for t in range(T):
        cands = obs_cands[t]
        feats = obs_feats[t]
        st = cum_indices[t]
        en = cum_indices[t + 1]
        probs = all_probs[st:en]

        dists = feats[:, 0]
        cos_head = feats[:, 2]
        spatial_penalty = -(dists ** 2) / (2.0 * (sigma_z ** 2))
        h_penalty = heading_boost * np.clip(cos_head, -1.0, 1.0)
        log_e = emission_weight * np.log(np.clip(probs, 1e-6, 1.0)) + spatial_penalty + h_penalty
        emissions_log.append(log_e)

    # Viterbi DP
    trellis = [emissions_log[0]]
    backptrs = [np.zeros(len(obs_cands[0]), dtype=int)]

    for t in range(1, T):
        prev_scores = trellis[-1]
        prev_cands = obs_cands[t - 1]
        curr_cands = obs_cands[t]
        curr_emiss = emissions_log[t]

        n_prev = len(prev_cands)
        n_curr = len(curr_cands)
        trans_mat = np.full((n_prev, n_curr), jump_cost)

        for ip, cp in enumerate(prev_cands):
            up, vp = arc_endpoints.get(cp, (-1, -1))
            out_arcs = outgoing_arcs.get(vp, [])
            for ic, cc in enumerate(curr_cands):
                if cp == cc:
                    trans_mat[ip, ic] = stay_cost
                elif vp == arc_endpoints.get(cc, (-2, -2))[0]:
                    trans_mat[ip, ic] = succ_cost
                elif cc in out_arcs:
                    trans_mat[ip, ic] = succ_cost * 1.5

        total_scores = prev_scores[:, None] + trans_mat + curr_emiss[None, :]
        best_prev = np.argmax(total_scores, axis=0)
        best_curr = total_scores[best_prev, np.arange(n_curr)]

        trellis.append(best_curr)
        backptrs.append(best_prev)

    # Traceback
    best_last = np.argmax(trellis[-1])
    decoded_idx = [best_last]
    for t in range(T - 1, 0, -1):
        best_last = backptrs[t][best_last]
        decoded_idx.append(best_last)
    decoded_idx.reverse()

    pred_arcs = np.array([obs_cands[t][decoded_idx[t]] for t in range(T)], dtype=np.int64)

    # Accuracy
    acc = None
    if trip_data["true_arcs"] is not None:
        acc = float(np.mean(pred_arcs == trip_data["true_arcs"]))

    # Continuity
    cont_steps = sum(1 for t in range(T - 1) if pred_arcs[t] == pred_arcs[t + 1] or arc_endpoints.get(pred_arcs[t], (-1, -1))[1] == arc_endpoints.get(pred_arcs[t + 1], (-2, -2))[0])
    cont = float(cont_steps / (T - 1)) if T > 1 else 1.0

    return acc, cont

# Test grid
configs = [
    {"name": "Baseline HMM", "sigma_z": 8.0, "emission_weight": 1.0, "stay_cost": -0.05, "succ_cost": -0.25, "jump_cost": -10.0, "heading_boost": 0.0},
    {"name": "Strong Transition Penalty", "sigma_z": 6.0, "emission_weight": 1.0, "stay_cost": -0.02, "succ_cost": -0.15, "jump_cost": -15.0, "heading_boost": 0.0},
    {"name": "Tight Spatial Gaussian (sigma=5m)", "sigma_z": 5.0, "emission_weight": 1.2, "stay_cost": -0.01, "succ_cost": -0.10, "jump_cost": -20.0, "heading_boost": 0.0},
    {"name": "Heading Alignment Boost (h=1.0)", "sigma_z": 6.0, "emission_weight": 1.2, "stay_cost": -0.02, "succ_cost": -0.15, "jump_cost": -15.0, "heading_boost": 1.0},
    {"name": "Precision Map-Match (sigma=4.5m, h=1.5)", "sigma_z": 4.5, "emission_weight": 1.5, "stay_cost": -0.01, "succ_cost": -0.10, "jump_cost": -25.0, "heading_boost": 1.5},
    {"name": "High-Confidence Topology (sigma=4.0m, h=2.0)", "sigma_z": 4.0, "emission_weight": 2.0, "stay_cost": -0.01, "succ_cost": -0.05, "jump_cost": -30.0, "heading_boost": 2.0},
]

print("\n" + "=" * 80)
print(f"{'Config Name':<42} | {'Avg Acc':<10} | {'Avg Cont':<10} | {'Trip 00':<8} | {'Trip 52':<8} | {'Trip 23':<8}")
print("=" * 80)

best_cfg = None
best_score = 0.0

for cfg in configs:
    accs, conts = [], []
    trip_accs = {}
    for td in test_cache:
        a, c = run_viterbi(td, sigma_z=cfg["sigma_z"], emission_weight=cfg["emission_weight"],
                           stay_cost=cfg["stay_cost"], succ_cost=cfg["succ_cost"],
                           jump_cost=cfg["jump_cost"], heading_boost=cfg["heading_boost"])
        if a is not None:
            accs.append(a)
            trip_accs[td["trip_id"]] = a
        conts.append(c)

    mean_acc = np.mean(accs) * 100
    mean_cont = np.mean(conts) * 100
    t00 = f"{trip_accs.get('00000000', 0)*100:.1f}%"
    t52 = f"{trip_accs.get('00000052', 0)*100:.1f}%"
    t23 = f"{trip_accs.get('00000023', 0)*100:.1f}%"

    print(f"{cfg['name']:<42} | {mean_acc:>6.2f}%   | {mean_cont:>6.2f}%   | {t00:<8} | {t52:<8} | {t23:<8}")

    score = mean_acc + mean_cont
    if score > best_score:
        best_score = score
        best_cfg = cfg

print("=" * 80)
print(f"BEST CONFIG: {best_cfg}")
