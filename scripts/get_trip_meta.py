import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_loader import TripDataLoader
import numpy as np
import json

loader = TripDataLoader()
meta = {}
for part in ['evaluator_test', 'student_test']:
    meta[part] = {}
    trips = loader.list_trips(part)
    for tid in trips:
        d = loader.load_trip(tid, part)
        track = d['track']
        t_sec = float(track[-1, 2] - track[0, 2])
        mins = max(1, int(round(t_sec / 60)))
        lat = np.radians(track[:, 1])
        lon = np.radians(track[:, 0])
        dlat = np.diff(lat)
        dlon = np.diff(lon)
        a = np.sin(dlat/2.0)**2 + np.cos(lat[:-1])*np.cos(lat[1:])*np.sin(dlon/2.0)**2
        dist_km = float(np.sum(6371.0 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))))
        meta[part][tid] = {
            "id": tid,
            "dist": f"{dist_km:.1f} km",
            "fixes": f"{len(track):,} fixes",
            "duration": f"{mins} min",
            "pings": len(track)
        }

print(json.dumps(meta, indent=2))
with open('artifacts/trips_metadata.json', 'w') as f:
    json.dump(meta, f, indent=2)
