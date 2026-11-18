import os
import sys
import time
import datetime
import pickle
import numpy as np
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from api.schemas import EvaluateRequest, EvaluateResponse, TripListResponse
from src.data_loader import TripDataLoader
from src.evaluator import TripEvaluator
from src.visualizer import RouteMapVisualizer

app = FastAPI(
    title="PathSnap Distributed Map-Matching API",
    description="Backend API for real-time GPS map-matching and sequential route reconstruction.",
    version="1.0.0"
)

# Enable CORS for cross-laptop communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global inference engine and model cache
data_loader = TripDataLoader()
evaluator = TripEvaluator(data_loader)
active_model = None
model_name = "HistGradientBoosting"

def load_default_model():
    global active_model, model_name
    models_dir = os.path.join(project_root, "artifacts", "saved_models")
    # Prefer tuned model, fallback to HistGradientBoosting or RandomForest
    candidate_files = [
        "HistGradientBoosting_tuned.pkl",
        "HistGradientBoosting.pkl",
        "RandomForest.pkl",
        "LogisticRegression.pkl"
    ]
    for mf in candidate_files:
        p = os.path.join(models_dir, mf)
        if os.path.exists(p):
            with open(p, "rb") as f:
                active_model = pickle.load(f)
            model_name = mf.replace(".pkl", "")
            print(f"[API SERVER] Successfully loaded active model: {model_name} from {p}")
            return
    print("[API SERVER] Warning: No saved model found in artifacts/saved_models. Models will be loaded on first request.")

@app.on_event("startup")
def startup_event():
    load_default_model()
    print("=" * 65)
    print("PATHSNAP FASTAPI INFERENCE SERVER RUNNING (LAPTOP 1)")
    print(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Ready to process evaluator inference requests on /evaluate")
    print("=" * 65)

@app.get("/", tags=["Health"])
def health_check():
    return {
        "status": "online",
        "system": "PathSnap Distributed Map-Matching Engine",
        "active_model": model_name,
        "endpoints": ["/trips", "/evaluate", "/map/{trip_id}"]
    }

@app.get("/trips", response_model=TripListResponse, tags=["Trips"])
def list_available_trips(partition: str = "student_test"):
    trips = data_loader.list_trips(partition)
    return {
        "available_trips": trips,
        "count": len(trips),
        "active_model": model_name
    }

@app.post("/evaluate", response_model=EvaluateResponse, tags=["Evaluation"])
def evaluate_trip_endpoint(req: EvaluateRequest, request: Request):
    global active_model
    client_ip = request.client.host if request.client else "unknown"
    req_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    print("\n" + "-" * 60)
    print(f"[INFERENCE REQUEST] Received request for Trip ID: {req.trip_id}")
    print(f"  Client IP: {client_ip} | Timestamp: {req_time}")
    print(f"  Use HMM Decoding: {req.use_hmm} | Search Radius: {req.search_radius}m")

    if active_model is None:
        load_default_model()
        if active_model is None:
            raise HTTPException(status_code=500, detail="Inference model is not yet loaded on server.")

    try:
        t0 = time.time()
        result = evaluator.evaluate_trip(
            trip_id=req.trip_id,
            model=active_model,
            partition="student_test",
            use_hmm=req.use_hmm,
            search_radius=req.search_radius
        )
        total_latency = time.time() - t0

        print(f"[PIPELINE COMPLETE] Processed {result['observations_count']} GPS observations in {total_latency:.2f}s")
        print(f"  Top-1 Observation Accuracy: {result['top1_accuracy']}")
        print(f"  Route Continuity:           {result['route_continuity'] * 100:.2f}%")
        print(f"  Candidate Miss Rate:        {result['candidate_miss_rate'] * 100:.2f}%")
        print("-" * 60 + "\n")

        return {
            "trip_id": result["trip_id"],
            "observations_count": result["observations_count"],
            "use_hmm": result["use_hmm"],
            "top1_accuracy": result["top1_accuracy"],
            "route_continuity": result["route_continuity"],
            "candidate_miss_rate": result["candidate_miss_rate"],
            "inference_time_sec": result["inference_time_sec"],
            "reconstructed_arcs_count": len(result["reconstructed_arcs"]),
            "reconstructed_arcs": result["reconstructed_arcs"],
            "raw_classifier_arcs": result["raw_classifier_arcs"],
            "server_timestamp": req_time
        }
    except Exception as e:
        print(f"[ERROR] Inference failed for trip {req.trip_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/map/{trip_id}", response_class=HTMLResponse, tags=["Visualization"])
def get_trip_map_endpoint(trip_id: str, use_hmm: bool = True):
    if active_model is None:
        load_default_model()

    result = evaluator.evaluate_trip(
        trip_id=trip_id,
        model=active_model,
        partition="student_test",
        use_hmm=use_hmm
    )

    visualizer = RouteMapVisualizer()
    fmap = visualizer.build_map(
        track_lonlat=np.array(result["track_lonlat"]),
        nodes_lonlat=np.array(result["nodes_lonlat"]),
        arcs=np.array(result["arcs"]),
        network_arc_ids=result["bbox_arc_ids"],
        predicted_arc_ids=np.array(result["reconstructed_arcs"]),
        reference_arc_ids=np.array(result["reference_arcs"]) if result["reference_arcs"] else None,
        trip_title=f"PathSnap Reconstruction - Trip {trip_id} ({'HMM Decoded' if use_hmm else 'Classifier Only'})"
    )
    return HTMLResponse(content=fmap.get_root().render(), status_code=200)

if __name__ == "__main__":
    import uvicorn
    # Bind to 0.0.0.0 to enable access from Laptop 2 across the LAN
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
