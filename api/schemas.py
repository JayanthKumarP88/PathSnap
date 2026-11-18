from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict

class EvaluateRequest(BaseModel):
    trip_id: str = Field(..., description="ID of the evaluation trip to process (e.g. '00000015')")
    partition: str = Field("evaluator_test", description="Dataset partition ('evaluator_test' or 'student_test')")
    use_hmm: bool = Field(True, description="Whether to apply HMM Viterbi route reconstruction")
    search_radius: float = Field(35.0, description="Candidate search radius in meters")

class EvaluateResponse(BaseModel):
    trip_id: str
    observations_count: int
    use_hmm: bool
    top1_accuracy: Any # float or "N/A"
    route_continuity: float
    candidate_miss_rate: float
    inference_time_sec: float
    reconstructed_arcs_count: int
    reconstructed_arcs: List[int]
    raw_classifier_arcs: List[int]
    server_timestamp: str
    predicted_coords: Optional[List[List[float]]] = None
    reference_coords: Optional[List[List[float]]] = None
    track_coords: Optional[List[List[float]]] = None

class TripListResponse(BaseModel):
    available_trips: List[str]
    count: int
    active_model: str
    metadata: Optional[Dict[str, Any]] = None
