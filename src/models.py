import time
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def create_model_suite() -> Dict[str, Any]:
    """
    Factory function returning 5 distinct models across 5 algorithm families
    strictly adhering to the hackathon guidelines.
    """
    models = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, C=1.0, random_state=42, class_weight="balanced"))
        ]),
        "DistanceKNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=-1))
        ]),
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=100, max_depth=16, min_samples_leaf=5,
                                           random_state=42, n_jobs=-1, class_weight="balanced_subsample"))
        ]),
        "HistGradientBoosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=31, min_samples_leaf=20,
                                                   learning_rate=0.1, random_state=42, class_weight="balanced"))
        ]),
        "MLPClassifier": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu",
                                  max_iter=150, early_stopping=True, random_state=42))
        ])
    }
    return models

MODEL_FAMILIES = {
    "LogisticRegression": "Linear and Probabilistic",
    "DistanceKNN": "Instance-based",
    "RandomForest": "Ensemble",
    "HistGradientBoosting": "Boosting",
    "MLPClassifier": "Neural Networks"
}
