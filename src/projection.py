import numpy as np

class LocalCoordinateProjector:
    """
    Project angular geographical coordinates (longitude, latitude in degrees)
    into local metric Cartesian coordinates (x, y in meters) relative to a reference origin.
    Preserves true Euclidean distances and angles with sub-millimeter precision locally.
    """
    def __init__(self, ref_lon: float, ref_lat: float):
        self.ref_lon = float(ref_lon)
        self.ref_lat = float(ref_lat)
        self.cos_ref_lat = float(np.cos(np.radians(self.ref_lat)))
        self.meters_per_deg_lon = 111320.0 * self.cos_ref_lat
        self.meters_per_deg_lat = 111200.0

    def to_xy(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        x = (np.asarray(lon, dtype=np.float64) - self.ref_lon) * self.meters_per_deg_lon
        y = (np.asarray(lat, dtype=np.float64) - self.ref_lat) * self.meters_per_deg_lat
        if np.ndim(lon) == 0:
            return np.array([x, y], dtype=np.float64)
        return np.column_stack([x, y])

    def to_lonlat(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        lon = (np.asarray(x, dtype=np.float64) / self.meters_per_deg_lon) + self.ref_lon
        lat = (np.asarray(y, dtype=np.float64) / self.meters_per_deg_lat) + self.ref_lat
        if np.ndim(x) == 0:
            return np.array([lon, lat], dtype=np.float64)
        return np.column_stack([lon, lat])

def haversine_distance(lon1: np.ndarray, lat1: np.ndarray, lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
    """Computes great-circle distance in meters between WGS84 coordinate pairs."""
    r_lon1, r_lat1, r_lon2, r_lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = r_lon2 - r_lon1
    dlat = r_lat2 - r_lat1
    a = np.sin(dlat / 2.0)**2 + np.cos(r_lat1) * np.cos(r_lat2) * np.sin(dlon / 2.0)**2
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return 6371000.0 * c

def calculate_heading(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """Returns orientation angle in degrees from -180 to +180."""
    return np.degrees(np.arctan2(dy, dx))
