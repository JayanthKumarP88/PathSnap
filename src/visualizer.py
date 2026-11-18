import os
import folium
import numpy as np
from typing import List, Optional, Tuple, Dict, Any

class RouteMapVisualizer:
    """
    Creates interactive multi-layer Folium maps strictly satisfying
    hackathon visualization guidelines:
    - Original GPS observations (Blue markers)
    - Road network geometry (Gray line segments)
    - Ground-truth reference route (Green path)
    - Predicted reconstructed route (Orange/Red dashed path)
    - Layer controls and informative custom legend
    """
    def __init__(self, tiles: str = "CartoDB positron"):
        self.tiles = tiles

    def build_map(self, track_lonlat: np.ndarray,
                  nodes_lonlat: np.ndarray,
                  arcs: np.ndarray,
                  network_arc_ids: List[int],
                  predicted_arc_ids: np.ndarray,
                  reference_arc_ids: Optional[np.ndarray] = None,
                  trip_title: str = "PathSnap Map-Matching Reconstruction") -> folium.Map:

        center_lat = float(np.mean(track_lonlat[:, 1]))
        center_lon = float(np.mean(track_lonlat[:, 0]))

        fmap = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles=self.tiles)

        # Feature Groups for Layer Control
        fg_network = folium.FeatureGroup(name="Road Network Segments", show=True)
        fg_gps = folium.FeatureGroup(name="Original GPS Observations", show=True)
        fg_ref = folium.FeatureGroup(name="Ground-Truth Route (Green)", show=True)
        fg_pred = folium.FeatureGroup(name="Predicted Reconstructed Route (Orange)", show=True)

        # 1. Road Network Layer (subsampled or bbox arcs)
        # Limit to 5000 arcs maximum for high web rendering performance
        display_arcs = network_arc_ids[:5000] if len(network_arc_ids) > 5000 else network_arc_ids
        for arc_id in display_arcs:
            u, v = int(arcs[arc_id, 0]), int(arcs[arc_id, 1])
            u_pt = [float(nodes_lonlat[u, 1]), float(nodes_lonlat[u, 0])]
            v_pt = [float(nodes_lonlat[v, 1]), float(nodes_lonlat[v, 0])]
            folium.PolyLine(locations=[u_pt, v_pt], color="#78909C", weight=2, opacity=0.5,
                            tooltip=f"Arc #{arc_id}").add_to(fg_network)

        # 2. Ground-Truth Reference Route
        if reference_arc_ids is not None and len(reference_arc_ids) > 0:
            ref_pts = []
            for arc_id in reference_arc_ids:
                u, v = int(arcs[arc_id, 0]), int(arcs[arc_id, 1])
                ref_pts.append([float(nodes_lonlat[u, 1]), float(nodes_lonlat[u, 0])])
                ref_pts.append([float(nodes_lonlat[v, 1]), float(nodes_lonlat[v, 0])])
            folium.PolyLine(locations=ref_pts, color="#2E7D32", weight=6, opacity=0.85,
                            tooltip="Ground-Truth Route").add_to(fg_ref)

        # 3. Predicted Route Layer
        if len(predicted_arc_ids) > 0:
            pred_pts = []
            for arc_id in predicted_arc_ids:
                u, v = int(arcs[arc_id, 0]), int(arcs[arc_id, 1])
                pred_pts.append([float(nodes_lonlat[u, 1]), float(nodes_lonlat[u, 0])])
                pred_pts.append([float(nodes_lonlat[v, 1]), float(nodes_lonlat[v, 0])])
            folium.PolyLine(locations=pred_pts, color="#E65100", weight=4, opacity=0.9,
                            dash_array="8, 6", tooltip="Predicted Route (PathSnap)").add_to(fg_pred)

        # 4. GPS Observation Points Layer
        step = max(1, len(track_lonlat) // 300) # Keep responsive
        for idx in range(0, len(track_lonlat), step):
            lon, lat = float(track_lonlat[idx, 0]), float(track_lonlat[idx, 1])
            folium.CircleMarker(
                location=[lat, lon],
                radius=3.5,
                color="#0D47A1",
                fill=True,
                fill_color="#1E88E5",
                fill_opacity=0.8,
                tooltip=f"GPS Ping #{idx} (Lon: {lon:.5f}, Lat: {lat:.5f})"
            ).add_to(fg_gps)

        # Add layers to map
        fg_network.add_to(fmap)
        fg_ref.add_to(fmap)
        fg_pred.add_to(fmap)
        fg_gps.add_to(fmap)
        folium.LayerControl(position="topright", collapsed=False).add_to(fmap)

        # Floating HTML Legend
        legend_html = f"""
        <div style="position: fixed;
                    bottom: 25px; left: 25px; width: 280px; height: 160px;
                    background-color: white; border:2px solid #ccc; z-index:9999;
                    font-size:12px; font-family: sans-serif; padding: 12px;
                    border-radius: 8px; box-shadow: 2px 2px 6px rgba(0,0,0,0.3);">
            <b style="font-size:13px;">{trip_title}</b><br><br>
            <span style="color:#0D47A1;">&#9679;</span> Raw GPS Observations<br>
            <span style="color:#78909C; font-weight:bold;">&mdash;</span> Road Network Arcs<br>
            <span style="color:#2E7D32; font-weight:bold;">&horbar;</span> Ground Truth Route<br>
            <span style="color:#E65100; font-weight:bold;">- - -</span> Reconstructed Route (PathSnap)
        </div>
        """
        fmap.get_root().html.add_child(folium.Element(legend_html))
        return fmap

    def save_map(self, fmap: folium.Map, output_path: str):
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fmap.save(output_path)
        print(f"Interactive Folium map saved to: {output_path}")
