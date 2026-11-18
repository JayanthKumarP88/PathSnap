import os
import folium
import numpy as np
from typing import List, Optional, Tuple, Dict, Any

class RouteMapVisualizer:
    """
    Creates interactive multi-layer Folium maps strictly satisfying
    hackathon visualization guidelines:
    - Original GPS observations (Blue markers + dashed trace)
    - Road network geometry (Subtle gray lines)
    - Ground-truth reference route (Thick forest green path)
    - Predicted reconstructed route (Vibrant orange path with continuous road alignment)
    - Trip departure & destination pins (Green departure, Red destination)
    - Auto-bounding box fitting to trajectory extent
    - Layer controls and informative custom legend
    """
    def __init__(self, tiles: str = "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
                 attr: str = "Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ"):
        self.tiles = tiles
        self.attr = attr

    def build_map(self, track_lonlat: np.ndarray,
                  nodes_lonlat: np.ndarray,
                  arcs: np.ndarray,
                  network_arc_ids: List[int],
                  predicted_arc_ids: np.ndarray,
                  reference_arc_ids: Optional[np.ndarray] = None,
                  trip_title: str = "PathSnap Map-Matching Reconstruction") -> folium.Map:

        center_lat = float(np.mean(track_lonlat[:, 1]))
        center_lon = float(np.mean(track_lonlat[:, 0]))

        fmap = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles=self.tiles, attr=self.attr)

        # Feature Groups for Layer Control
        fg_network = folium.FeatureGroup(name="🛣️ Road Network Arcs", show=True)
        fg_gps = folium.FeatureGroup(name="📍 Raw GPS Observations", show=True)
        fg_ref = folium.FeatureGroup(name="🟩 Ground-Truth Route (Server-side)", show=True)
        fg_pred = folium.FeatureGroup(name="🟧 Predicted Reconstructed Route (PathSnap)", show=True)

        # 1. Road Network Layer (Arcs within bounding box)
        display_arcs = network_arc_ids[:6000] if len(network_arc_ids) > 6000 else network_arc_ids
        for arc_id in display_arcs:
            u, v = int(arcs[arc_id, 0]), int(arcs[arc_id, 1])
            u_pt = [float(nodes_lonlat[u, 1]), float(nodes_lonlat[u, 0])]
            v_pt = [float(nodes_lonlat[v, 1]), float(nodes_lonlat[v, 0])]
            folium.PolyLine(locations=[u_pt, v_pt], color="#90A4AE", weight=1.8, opacity=0.45,
                            tooltip=f"Road Arc #{arc_id}").add_to(fg_network)

        # 2. Ground-Truth Reference Route (Green)
        if reference_arc_ids is not None and len(reference_arc_ids) > 0:
            for seq_idx, arc_id in enumerate(reference_arc_ids):
                u, v = int(arcs[arc_id, 0]), int(arcs[arc_id, 1])
                u_pt = [float(nodes_lonlat[u, 1]), float(nodes_lonlat[u, 0])]
                v_pt = [float(nodes_lonlat[v, 1]), float(nodes_lonlat[v, 0])]
                folium.PolyLine(locations=[u_pt, v_pt], color="#1B5E20", weight=6, opacity=0.85,
                                tooltip=f"GT Step #{seq_idx + 1} (Arc #{arc_id})").add_to(fg_ref)

        # 3. Predicted Reconstructed Route (Vibrant Orange / Red-Orange)
        if len(predicted_arc_ids) > 0:
            # Deduplicate consecutive identical arcs to extract clean sequence
            traversed_arcs = []
            for arc_id in predicted_arc_ids:
                if not traversed_arcs or traversed_arcs[-1] != arc_id:
                    traversed_arcs.append(arc_id)

            for seq_idx, arc_id in enumerate(traversed_arcs):
                u, v = int(arcs[arc_id, 0]), int(arcs[arc_id, 1])
                u_pt = [float(nodes_lonlat[u, 1]), float(nodes_lonlat[u, 0])]
                v_pt = [float(nodes_lonlat[v, 1]), float(nodes_lonlat[v, 0])]
                folium.PolyLine(
                    locations=[u_pt, v_pt],
                    color="#FF6F00",
                    weight=4.5,
                    opacity=0.95,
                    dash_array="6, 4",
                    tooltip=f"Predicted Step #{seq_idx + 1} (Arc #{arc_id})"
                ).add_to(fg_pred)

        # 4. Raw GPS Trace & Observation Points Layer
        # Draw GPS trajectory line
        gps_line_pts = [[float(lat), float(lon)] for lon, lat in track_lonlat[:, :2]]
        folium.PolyLine(
            locations=gps_line_pts,
            color="#0288D1",
            weight=1.5,
            opacity=0.4,
            dash_array="4, 4",
            tooltip="Raw GPS Observation Trace"
        ).add_to(fg_gps)

        # Draw sampled GPS observation markers
        step = max(1, len(track_lonlat) // 250)
        for idx in range(0, len(track_lonlat), step):
            lon, lat = float(track_lonlat[idx, 0]), float(track_lonlat[idx, 1])
            folium.CircleMarker(
                location=[lat, lon],
                radius=3.5,
                color="#0D47A1",
                fill=True,
                fill_color="#29B6F6",
                fill_opacity=0.85,
                tooltip=f"GPS Observation #{idx} (Lon: {lon:.5f}, Lat: {lat:.5f})"
            ).add_to(fg_gps)

        # 5. Start and End Markers
        start_pt = [float(track_lonlat[0, 1]), float(track_lonlat[0, 0])]
        end_pt = [float(track_lonlat[-1, 1]), float(track_lonlat[-1, 0])]

        folium.Marker(
            location=start_pt,
            popup="<b>Departure Origin</b><br>Trip Start (t=0)",
            icon=folium.Icon(color="green", icon="play", prefix="fa")
        ).add_to(fg_gps)

        folium.Marker(
            location=end_pt,
            popup="<b>Arrival Destination</b><br>Trip End",
            icon=folium.Icon(color="red", icon="flag-checkered", prefix="fa")
        ).add_to(fg_gps)

        # Add layers to map
        fg_network.add_to(fmap)
        fg_ref.add_to(fmap)
        fg_pred.add_to(fmap)
        fg_gps.add_to(fmap)
        folium.LayerControl(position="topright", collapsed=False).add_to(fmap)

        # Auto-fit bounds to trajectory bounding box with padding
        min_lat, max_lat = float(track_lonlat[:, 1].min()), float(track_lonlat[:, 1].max())
        min_lon, max_lon = float(track_lonlat[:, 0].min()), float(track_lonlat[:, 0].max())
        fmap.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]], padding=[30, 30])

        # Floating HTML Legend
        has_gt = "Available" if (reference_arc_ids is not None and len(reference_arc_ids) > 0) else "Hidden (Test Partition)"
        legend_html = f"""
        <div style="position: fixed;
                    bottom: 25px; left: 25px; width: 290px;
                    background-color: rgba(255, 255, 255, 0.95);
                    border: 2px solid #90A4AE; z-index: 9999;
                    font-size: 11px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    padding: 12px 14px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
            <div style="font-size: 13px; font-weight: 700; color: #1A237E; margin-bottom: 8px; border-bottom: 1px solid #ECEFF1; padding-bottom: 4px;">
                🗺️ {trip_title}
            </div>
            <div style="margin-bottom: 4px;">
                <span style="color:#0288D1; font-size:14px;">&#9679;</span> <b>Raw GPS Pings</b> (Trace & Circles)
            </div>
            <div style="margin-bottom: 4px;">
                <span style="color:#90A4AE; font-weight:bold; font-size:14px;">&mdash;</span> <b>Road Network</b> (Topological Arcs)
            </div>
            <div style="margin-bottom: 4px;">
                <span style="color:#1B5E20; font-weight:bold; font-size:14px;">&horbar;</span> <b>Ground-Truth Route</b> ({has_gt})
            </div>
            <div style="margin-bottom: 4px;">
                <span style="color:#FF6F00; font-weight:bold; font-size:14px;">- - -</span> <b>PathSnap Snapped Route</b> (Predicted)
            </div>
            <div style="margin-top: 6px; font-size: 10px; color: #546E7A; border-top: 1px solid #ECEFF1; padding-top: 4px;">
                🟢 <b>Start</b>: Origin | 🔴 <b>End</b>: Destination
            </div>
        </div>
        """
        fmap.get_root().html.add_child(folium.Element(legend_html))
        return fmap

    def save_map(self, fmap: folium.Map, output_path: str):
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fmap.save(output_path)
        print(f"Interactive Folium map saved to: {output_path}")
