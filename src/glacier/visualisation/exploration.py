from __future__ import annotations
import numpy as np
import plotly.graph_objects as go
import geopandas as gpd

REGIONS = {
    "Alpes":     {"lon": (4, 16),    "lat": (43, 49)},
    "Himalaya":  {"lon": (75, 105),  "lat": (25, 37)},
    "Alaska":    {"lon": (-170, -130),"lat": (52, 72)},
    "Patagonie": {"lon": (-76, -65), "lat": (-56, -40)},
    "Islande":   {"lon": (-25, -12), "lat": (63, 67)},
}

def make_geojson_with_ids(gdf_sub: gpd.GeoDataFrame) -> dict:
    gj = gdf_sub.__geo_interface__
    for i, feat in enumerate(gj["features"]):
        feat["id"] = str(i)
    return gj

def subset_region(gdf: gpd.GeoDataFrame, bbox: dict, max_polys: int = 2000, seed: int = 0) -> gpd.GeoDataFrame:
    lon0, lon1 = bbox["lon"]
    lat0, lat1 = bbox["lat"]
    sub = gdf.cx[lon0:lon1, lat0:lat1]
    if len(sub) > max_polys:
        sub = sub.sample(max_polys, random_state=seed)
    return sub

def bbox_to_zoom(lon_min: float, lat_min: float, lon_max: float, lat_max: float) -> float:
    lon_span = max(1e-6, lon_max - lon_min)
    lat_span = max(1e-6, lat_max - lat_min)
    span = max(lon_span, lat_span)
    z = 8 - (np.log(span) / np.log(2))
    return float(np.clip(z, 1.5, 10.5))

def compute_view(sub: gpd.GeoDataFrame, pad: float = 0.05) -> dict:
    lon_min, lat_min, lon_max, lat_max = sub.total_bounds
    pad_x = (lon_max - lon_min) * pad if lon_max > lon_min else 0.5
    pad_y = (lat_max - lat_min) * pad if lat_max > lat_min else 0.5
    lon_min, lon_max = lon_min - pad_x, lon_max + pad_x
    lat_min, lat_max = lat_min - pad_y, lat_max + pad_y
    center = {"lon": (lon_min + lon_max) / 2, "lat": (lat_min + lat_max) / 2}
    zoom = bbox_to_zoom(lon_min, lat_min, lon_max, lat_max)
    return {"center": center, "zoom": zoom}

def build_glacier_mapbox_dropdown(
    gdf: gpd.GeoDataFrame,
    regions: dict,
    start: str = "Alpes",
    max_polys: int = 5000,
    basemap: str = "carto-positron",
) -> go.Figure:
    region_names = list(regions.keys())
    if start not in region_names:
        start = region_names[0]

    traces = []
    views = {}

    for name in region_names:
        sub = subset_region(gdf, regions[name], max_polys=max_polys)
        if len(sub) == 0:
            # skip régions vides
            continue

        views[name] = compute_view(sub)
        gj = make_geojson_with_ids(sub)
        locs = [str(i) for i in range(len(gj["features"]))]

        traces.append(
            go.Choroplethmapbox(
                geojson=gj,
                locations=locs,
                featureidkey="id",
                z=[1] * len(locs),
                showscale=False,
                visible=False,
                marker_line_width=0.2,
                colorscale=[[0, "rgba(0,120,255,0.45)"], [1, "rgba(0,120,255,0.45)"]],
                zmin=0, zmax=1,
                name=name,
            )
        )

    # activer trace start
    idx = region_names.index(start)
    traces[idx].visible = True
    v0 = views[start]

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"Zoom: {start}",
        margin=dict(l=0, r=0, t=40, b=0),
        mapbox=dict(style=basemap, center=v0["center"], zoom=v0["zoom"]),
        dragmode="pan",
        uirevision=f"region:{start}",
    )

    buttons = []
    for i, name in enumerate(region_names):
        visible = [False] * len(region_names)
        visible[i] = True
        v = views[name]

        buttons.append(
            dict(
                label=name,
                method="update",
                args=[
                    {"visible": visible},
                    {
                        "title": f"Zoom: {name}",
                        "uirevision": f"region:{name}",
                        "mapbox.center": v["center"],
                        "mapbox.zoom": v["zoom"],
                        "mapbox.style": basemap,
                    },
                ],
            )
        )

    fig.update_layout(
        updatemenus=[dict(buttons=buttons, direction="down", x=0.01, y=0.99)]
    )
    return fig