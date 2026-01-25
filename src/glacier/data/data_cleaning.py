from __future__ import annotations

import pandas as pd
import geopandas as gpd


def keep_outlines(gdf: gpd.GeoDataFrame, value: str = "glac_bound") -> gpd.GeoDataFrame:
    if "line_type" not in gdf.columns:
        return gdf
    return gdf[gdf["line_type"] == value]


def drop_empty_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf[gdf.geometry.notnull()]
    return gdf[~gdf.geometry.is_empty]


def parse_anlys_time(gdf: gpd.GeoDataFrame, col: str = "anlys_time") -> gpd.GeoDataFrame:
    if col not in gdf.columns:
        return gdf
    gdf[col] = pd.to_datetime(gdf[col], errors="coerce", utc=True)
    return gdf[gdf[col].notna()]


def ensure_crs(gdf: gpd.GeoDataFrame, epsg: int = 4326) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        return gdf.set_crs(epsg)
    return gdf


def fix_invalid_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # GeoPandas >= 0.12 has make_valid; fallback: buffer(0)
    try:
        gdf["geometry"] = gdf.geometry.make_valid()
    except Exception:
        gdf["geometry"] = gdf.geometry.buffer(0)
    return drop_empty_geometries(gdf)


def explode_multipolygons(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return gdf.explode(index_parts=False)


def filter_positive_area(gdf: gpd.GeoDataFrame, col: str = "area") -> gpd.GeoDataFrame:
    if col not in gdf.columns:
        return gdf
    gdf = gdf[gdf[col].notna()]
    return gdf[gdf[col] > 0]


def cast_categories(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    cols = ["primeclass", "surge_type", "term_type", "gtnq1reg", "gtnq2reg", "rgi_gl_typ", "conn_lvl"]
    for c in cols:
        if c in gdf.columns:
            gdf[c] = gdf[c].astype("category")
    return gdf


def drop_exact_dupes(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # Exact duplicate: same glac_id, same anlys_time, same geometry
    if "glac_id" not in gdf.columns:
        return gdf

    gdf = gdf.copy()
    gdf["_geom_wkb"] = gdf.geometry.to_wkb()

    subset = ["glac_id"] + (["anlys_time"] if "anlys_time" in gdf.columns else []) + ["_geom_wkb"]
    gdf = gdf.drop_duplicates(subset=subset)

    return gdf.drop(columns=["_geom_wkb"])


def clean_glims_outlines(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Cleans GLIMS glacier outline polygons for analysis/modeling.
    - keeps glac_bound
    - removes empty geometries
    - parses anlys_time
    - ensures CRS
    - fixes invalid geometries
    - explodes multipolygons
    - filters positive area (if 'area' exists)
    - casts categorical columns
    - drops exact duplicates
    """
    gdf = gdf.copy()

    gdf = keep_outlines(gdf)
    gdf = drop_empty_geometries(gdf)
    gdf = parse_anlys_time(gdf)
    gdf = ensure_crs(gdf, epsg=4326)
    gdf = fix_invalid_geometries(gdf)
    gdf = explode_multipolygons(gdf)
    gdf = filter_positive_area(gdf, col="area")
    gdf = cast_categories(gdf)
    gdf = drop_exact_dupes(gdf)

    return gdf.reset_index(drop=True)