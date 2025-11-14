from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional
import matplotlib.axes
import numpy as np
from shapely.geometry import Polygon, Point, LineString, shape, box
from shapely.ops import transform, unary_union
from pyproj import CRS, Transformer
import fiona


from .config import RegionConfig


@dataclass
class ChartEnvironment:
    """
    Encapsulates coastline data, local projection, and navigability queries.

    """

    region_cfg: RegionConfig
    crs_geo: CRS
    crs_local: CRS
    transformer_to_local: Transformer
    transformer_to_geo: Transformer

    # union of land polygons in local (x, y) coordinates
    land_geometry: Polygon

    def __init__(self, region_cfg: RegionConfig):
        self.region_cfg = region_cfg

        # 1) Define CRS: geographic (lat/lon) + local projected (e.g. UTM / local ENU)
        self.crs_geo = CRS.from_epsg(4326)  # WGS84
        lat_c = 0.5 * (region_cfg.lat_min + region_cfg.lat_max)
        lon_c = 0.5 * (region_cfg.lon_min + region_cfg.lon_max)

        # UTM zone: 1–60
        zone = int((lon_c + 180) // 6) + 1

        # Hemisphere
        is_north = lat_c >= 0

        # EPSG code (Northern: 326XX, Southern: 327XX)
        epsg_code = 32600 + zone if is_north else 32700 + zone

        self.crs_geo = CRS.from_epsg(4326)
        self.crs_local = CRS.from_epsg(epsg_code)

        self.transformer_to_local = Transformer.from_crs(
            self.crs_geo, self.crs_local, always_xy=True
        )
        self.transformer_to_geo = Transformer.from_crs(
            self.crs_local, self.crs_geo, always_xy=True
        )

        print(f"[ENV] Using UTM zone {zone}{'N' if is_north else 'S'} (EPSG:{epsg_code})")


        # 2) Load coastline polygons and transform to local coordinates
        self.land_geometry = self._load_and_prepare_land()

    def plot_base_map(self, ax: "matplotlib.axes.Axes") -> None:
        if self.land_geometry.is_empty:
            return
        geom = self.land_geometry
        for g in geom.geoms:
            xs, ys = g.exterior.xy
            ax.fill(xs, ys, alpha=0.3, color="grey")
    # ---------- coordinate transforms ----------

    def to_local(self, lat: float, lon: float) -> Tuple[float, float]:
        """
        (lat, lon) -> (x, y) in meters in local frame.
        """
        x, y = self.transformer_to_local.transform(lon, lat)
        return float(x), float(y)

    def to_geo(self, x: float, y: float) -> Tuple[float, float]:
        """
        (x, y) in local frame -> (lat, lon).
        """
        lon, lat = self.transformer_to_geo.transform(x, y)
        return float(lat), float(lon)

    # ---------- navigability queries ----------

    def is_navigable(self, x: float, y: float) -> bool:
        """
        True if (x, y) is in open sea (not land, outside coastal buffer).
        """
        pt = Point(x, y)
        # land_geometry is already buffered by coastal_buffer
        return not self.land_geometry.contains(pt)

    def distance_to_shore(self, x: float, y: float) -> float:
        """
        Signed distance to buffered shoreline:
          - Negative inside land/buffer,
          - Zero on boundary,
          - Positive in open water.
        """
        pt = Point(x, y)
        d = pt.distance(self.land_geometry.boundary)
        if self.land_geometry.contains(pt):
            return -float(d)
        return float(d)
    
    def line_intersects_land(self, x1, y1, x2, y2) -> bool:
        """
        True if the straight segment (x1, y1)→(x2, y2) intersects land geometry.
        """
        seg = LineString([(x1, y1), (x2, y2)])
        return seg.intersects(self.land_geometry)

    # ---------- internal helpers ----------

    def _geom_to_local(self, geom):
        """
        Transform a shapely geometry from geographic (lon, lat) to local (x, y).
        """
        def _proj(lon, lat):
            x, y = self.transformer_to_local.transform(lon, lat)
            return x, y

        return transform(_proj, geom)


    def _load_and_prepare_land(self) -> Polygon:
        """
        Load coastline/land polygons for the region and return a buffered
        land polygon in LOCAL (x, y) coordinates.

        If coastline_file is None, fall back to a synthetic toy island so that
        planners and simulators can still be tested.
        """

        # --- DEV MODE: toy island if no charts are provided ---
        if self.region_cfg.coastline_file is None:
            lat_c = 0.5 * (self.region_cfg.lat_min + self.region_cfg.lat_max)
            lon_c = 0.5 * (self.region_cfg.lon_min + self.region_cfg.lon_max)
            x_c, y_c = self.to_local(lat_c, lon_c)

            size = 10_000.0  # 10 km half-size
            square = Polygon([
                (x_c - size, y_c - size),
                (x_c + size, y_c - size),
                (x_c + size, y_c + size),
                (x_c - size, y_c + size),
            ])

            print("[ENV] Using toy island at:", (x_c, y_c))
            return square.buffer(self.region_cfg.coastal_buffer)

        # --- REAL MODE: read coastline polygons from file in WGS84 (lat/lon) ---
        coast_path = self.region_cfg.coastline_file
        print(f"[ENV] Loading coastline from: {coast_path}")

        lat_min, lat_max = self.region_cfg.lat_min, self.region_cfg.lat_max
        lon_min, lon_max = self.region_cfg.lon_min, self.region_cfg.lon_max

        # geographic bbox (lon/lat order because Fiona uses that)
        bbox_geo = box(lon_min, lat_min, lon_max, lat_max)

        land_polys_local = []

        with fiona.open(coast_path, "r") as src:
            for feat in src:
                geom = shape(feat["geometry"])  # shapely geometry in lon/lat
                if not geom.is_valid or geom.is_empty:
                    continue

                # Quickly skip features far outside region
                if not geom.intersects(bbox_geo):
                    continue

                # Clip to region bbox to keep polygons manageable
                geom_clipped = geom.intersection(bbox_geo)

                if geom_clipped.is_empty:
                    continue

                # Transform to local CRS
                geom_local = self._geom_to_local(geom_clipped)
                land_polys_local.append(geom_local)

        if not land_polys_local:
            print("[ENV] WARNING: No coastline polygons found in region. "
                  "Environment will be fully navigable water.")
            return Polygon()

        land_union = unary_union(land_polys_local)

        # Safety buffer around land
        land_buffered = land_union.buffer(self.region_cfg.coastal_buffer)

        print("[ENV] Land polygons loaded and buffered.")
        return land_buffered

