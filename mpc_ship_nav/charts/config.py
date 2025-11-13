from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class RegionConfig:
    # Geographic bounding box (lat/lon)
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

    # Grid resolution in meters (for planner)
    grid_resolution: float = 250.0

    # Path to coastline / chart data (e.g., GSHHG or preprocessed shapes)
    coastline_file: Optional[str] = None

    # Safety buffer (meters) around land -> treated as non-navigable
    coastal_buffer: float = 500.0
