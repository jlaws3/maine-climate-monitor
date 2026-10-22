
from pathlib import Path
from datetime import datetime
import json
import zipfile
import urllib.request

import numpy as np
import pandas as pd
import xarray as xr

from shapely.geometry import shape, mapping, box
from shapely.ops import unary_union
from pyproj import Geod


# ---------------------------------------
# SETTINGS
# ---------------------------------------

# This script lives in .github/workflows/.
# Data files belong in the repository's
# top-level data/ folder.
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)

BASELINE_FILE = DATA / "baseline.nc"
TARGET_FILE = DATA / "target.nc"

TARGET_DATE = "2025-10-15"
BASELINE_START = 1961
BASELINE_END = 1990

COUNTIES_URL = (
    "https://raw.githubusercontent.com/plotly/datasets/"
    "master/geojson-counties-fips.json"
)

GEOD = Geod(ellps="WGS84")


# ---------------------------------------
# FILE HANDLING
# ---------------------------------------

def open_climate_file(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing climate file: {path}"
        )

    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            names = [
                n for n in archive.namelist()
                if n.lower().endswith((".nc", ".nc4"))
            ]

            if len(names) != 1:
                raise ValueError(
                    f"Expected one NetCDF in {path}, "
                    f"found {len(names)}"
                )

            extracted = DATA / (
                path.stem + "_extracted.nc"
            )
            extracted.write_bytes(
                archive.read(names[0])
            )
            path = extracted

    return xr.open_dataset(
        path,
        engine="netcdf4"
    )


def temperature_array(path):
    ds = open_climate_file(path)

    candidates = [
        "t2m",
        "2m_temperature",
        "temperature_2m"
    ]

    name = next(
        (n for n in candidates if n in ds.data_vars),
        None
    )

    if name is None:
        raise ValueError(
            f"Temperature variable not found in {path}. "
            f"Variables: {list(ds.data_vars)}"
        )

    arr = ds[name]

    rename = {}

    if "lat" in arr.dims:
        rename["lat"] = "latitude"

    if "lon" in arr.dims:
        rename["lon"] = "longitude"

    arr = arr.rename(rename)

    if not {"latitude", "longitude"}.issubset(
        arr.dims
    ):
        raise ValueError(
            f"Missing spatial dimensions in {path}"
        )

    # The expected ERA5-Land product stores
    # absolute 2m temperatures in kelvin.
    units = str(
        arr.attrs.get("units", "")
    ).strip().lower()

    if units not in ("k", "kelvin"):
        raise ValueError(
            f"Expected kelvin units in {path}, "
            f"found {units!r}"
        )

    return arr


# ---------------------------------------
# DATE VALIDATION
# ---------------------------------------

def time_dimension(arr):
    dims = [
        d for d in arr.dims
        if d not in ("latitude", "longitude")
    ]

    if len(dims) != 1:
        raise ValueError(
            "Expected exactly one time dimension; "
            f"found {dims}"
        )

    return dims[0]


def dates_for(arr, dim):
    values = arr[dim].values

    try:
        dates = pd.to_datetime(values)
    except Exception as exc:
        raise ValueError(
            "Could not interpret climate dates"
        ) from exc

    if dates.isna().any():
        raise ValueError(
            "Climate dataset contains invalid dates"
        )

    return [
        d.strftime("%Y-%m-%d")
        for d in dates
    ]


def validate_historical(arr):
    dim = time_dimension(arr)
    actual = dates_for(arr, dim)

    month_day = TARGET_DATE[5:]

    expected = [
        f"{year}-{month_day}"
        for year in range(
            BASELINE_START,
            BASELINE_END + 1
        )
    ]

    if sorted(actual) != expected:
        missing = sorted(
            set(expected) - set(actual)
        )
        extra = sorted(
            set(actual) - set(expected)
        )

        raise ValueError(
            "Historical baseline must contain "
            "exactly 30 daily observations. "
            f"Missing: {missing}. "
            f"Unexpected: {extra}. "
            f"Actual count: {len(actual)}"
        )

    return arr.mean(
        dim=dim,
        skipna=False
    )


def validate_target(arr):
    dim = time_dimension(arr)
    dates = dates_for(arr, dim)

    if dates != [TARGET_DATE]:
        raise ValueError(
            f"Expected only {TARGET_DATE}; "
            f"found {dates}"
        )

    return arr.isel({dim: 0})


# ---------------------------------------
# AREA AND STATISTICS
# ---------------------------------------

def area_m2(geometry):
    if geometry.is_empty:
        return 0.0

    area, _ = GEOD.geometry_area_perimeter(
        geometry
    )

    return abs(float(area))


def weighted_mean(records, index):
    total_area = sum(
        row[3] for row in records
    )

    if total_area <= 0:
        return None

    return sum(
        row[index] * row[3]
        for row in records
    ) / total_area


def make_stats(records):
    if not records:
        return None

    return {
        "anomaly_f": round(
            weighted_mean(records, 0), 2
        ),
        "temperature_f": round(
            weighted_mean(records, 1), 2
        ),
        "normal_f": round(
            weighted_mean(records, 2), 2
        ),
        "grid_points": len(records),
        "area_km2": round(
            sum(r[3] for r in records) / 1e6,
            2
        )
    }


# ---------------------------------------
# LOAD CLIMATE DATA
# ---------------------------------------

print("Opening climate datasets...")

historical = temperature_array(BASELINE_FILE)
target = temperature_array(TARGET_FILE)

baseline = validate_historical(historical)
observed = validate_target(target)

# Verify identical spatial coordinates.
baseline, observed = xr.align(
    baseline,
    observed,
    join="exact"
)

# Load arrays into memory before looping.
baseline = baseline.load()
observed = observed.load()

# Sanity check for absolute temperatures.
for name, arr in [
    ("Baseline", baseline),
    ("Target", observed)
]:
    values = arr.values
    valid = values[np.isfinite(values)]

    if not valid.size:
        raise ValueError(
            f"{name} contains no valid temperatures"
        )

    if valid.min() < 180 or valid.max() > 340:
        raise ValueError(
            f"{name} has unexpected kelvin values: "
            f"{valid.min():.2f} to {valid.max():.2f}"
        )

anomaly = (observed - baseline) * 1.8

observed_f = (
    (observed - 273.15) * 1.8 + 32
)

baseline_f = (
    (baseline - 273.15) * 1.8 + 32
)

print("Baseline dates validated: 30")
print("Target date validated:", TARGET_DATE)


# ---------------------------------------
# LOAD MAINE COUNTY GEOGRAPHY
# ---------------------------------------

print("Loading Maine counties...")

with urllib.request.urlopen(
    COUNTIES_URL,
    timeout=60
) as response:
    us_counties = json.load(response)

maine_counties = [
    f for f in us_counties["features"]
    if str(f.get("id", "")).zfill(5)
    .startswith("23")
]

if len(maine_counties) != 16:
    raise ValueError(
        "Expected 16 Maine counties, "
        f"found {len(maine_counties)}"
    )

counties = {}

for feature in maine_counties:
    fips = str(feature["id"]).zfill(5)
    counties[fips] = shape(
        feature["geometry"]
    )

maine_land = unary_union(
    list(counties.values())
)


# ---------------------------------------
# BUILD CLIMATE GRID
# ---------------------------------------

latitudes = np.asarray(
    anomaly.latitude.values,
    dtype=float
)

longitudes = np.asarray(
    anomaly.longitude.values,
    dtype=float
)

if len(latitudes) < 2 or len(longitudes) < 2:
    raise ValueError(
        "Insufficient climate grid coordinates"
    )

lat_step = float(
    np.median(np.abs(np.diff(latitudes)))
)

lon_step = float(
    np.median(np.abs(np.diff(longitudes)))
)

if lat_step <= 0 or lon_step <= 0:
    raise ValueError("Invalid grid spacing")

features = []

county_records = {
    fips: [] for fips in counties
}

state_records = []

# Access array values by index rather than
# repeatedly using coordinate selection.
a_values = anomaly.values
t_values = observed_f.values
n_values = baseline_f.values

for i, lat in enumerate(latitudes):
    for j, lon in enumerate(longitudes):

        lat = float(lat)
        lon = float(lon)

        a = float(a_values[i, j])
        t = float(t_values[i, j])
        n = float(n_values[i, j])

        if not all(
            np.isfinite(v)
            for v in (a, t, n)
        ):
            continue

        # Build the original grid-cell
        # footprint, even if its center is
        # outside Maine.
        cell = box(
            lon - lon_step / 2,
            lat - lat_step / 2,
            lon + lon_step / 2,
            lat + lat_step / 2
        )

        clipped = cell.intersection(
            maine_land
        )

        if clipped.is_empty:
            continue

        land_area = area_m2(clipped)

        if land_area <= 0:
            continue

        # Store the original grid center.
        # These coordinates are NOT inferred
        # from the clipped polygon.
        features.append({
            "type": "Feature",
            "geometry": mapping(clipped),
            "properties": {
                "anomaly_f": round(a, 2),
                "temperature_f": round(t, 2),
                "normal_f": round(n, 2),
                "center_lon": round(lon, 6),
                "center_lat": round(lat, 6),
                "land_area_km2": round(
                    land_area / 1e6, 4
                )
            }
        })

        state_records.append(
            (a, t, n, land_area)
        )

        # Weight each grid value by the
        # area overlapping each county.
        for fips, county in counties.items():

            overlap = cell.intersection(
                county
            )

            if overlap.is_empty:
                continue

            overlap_area = area_m2(overlap)

            if overlap_area <= 0:
                continue

            county_records[fips].append(
                (a, t, n, overlap_area)
            )


if not features:
    raise RuntimeError(
        "No valid Maine climate grid cells"
    )


# ---------------------------------------
# WRITE GEOJSON
# ---------------------------------------

geojson = {
    "type": "FeatureCollection",
    "features": features
}

(DATA / "anomalies.geojson").write_text(
    json.dumps(
        geojson,
        separators=(",", ":")
    ),
    encoding="utf-8"
)


# ---------------------------------------
# WRITE SUMMARY
# ---------------------------------------

county_stats = {}

for fips, records in county_records.items():
    stats = make_stats(records)

    if stats is None:
        raise ValueError(
            f"No climate coverage for county {fips}"
        )

    county_stats[fips] = stats

state_stats = make_stats(state_records)

if len(county_stats) != 16:
    raise ValueError(
        "Missing county statistics"
    )

summary = {
    "date": TARGET_DATE,
    "baseline": "1961-1990",
    "baseline_method": (
        "Mean of October 15 daily "
        "temperatures for 1961-1990"
    ),
    "source": (
        "Copernicus ERA5-Land "
        "daily statistics"
    ),
    "measurement": (
        "Daily mean 2m temperature"
    ),
    "units": "Fahrenheit",
    "county_statistics": county_stats,
    "state_statistics": state_stats,
    "grid_cell_count": len(features),
    "weighting": (
        "Geodesic land-area overlap "
        "of ERA5-Land grid cells with "
        "Maine county polygons"
    ),
    "methodology_note": (
        "Grid values are weighted by "
        "their land-area overlap with "
        "each county. Interpolation "
        "is for visualization only. "
        "ERA5-Land is reanalysis data."
    )
}

(DATA / "summary.json").write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8"
)

print("Climate processing complete.")
print("Grid cells:", len(features))
print("Counties:", len(county_stats))
print(
    "Statewide anomaly:",
    state_stats["anomaly_f"],
    "degrees F"
)
print("Created data/anomalies.geojson")
print("Created data/summary.json")
