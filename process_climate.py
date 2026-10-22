
from pathlib import Path
import json
import zipfile
import urllib.request

import numpy as np
import xarray as xr

from shapely.geometry import shape, Point, mapping, box
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

BASELINE_FILE = DATA / "baseline.nc"
TARGET_FILE = DATA / "target.nc"

COUNTIES_URL = (
    "https://raw.githubusercontent.com/plotly/datasets/"
    "master/geojson-counties-fips.json"
)

def open_climate_file(path):
    """Open a NetCDF file, including ZIP-wrapped results."""
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            names = [
                name for name in archive.namelist()
                if name.endswith((".nc", ".nc4"))
            ]
            if not names:
                raise ValueError("No NetCDF file in ZIP")
            extracted = DATA / (path.stem + "_extracted.nc")
            extracted.write_bytes(archive.read(names[0]))
            path = extracted

    return xr.open_dataset(path, engine="netcdf4")

def temperature_array(path):
    ds = open_climate_file(path)

    candidates = ["t2m", "2m_temperature", "temperature_2m"]

    name = next(
        (n for n in candidates if n in ds.data_vars),
        None
    )

    if name is None:
        if len(ds.data_vars) != 1:
            raise ValueError(
                "Cannot identify temperature variable: "
                + str(list(ds.data_vars))
            )
        name = list(ds.data_vars)[0]

    arr = ds[name]

    rename = {}
    for old, new in [
        ("lat", "latitude"),
        ("lon", "longitude")
    ]:
        if old in arr.dims:
            rename[old] = new

    arr = arr.rename(rename)

    if not {"latitude", "longitude"}.issubset(arr.dims):
        raise ValueError("Missing latitude/longitude dimensions")

    return arr

print("Opening climate datasets...")

historical = temperature_array(BASELINE_FILE)
target = temperature_array(TARGET_FILE)

spatial = {"latitude", "longitude"}

historical_time = [
    d for d in historical.dims if d not in spatial
]
target_time = [
    d for d in target.dims if d not in spatial
]

if len(historical_time) != 1:
    raise ValueError(
        "Expected one historical time dimension: "
        + str(historical.dims)
    )

baseline = historical.mean(
    dim=historical_time[0],
    skipna=True
)

if len(target_time) == 1:
    observed = target.isel({target_time[0]: 0})
elif len(target_time) == 0:
    observed = target
else:
    raise ValueError("Unexpected target dimensions")

baseline, observed = xr.align(
    baseline, observed, join="exact"
)

# Temperature differences: 1 kelvin = 1.8 degrees F.
anomaly = (observed - baseline) * 1.8

# Absolute temperatures: Kelvin to Fahrenheit.
observed_f = (observed - 273.15) * 1.8 + 32
baseline_f = (baseline - 273.15) * 1.8 + 32

print("Loading Maine county boundaries...")

with urllib.request.urlopen(COUNTIES_URL) as response:
    us_counties = json.load(response)

maine_counties = [
    f for f in us_counties["features"]
    if str(f.get("id", "")).zfill(5).startswith("23")
]

if len(maine_counties) != 16:
    raise ValueError(
        f"Expected 16 Maine counties, found {len(maine_counties)}"
    )

county_shapes = [
    shape(f["geometry"]) for f in maine_counties
]

maine_land = unary_union(county_shapes)

latitudes = anomaly.latitude.values
longitudes = anomaly.longitude.values

lat_step = float(np.median(np.abs(np.diff(latitudes))))
lon_step = float(np.median(np.abs(np.diff(longitudes))))

features = []
county_values = {
    str(f["id"]).zfill(5): []
    for f in maine_counties
}

for lat in latitudes:
    for lon in longitudes:
        lat = float(lat)
        lon = float(lon)

        point = Point(lon, lat)

        if not maine_land.covers(point):
            continue

        value = float(
            anomaly.sel(
                latitude=lat,
                longitude=lon
            ).values
        )

        current = float(
            observed_f.sel(
                latitude=lat,
                longitude=lon
            ).values
        )

        normal = float(
            baseline_f.sel(
                latitude=lat,
                longitude=lon
            ).values
        )

        if not all(np.isfinite(v) for v in [
            value, current, normal
        ]):
            continue

        cell = box(
            lon - lon_step / 2,
            lat - lat_step / 2,
            lon + lon_step / 2,
            lat + lat_step / 2
        )

        clipped = cell.intersection(maine_land)

        if clipped.is_empty:
            continue

        features.append({
            "type": "Feature",
            "geometry": mapping(clipped),
            "properties": {
                "anomaly_f": round(value, 2),
                "temperature_f": round(current, 2),
                "normal_f": round(normal, 2)
            }
        })

        for f, geom in zip(maine_counties, county_shapes):
            if geom.covers(point):
                county_values[str(f["id"]).zfill(5)].append(
                    (value, current, normal)
                )
                break

if not features:
    raise RuntimeError("No valid Maine climate grid cells")

output = {
    "type": "FeatureCollection",
    "features": features
}

(DATA / "anomalies.geojson").write_text(
    json.dumps(output, separators=(",", ":"))
)

county_stats = {}

for f in maine_counties:
    fips = str(f["id"]).zfill(5)
    values = county_values[fips]

    if not values:
        continue

    a = np.array(values)

    county_stats[fips] = {
        "anomaly_f": round(float(a[:, 0].mean()), 2),
        "temperature_f": round(float(a[:, 1].mean()), 2),
        "normal_f": round(float(a[:, 2].mean()), 2),
        "grid_points": len(values)
    }

summary = {
    "date": "2025-10-15",
    "baseline": "1961-1990",
    "baseline_method": "Same calendar day, 30 years",
    "source": "Copernicus ERA5-Land daily statistics",
    "measurement": "Daily mean 2m temperature",
    "units": "Fahrenheit",
    "county_statistics": county_stats,
    "grid_cell_count": len(features),
    "methodology_note": (
        "Grid-center sampling, with equal weighting for "
        "county summaries. Not area-weighted county means."
    )
}

(DATA / "summary.json").write_text(
    json.dumps(summary, indent=2)
)

print("Climate processing complete.")
print("Grid cells:", len(features))
print("Counties with data:", len(county_stats))
print("Created data/anomalies.geojson")
print("Created data/summary.json")
