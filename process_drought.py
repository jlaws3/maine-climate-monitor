
"""Maine Climate Monitor — weekly U.S. Drought Monitor processing.

Reads the five separate D0–D4 shapefiles in the official
USDM_Current.zip archive.

Produces:
  data/drought.geojson
  data/drought-summary.json
  data/drought-history/YYYY-MM-DD/...
  data/drought-history/index.json

The source GIS creation date is used to infer the most recent
Tuesday reporting date. This inference is explicitly recorded
in the output metadata and should be independently verified
against the official USDM publication before production use.
"""

from pathlib import Path
from datetime import datetime, timedelta, timezone
from collections import defaultdict
import json
import re
import tempfile
import urllib.request
import zipfile
import xml.etree.ElementTree as ET

import geopandas as gpd
from shapely.geometry import shape, mapping
from shapely.ops import unary_union
from pyproj import Geod


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)

URL = (
    "https://droughtmonitor.unl.edu/Data/"
    "shapefiles_r/USDM_Current.zip"
)

COUNTIES_URL = (
    "https://raw.githubusercontent.com/plotly/datasets/"
    "master/geojson-counties-fips.json"
)

GEOD = Geod(ellps="WGS84")


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def geodesic_area(geometry):
    if geometry is None or geometry.is_empty:
        return 0.0

    area, _ = GEOD.geometry_area_perimeter(geometry)
    return abs(float(area))


def download(url):
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "MaineClimateMonitor/2.0"}
    )

    with urllib.request.urlopen(
        request,
        timeout=180
    ) as response:
        return response.read()


def union_geometries(geometries):
    valid = [
        g for g in geometries
        if g is not None and not g.is_empty
    ]

    if not valid:
        return None

    return unary_union(valid)


def get_creation_date(archive):
    """Read the source GIS creation date, not an observation date."""

    xml_files = [
        name for name in archive.namelist()
        if name.lower().endswith(
            "drought_areas_us_d0.shp.xml"
        )
    ]

    if not xml_files:
        raise RuntimeError(
            "D0 GIS metadata file is missing; "
            "cannot establish source date."
        )

    root = ET.fromstring(archive.read(xml_files[0]))

    for element in root.iter():
        tag = element.tag.split("}")[-1].lower()

        if tag == "creadate":
            value = (element.text or "").strip()

            if re.fullmatch(r"\d{8}", value):
                return datetime.strptime(
                    value, "%Y%m%d"
                ).date()

    raise RuntimeError(
        "Could not find a valid GIS creation date."
    )


def reporting_date_from_creation(created):
    """Infer the Tuesday on or before the GIS creation date."""

    days_since_tuesday = (created.weekday() - 1) % 7

    return created - timedelta(days=days_since_tuesday)


# --------------------------------------------------
# LOAD MAINE COUNTY GEOGRAPHY
# --------------------------------------------------

print("Downloading Maine county boundaries...", flush=True)

county_data = json.loads(
    download(COUNTIES_URL).decode("utf-8")
)

counties = {}

for feature in county_data["features"]:
    fips = str(feature.get("id", "")).zfill(5)

    if fips.startswith("23"):
        counties[fips] = shape(feature["geometry"])

if len(counties) != 16:
    raise RuntimeError(
        f"Expected 16 Maine counties, found {len(counties)}"
    )

maine = unary_union(list(counties.values()))

print("Loaded 16 Maine counties.", flush=True)


# --------------------------------------------------
# DOWNLOAD OFFICIAL DROUGHT DATA
# --------------------------------------------------

print("Downloading U.S. Drought Monitor...", flush=True)

raw = download(URL)

if len(raw) < 10000:
    raise RuntimeError(
        "Drought download is unexpectedly small."
    )

raw_groups = defaultdict(list)

with tempfile.TemporaryDirectory() as temp:
    temp_path = Path(temp).resolve()
    zip_path = temp_path / "usdm.zip"
    zip_path.write_bytes(raw)

    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()

        creation_date = get_creation_date(archive)
        observation_date = reporting_date_from_creation(
            creation_date
        )

        observation = observation_date.isoformat()

        print(
            "GIS creation date:",
            creation_date,
            flush=True
        )

        print(
            "Inferred Tuesday reporting date:",
            observation,
            flush=True
        )

        # Reject unsafe ZIP paths.
        for name in names:
            destination = (temp_path / name).resolve()

            if not destination.is_relative_to(temp_path):
                raise RuntimeError(
                    f"Unsafe archive member: {name}"
                )

        archive.extractall(temp_path)

    # --------------------------------------------------
    # READ FIVE DROUGHT SEVERITY LAYERS
    # --------------------------------------------------

    for level in range(5):
        filename = f"Drought_Areas_US_D{level}.shp"

        matches = list(temp_path.rglob(filename))

        if len(matches) != 1:
            raise RuntimeError(
                f"Expected exactly one {filename}; "
                f"found {len(matches)}"
            )

        frame = gpd.read_file(matches[0])

        if frame.crs is None:
            raise RuntimeError(
                f"{filename} has no coordinate system."
            )

        frame = frame.to_crs("EPSG:4326")

        count = 0

        for geometry in frame.geometry:
            if geometry is None or geometry.is_empty:
                continue

            if not geometry.is_valid:
                geometry = geometry.buffer(0)

            if geometry.is_empty:
                continue

            clipped = geometry.intersection(maine)

            if not clipped.is_empty:
                raw_groups[level].append(clipped)
                count += 1

        print(
            f"D{level}: {count} polygons intersect Maine",
            flush=True
        )


# --------------------------------------------------
# MAKE SEVERITY CATEGORIES MUTUALLY EXCLUSIVE
# --------------------------------------------------

# Higher severity takes precedence.
# This works whether the input layers contain
# nested thresholds or disjoint severity polygons.

exclusive = {}
claimed = None

for level in range(4, -1, -1):
    geometry = union_geometries(raw_groups[level])

    if geometry is None:
        continue

    if claimed is not None:
        geometry = geometry.difference(claimed)

    if geometry.is_empty:
        continue

    exclusive[level] = geometry

    # Include the complete assigned area before
    # processing lower severity categories.
    if claimed is None:
        claimed = geometry
    else:
        claimed = unary_union([claimed, geometry])


# --------------------------------------------------
# CREATE DROUGHT GEOJSON
# --------------------------------------------------

features = []

for level in sorted(exclusive):
    geometry = exclusive[level]

    features.append({
        "type": "Feature",
        "geometry": mapping(geometry),
        "properties": {
            "dm": f"D{level}",
            "severity": level
        }
    })

geojson = {
    "type": "FeatureCollection",
    "features": features
}

if not features:
    print(
        "No drought polygons intersect Maine.",
        flush=True
    )


# --------------------------------------------------
# COUNTY AND STATE STATISTICS
# --------------------------------------------------

def statistics(region):
    total = geodesic_area(region)

    if total <= 0:
        raise RuntimeError("Invalid region area.")

    # Calculate unrounded percentages first to
    # minimize rounding discrepancies.
    raw_pct = {}

    for level in range(5):
        geometry = exclusive.get(level)

        covered = (
            geodesic_area(region.intersection(geometry))
            if geometry is not None
            else 0.0
        )

        raw_pct[f"D{level}"] = (
            100.0 * covered / total
        )

    assigned = sum(raw_pct.values())

    if assigned > 100.25:
        raise RuntimeError(
            "Overlapping drought coverage exceeds "
            f"100%: {assigned:.3f}%"
        )

    raw_none = max(0.0, 100.0 - assigned)

    result = {
        key: round(value, 2)
        for key, value in raw_pct.items()
    }

    result["none"] = round(raw_none, 2)

    result["D1_plus"] = round(
        sum(raw_pct[f"D{i}"] for i in range(1, 5)),
        2
    )

    result["D0_plus"] = round(assigned, 2)

    result["area_km2"] = round(total / 1e6, 2)

    return result


state_statistics = statistics(maine)

county_statistics = {
    fips: statistics(geometry)
    for fips, geometry in counties.items()
}

if len(county_statistics) != 16:
    raise RuntimeError(
        "Expected statistics for all 16 counties."
    )


# --------------------------------------------------
# OUTPUT METADATA
# --------------------------------------------------

summary = {
    "date": observation,
    "source": "U.S. Drought Monitor",
    "source_url": URL,
    "units": "Percent of mapped land area",
    "source_file_created": creation_date.isoformat(),
    "date_method": (
        "Provisional reporting date inferred as the "
        "most recent Tuesday on or before the GIS "
        "creation date. Verify against the official "
        "USDM valid date before publication."
    ),
    "date_verified": False,
    "processed_at": datetime.now(
        timezone.utc
    ).isoformat(),
    "state_statistics": state_statistics,
    "county_statistics": county_statistics,
    "methodology_note": (
        "Official USDM D0-D4 polygons are assigned "
        "to mutually exclusive highest-severity "
        "categories and clipped to Maine county "
        "boundaries. Coverage is calculated using "
        "geodesic land area. D0 is abnormally dry, "
        "not drought."
    )
}


# --------------------------------------------------
# VALIDATION
# --------------------------------------------------

def validate_row(row, label):
    values = [
        row[f"D{i}"] for i in range(5)
    ] + [row["none"]]

    if any(value < 0 or value > 100 for value in values):
        raise RuntimeError(
            f"Invalid percentages for {label}"
        )

    if abs(sum(values) - 100) >= 0.25:
        raise RuntimeError(
            f"Percentages do not total 100 "
            f"for {label}: {sum(values)}"
        )


validate_row(state_statistics, "Maine")

for fips, row in county_statistics.items():
    validate_row(row, fips)


# --------------------------------------------------
# SAVE CURRENT FILES
# --------------------------------------------------

current_geojson = DATA / "drought.geojson"
current_summary = DATA / "drought-summary.json"

current_geojson.write_text(
    json.dumps(geojson, separators=(",", ":")),
    encoding="utf-8"
)

current_summary.write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8"
)


# --------------------------------------------------
# SAVE HISTORICAL ARCHIVE
# --------------------------------------------------

archive_dir = DATA / "drought-history" / observation

archive_dir.mkdir(
    parents=True,
    exist_ok=True
)

(archive_dir / "drought.geojson").write_bytes(
    current_geojson.read_bytes()
)

(archive_dir / "drought-summary.json").write_bytes(
    current_summary.read_bytes()
)

manifest = DATA / "drought-history" / "index.json"

if manifest.exists():
    old_dates = json.loads(
        manifest.read_text(encoding="utf-8")
    )
else:
    old_dates = []

all_dates = sorted(
    set(old_dates + [observation]),
    reverse=True
)

manifest.write_text(
    json.dumps(all_dates, indent=2),
    encoding="utf-8"
)


# --------------------------------------------------
# COMPLETE
# --------------------------------------------------

print("Drought processing complete.", flush=True)
print("Reporting date (inferred):", observation)
print(
    "Maine D1+ coverage:",
    state_statistics["D1_plus"],
    "%",
    flush=True
)
print(
    "Drought polygons:",
    len(features),
    flush=True
)
print(
    "IMPORTANT: Reporting date has not been "
    "independently verified.",
    flush=True
)
