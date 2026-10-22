
"""Maine Climate Monitor: process one historical USDM week.

Usage: python3 process_drought_history.py 2020-01-07
Requires: geopandas, shapely, pyproj
Does not modify current drought files.
"""

import argparse
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
import io
import json
import urllib.error
import urllib.request
import zipfile
import tempfile

import geopandas as gpd
from pyproj import Geod
from shapely.geometry import shape, mapping
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
HISTORY = DATA / "drought-history"

COUNTIES_URL = (
    "https://raw.githubusercontent.com/plotly/datasets/"
    "master/geojson-counties-fips.json"
)

GEOD = Geod(ellps="WGS84")


def download(url):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "MaineClimateMonitor/2.0 (historical research)"
        },
    )
    with urllib.request.urlopen(req, timeout=180) as response:
        return response.read()


def geodesic_area(geometry):
    if geometry is None or geometry.is_empty:
        return 0.0
    area, _ = GEOD.geometry_area_perimeter(geometry)
    return abs(float(area))


def append_clipped(groups, level, geometry, maine):
    if geometry is None or geometry.is_empty:
        return

    if not geometry.is_valid:
        geometry = geometry.buffer(0)

    if geometry.is_empty:
        return

    clipped = geometry.intersection(maine)

    if not clipped.is_empty:
        groups[level].append(clipped)


def download_archive(stamp):
    urls = [
        (
            "https://droughtmonitor.unl.edu/Data/"
            f"shapefiles_r/USDM_{stamp}.zip"
        ),
        (
            "https://droughtmonitor.unl.edu/Data/"
            f"shapefiles_m/USDM_{stamp}_M.zip"
        ),
    ]

    for url in urls:
        print(f"Downloading official USDM archive: {url}", flush=True)

        try:
            raw = download(url)
            if len(raw) < 10000:
                raise RuntimeError("Archive is unexpectedly small")
            return raw, url

        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                print("Archive not found; trying alternate format...")
                continue
            raise

    raise RuntimeError(
        f"No supported USDM archive found for {stamp}"
    )


def load_drought_groups(raw, maine, stamp):
    groups = defaultdict(list)

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp).resolve()

        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            for member in archive.infolist():
                target = (base / member.filename).resolve()

                if not target.is_relative_to(base):
                    raise RuntimeError(
                        f"Unsafe ZIP member: {member.filename}"
                    )

            archive.extractall(base)

        shapefiles = list(base.rglob("*.shp"))

        separate = {}

        for level in range(5):
            expected = f"Drought_Areas_US_D{level}.shp"
            matches = [
                p for p in shapefiles
                if p.name.lower() == expected.lower()
            ]

            if len(matches) == 1:
                separate[level] = matches[0]

        if len(separate) == 5:
            print("Archive format: separate D0-D4 shapefiles")

            for level in range(5):
                frame = gpd.read_file(separate[level])

                if frame.crs is None:
                    raise RuntimeError("Shapefile has no CRS")

                frame = frame.to_crs("EPSG:4326")

                for geometry in frame.geometry:
                    append_clipped(
                        groups, level, geometry, maine
                    )

        else:
            expected = f"USDM_{stamp}.shp"

            matches = [
                p for p in shapefiles
                if p.name.lower() == expected.lower()
            ]

            if len(matches) != 1:
                raise RuntimeError(
                    "Unrecognized USDM archive format"
                )

            print(
                "Archive format: combined shapefile with DM attribute"
            )

            frame = gpd.read_file(matches[0])

            if frame.crs is None:
                raise RuntimeError("Shapefile has no CRS")

            if "DM" not in frame.columns:
                raise RuntimeError(
                    "Combined shapefile has no DM field"
                )

            frame = frame.to_crs("EPSG:4326")

            for _, row in frame.iterrows():
                level = int(row["DM"])

                if level not in range(5):
                    raise RuntimeError(
                        f"Unexpected drought severity: {level}"
                    )

                append_clipped(
                    groups, level, row.geometry, maine
                )

    for level in range(5):
        print(
            f"D{level}: {len(groups[level])} "
            "Maine-intersecting polygons",
            flush=True,
        )

    return groups


def process(iso_date):
    day = date.fromisoformat(iso_date)

    if day.isoformat() != iso_date or day.weekday() != 1:
        raise ValueError(
            "Date must be a Tuesday in YYYY-MM-DD format"
        )

    if not date(2000, 1, 1) <= day <= date.today():
        raise ValueError("Date outside historical range")

    stamp = day.strftime("%Y%m%d")
    destination = HISTORY / iso_date

    if destination.exists():
        raise FileExistsError(
            f"{destination} already exists; no overwrite"
        )

    print("Downloading county boundaries...", flush=True)

    county_json = json.loads(download(COUNTIES_URL))
    counties = {}

    for feature in county_json["features"]:
        fips = str(feature.get("id", "")).zfill(5)

        if fips.startswith("23"):
            counties[fips] = shape(feature["geometry"])

    if len(counties) != 16:
        raise RuntimeError(
            f"Expected 16 Maine counties, found {len(counties)}"
        )

    maine = unary_union(list(counties.values()))

    raw, url = download_archive(stamp)
    groups = load_drought_groups(raw, maine, stamp)

    exclusive = {}
    claimed = None

    for level in range(4, -1, -1):
        if not groups[level]:
            continue

        geometry = unary_union(groups[level])

        if claimed is not None:
            geometry = geometry.difference(claimed)

        if geometry.is_empty:
            continue

        exclusive[level] = geometry

        claimed = (
            geometry
            if claimed is None
            else unary_union([claimed, geometry])
        )

    features = [
        {
            "type": "Feature",
            "geometry": mapping(exclusive[level]),
            "properties": {
                "dm": f"D{level}",
                "severity": level,
            },
        }
        for level in sorted(exclusive)
    ]

    geojson = {
        "type": "FeatureCollection",
        "features": features,
    }

    def statistics(region):
        total = geodesic_area(region)

        if total <= 0:
            raise RuntimeError("Invalid region area")

        raw_pct = {}

        for level in range(5):
            geometry = exclusive.get(level)

            covered = (
                geodesic_area(region.intersection(geometry))
                if geometry is not None
                else 0.0
            )

            raw_pct[f"D{level}"] = 100.0 * covered / total

        assigned = sum(raw_pct.values())

        if assigned > 100.25:
            raise RuntimeError(
                f"Overlapping coverage exceeds 100%: {assigned}"
            )

        result = {
            key: round(value, 2)
            for key, value in raw_pct.items()
        }

        result["none"] = round(
            max(0.0, 100.0 - assigned), 2
        )

        result["D1_plus"] = round(
            sum(raw_pct[f"D{i}"] for i in range(1, 5)), 2
        )

        result["D0_plus"] = round(assigned, 2)
        result["area_km2"] = round(total / 1e6, 2)

        return result

    state_stats = statistics(maine)

    county_stats = {
        fips: statistics(geom)
        for fips, geom in counties.items()
    }

    for label, row in [
        ("Maine", state_stats),
        *county_stats.items(),
    ]:
        values = [
            row[f"D{i}"] for i in range(5)
        ] + [row["none"]]

        if (
            any(not 0 <= v <= 100 for v in values)
            or abs(sum(values) - 100) >= 0.25
        ):
            raise RuntimeError(
                f"Invalid percentages for {label}: {values}"
            )

    summary = {
        "date": iso_date,
        "source": "U.S. Drought Monitor",
        "source_url": url,
        "units": "Percent of mapped land area",
        "date_method": (
            "Official date encoded in the dated USDM archive filename"
        ),
        "date_verified": True,
        "processed_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "state_statistics": state_stats,
        "county_statistics": county_stats,
        "methodology_note": (
            "D0-D4 polygons clipped to Maine counties, "
            "assigned mutually exclusive highest-severity "
            "categories, with geodesic area coverage. "
            "D0 is abnormally dry, not drought."
        ),
    }

    HISTORY.mkdir(parents=True, exist_ok=True)
    manifest = HISTORY / "index.json"

    existing = (
        json.loads(manifest.read_text(encoding="utf-8"))
        if manifest.exists()
        else []
    )

    if not isinstance(existing, list):
        raise RuntimeError(
            "History index.json must be a list"
        )

    destination.mkdir(exist_ok=False)

    try:
        (destination / "drought.geojson").write_text(
            json.dumps(geojson, separators=(",", ":")),
            encoding="utf-8",
        )

        (destination / "drought-summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

        manifest.write_text(
            json.dumps(
                sorted(
                    set(existing + [iso_date]),
                    reverse=True,
                ),
                indent=2,
            ),
            encoding="utf-8",
        )

    except Exception:
        for p in destination.glob("*"):
            p.unlink()

        destination.rmdir()
        raise

    print(f"\nSUCCESS: {iso_date}", flush=True)
    print(f"Historical files: {destination}", flush=True)
    print(
        f'Maine D1+ coverage: {state_stats["D1_plus"]}%',
        flush=True,
    )
    print(
        "Current drought.geojson and "
        "drought-summary.json were not modified.",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Process one historical U.S. Drought Monitor Tuesday"
        )
    )

    parser.add_argument(
        "date",
        help="Observation date, e.g. 2020-01-07",
    )

    args = parser.parse_args()
    process(args.date)
