"""Build Maine weekly USDM polygons and exclusive land-area percentages.
Official USDM cleaned GIS archive: https://droughtmonitor.unl.edu/Data/shapefiles_r/USDM_Current.zip
"""
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import json, re, zipfile, tempfile, urllib.request
import geopandas as gpd
from shapely.geometry import shape, mapping
from shapely.ops import unary_union
from pyproj import Geod

DATA=Path('data'); DATA.mkdir(exist_ok=True)
URL='https://droughtmonitor.unl.edu/Data/shapefiles_r/USDM_Current.zip'
COUNTIES='https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json'
GEOD=Geod(ellps='WGS84')

def area(g):
    if g.is_empty:return 0.0
    return abs(GEOD.geometry_area_perimeter(g)[0])

with urllib.request.urlopen(COUNTIES, timeout=90) as r: us=json.load(r)
counties={str(f['id']).zfill(5):shape(f['geometry']) for f in us['features'] if str(f['id']).zfill(5).startswith('23')}
if len(counties)!=16:raise RuntimeError('Expected 16 Maine counties')
maine=unary_union(list(counties.values()))

with urllib.request.urlopen(URL, timeout=120) as r: raw=r.read()
if len(raw)<10000:raise RuntimeError('Unexpectedly small drought download')
with tempfile.TemporaryDirectory() as td:
    path=Path(td)/'drought.zip';path.write_bytes(raw)
    with zipfile.ZipFile(path) as z:
        names=z.namelist()
        shp=[n for n in names if n.lower().endswith('.shp') and '__macosx' not in n.lower()]
        if len(shp)!=1:raise RuntimeError(f'Expected one USDM polygon shapefile; found {shp}')
        # Archive contains official date in its member names (not the machine's current date).
        matches=re.findall(r'20\d{6}', ' '.join(names))
        if not matches:raise RuntimeError('Cannot determine drought observation date from source archive')
        valid=[]
        for x in matches:
            try:valid.append(datetime.strptime(x,'%Y%m%d').date())
            except ValueError:pass
        if not valid:raise RuntimeError('Invalid source drought date')
        observation=max(valid).isoformat()
        # Extract only after verifying paths cannot escape temporary directory.
        for name in names:
            target=(Path(td)/name).resolve()
            if not target.is_relative_to(Path(td).resolve()):raise RuntimeError('Unsafe archive path')
        z.extractall(td)
    frame=gpd.read_file(str(Path(td)/shp[0]))
    if frame.crs is None:raise RuntimeError('Drought source has no CRS')
    frame=frame.to_crs('EPSG:4326')

    cols={c.lower():c for c in frame.columns}
    dmcol=next((cols[c] for c in ('dm','drought','severity','droughtlevel') if c in cols),None)
    if dmcol is None:raise RuntimeError(f'Cannot find USDM category field: {list(frame.columns)}')
    raw_groups=defaultdict(list)
    for _,row in frame.iterrows():
        value=str(row[dmcol]).strip().upper()
        match=re.search(r'(?:D)?([0-4])',value)
        if not match:raise RuntimeError(f'Unexpected USDM category: {value}')
        level=int(match.group(1))
        geom=row.geometry
        if geom is not None and not geom.is_empty:
            geom=geom.intersection(maine)
            if not geom.is_empty:raw_groups[level].append(geom)

# Some source formats are nested thresholds. Always assign each location its highest severity.
exclusive={}
claimed=None
for level in range(4,-1,-1):
    g=unary_union(raw_groups[level]) if raw_groups[level] else None
    if g is None:continue
    if claimed is not None:g=g.difference(claimed)
    if not g.is_empty:exclusive[level]=g
    claimed=unary_union([claimed,g]) if claimed is not None else g

features=[{'type':'Feature','geometry':mapping(g),'properties':{'dm':f'D{level}','severity':level}} for level,g in sorted(exclusive.items())]
if not features:
    # A legitimately drought-free Maine is valid; source date still required.
    print('No drought polygons intersect Maine.')


def statistics(region):
    total=area(region)
    if total<=0:raise RuntimeError('Invalid region area')
    pct={}
    for level in range(5):
        g=exclusive.get(level)
        covered=area(region.intersection(g)) if g is not None else 0
        pct[f'D{level}']=round(min(100,100*covered/total),2)
    pct['none']=round(max(0,100-sum(pct.values())),2)
    pct['D1_plus']=round(sum(pct[f'D{i}'] for i in range(1,5)),2)
    pct['D0_plus']=round(sum(pct[f'D{i}'] for i in range(5)),2)
    pct['area_km2']=round(total/1e6,2)
    return pct

summary={'date':observation,'source':'U.S. Drought Monitor','source_url':URL,'units':'Percent of mapped land area','state_statistics':statistics(maine),'county_statistics':{fips:statistics(g) for fips,g in counties.items()},'methodology_note':'Exclusive highest-severity USDM polygons, clipped to county boundaries; geodesic land-area shares. D0 is abnormally dry, not drought.'}
geo={'type':'FeatureCollection','features':features}
for path,object_ in [(DATA/'drought.geojson',geo),(DATA/'drought-summary.json',summary)]:path.write_text(json.dumps(object_,separators=(',',':')))
archive=DATA/'drought-history'/observation;archive.mkdir(parents=True,exist_ok=True)
(archive/'drought.geojson').write_bytes((DATA/'drought.geojson').read_bytes())
(archive/'drought-summary.json').write_bytes((DATA/'drought-summary.json').read_bytes())
manifest=DATA/'drought-history'/'index.json'
old=json.loads(manifest.read_text()) if manifest.exists() else []
manifest.write_text(json.dumps(sorted(set(old+[observation]),reverse=True),indent=2))
print('USDM valid date',observation,'Maine D1+ area',summary['state_statistics']['D1_plus'],'%')
