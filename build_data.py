"""Build real ERA5-Land daily anomaly grid and county boundaries for Maine.
Requires: pip install requests shapely
Run: python build_data.py --date 2026-10-15
"""
import argparse, datetime as dt, json, math, pathlib, time
import requests
from shapely.geometry import shape, Point, mapping
from shapely.ops import unary_union

HERE=pathlib.Path(__file__).parent
GIS='https://gis.mcht.org/arcgis/rest/services/AdminPolitical/MEGIS_Boundary_Counties/MapServer/0/query'
API='https://archive-api.open-meteo.com/v1/archive'
s=requests.Session()

def fetch(url, params):
    for n in range(5):
        try:
            r=s.get(url,params=params,timeout=120);r.raise_for_status();return r.json()
        except Exception:
            if n==4:raise
            time.sleep(2**n)

def counties():
    path=HERE/'counties.geojson'
    if path.exists():return json.loads(path.read_text())
    data=fetch(GIS,{'where':'1=1','outFields':'*','returnGeometry':'true','outSR':'4326','f':'geojson'})
    if data.get('type')!='FeatureCollection' or len(data.get('features',[]))<16:
        raise RuntimeError('County GIS service did not return 16 polygons; inspect endpoint and fields')
    path.write_text(json.dumps(data));return data

def get_name(props):
    for k,v in props.items():
        if isinstance(v,str) and v.strip().lower().replace(' county','') in {x.lower() for x in NAMES}:return v.strip().replace(' County','')
    return None
NAMES=['Androscoggin','Aroostook','Cumberland','Franklin','Hancock','Kennebec','Knox','Lincoln','Oxford','Penobscot','Piscataquis','Sagadahoc','Somerset','Waldo','Washington','York']

def main():
    p=argparse.ArgumentParser();p.add_argument('--date',default=(dt.date.today()-dt.timedelta(days=8)).isoformat());p.add_argument('--step',type=float,default=.1);args=p.parse_args()
    target=dt.date.fromisoformat(args.date)
    if target>dt.date.today()-dt.timedelta(days=5):raise SystemExit('ERA5-Land may lag ~5 days. Choose a date at least 5 days old.')
    geo=counties();features=[]
    for f in geo['features']:
        name=get_name(f.get('properties',{}))
        if name:features.append({'type':'Feature','properties':{'name':name},'geometry':f['geometry']})
    if len(features)!=16:raise RuntimeError(f'Expected 16 named counties; found {len(features)}. Inspect counties.geojson attributes.')
    (HERE/'counties.geojson').write_text(json.dumps({'type':'FeatureCollection','features':features}))
    polys=[(f['properties']['name'],shape(f['geometry'])) for f in features]
    land=unary_union([g for _,g in polys]);xmin,ymin,xmax,ymax=land.bounds
    # Grid cell centers; include cells whose centers fall on land. These are samples of model values.
    step=args.step;coords=[];y=math.floor(ymin/step)*step
    while y<=ymax:
        x=math.floor(xmin/step)*step
        while x<=xmax:
            pt=Point(x,y)
            if land.covers(pt):
                county=next((n for n,g in polys if g.covers(pt)),None)
                if county:coords.append((round(y,4),round(x,4),county))
            x+=step
        y+=step
    print('Grid sample points:',len(coords),flush=True)
    # Query a 31-day climatology window for each year 1961–1990, plus target day.
    # Batch only a few points to avoid oversized API requests.
    doy=target.timetuple().tm_yday
    dates=[]
    for yr in range(1961,1991):
        center=dt.date(yr,1,1)+dt.timedelta(days=min(doy,365)-1)
        dates.extend([center+dt.timedelta(days=k) for k in range(-15,16)])
    # Archive API supports contiguous date ranges; 30 years per point would be costly.
    # Download the full baseline once per batch, then subset dates locally.
    wanted={d.isoformat() for d in dates}
    results=[]
    for start in range(0,len(coords),5):
        batch=coords[start:start+5]
        params={'latitude':','.join(str(v[0]) for v in batch),'longitude':','.join(str(v[1]) for v in batch),
                'start_date':'1960-12-01','end_date':target.isoformat(),'daily':'temperature_2m_mean',
                'models':'era5_land','temperature_unit':'fahrenheit','timezone':'America/New_York','cell_selection':'land'}
        response=fetch(API,params);response=response if isinstance(response,list) else [response]
        if len(response)!=len(batch):raise RuntimeError('Unexpected API batch response')
        for (lat,lon,county),item in zip(batch,response):
            times=item['daily']['time'];values=item['daily']['temperature_2m_mean'];lookup=dict(zip(times,values))
            baseline=[lookup[d] for d in wanted if lookup.get(d) is not None]
            current=lookup.get(target.isoformat())
            if len(baseline)<850 or current is None:continue
            normal=sum(baseline)/len(baseline)
            results.append({'lat':lat,'lon':lon,'county':county,'temperature_f':round(current,1),'normal_f':round(normal,1),'anomaly_f':round(current-normal,1)})
        print(f'Processed {min(start+5,len(coords))}/{len(coords)}',flush=True)
        time.sleep(.3)
    if not results:raise RuntimeError('No data returned')
    summaries={}
    for name in NAMES:
        pts=[r for r in results if r['county']==name]
        if pts:summaries[name]={key:round(sum(p[key] for p in pts)/len(pts),1) for key in ('temperature_f','normal_f','anomaly_f')}
    out={'date':target.isoformat(),'baseline':'1961–1990, 31-day window','model':'ERA5-Land','step_degrees':step,
         'method':'Equal-weight grid-center sampling; approximate county means, not exact area-weighted values',
         'grid':results,'counties':summaries}
    (HERE/'climate.json').write_text(json.dumps(out,separators=(',',':')))
    print('Saved climate.json with',len(results),'samples')
if __name__=='__main__':main()
