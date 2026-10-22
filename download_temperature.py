"""Download latest available ERA5-Land daily mean and calendar-day 1961-1990 baseline.
Requires CDS_API_KEY repository secret. Baseline cached in data/baselines/.
"""
import os
from pathlib import Path
from datetime import date, timedelta
import zipfile
import cdsapi
import xarray as xr

DATA = Path('data')
DATA.mkdir(exist_ok=True)
(DATA/'baselines').mkdir(exist_ok=True)
client = cdsapi.Client(url=os.getenv('CDSAPI_URL', 'https://cds.climate.copernicus.eu/api'), key=os.environ['CDSAPI_KEY'])
DATASET = 'derived-era5-land-daily-statistics'

def request(day, years):
    return dict(variable=['2m_temperature'], year=[str(y) for y in years], month=[f'{day.month:02d}'], day=[f'{day.day:02d}'], daily_statistic='daily_mean', time_zone='utc+00:00', frequency='1_hourly', area=[47.6,-71.2,42.8,-66.8])

def download(day, years, destination):
    tmp = destination.with_suffix('.download')
    client.retrieve(DATASET, request(day, years)).download(str(tmp))
    if not tmp.exists() or tmp.stat().st_size == 0:
        raise RuntimeError(f'Empty download: {tmp}')
    if zipfile.is_zipfile(tmp):
        with zipfile.ZipFile(tmp) as archive:
            members = [n for n in archive.namelist() if n.lower().endswith(('.nc','.nc4'))]
            if len(members) != 1: raise RuntimeError('Expected exactly one NetCDF in CDS archive')
            tmp.write_bytes(archive.read(members[0]))
    with xr.open_dataset(tmp, engine='netcdf4') as ds:
        if not ds.data_vars: raise RuntimeError('Empty NetCDF')
        print('Downloaded', destination, dict(ds.sizes), flush=True)
    tmp.replace(destination)

# Start conservatively 6 days back and walk backwards to tolerate ERA5-Land release lag.
# Never publish a stale date if a newer valid daily file exists.
forced = os.getenv('TARGET_DATE')
if forced:
    candidates = [date.fromisoformat(forced)]
else:
    candidates = [date.today()-timedelta(days=n) for n in range(3, 22)]

errors=[]
for day in candidates:
    try:
        download(day, [day.year], DATA/'target.nc')
        baseline = DATA/'baselines'/f'{day.month:02d}-{day.day:02d}.nc'
        if not baseline.exists():
            # Feb 29 only exists in leap years; processor validates available count.
            years = [y for y in range(1961,1991) if day.month!=2 or day.day!=29 or (y%4==0 and (y%100!=0 or y%400==0))]
            download(day, years, baseline)
        (DATA/'baseline.nc').write_bytes(baseline.read_bytes())
        (DATA/'target-date.txt').write_text(day.isoformat())
        print('Chosen ERA5-Land date:', day, flush=True)
        break
    except Exception as exc:
        errors.append(f'{day}: {exc}')
        print('Date unavailable:', day, str(exc), flush=True)
else:
    raise RuntimeError('No recent ERA5-Land date available; retaining previous published data.\n'+'\n'.join(errors))
