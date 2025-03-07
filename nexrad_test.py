from siphon.radarserver import RadarServer
from datetime import datetime, timedelta
import numpy as np
import metpy.plots as mpplots
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pytz

rs = RadarServer('http://tds-nexrad.scigw.unidata.ucar.edu/thredds/radarServer/nexrad/level2/S3/')
query = rs.query()

dt = datetime(2007, 12, 5, 12, 12, 0, tzinfo=pytz.UTC)
query.stations('KLVX').time_range(dt, dt + timedelta(hours=1))

rs.validate_query(query)

catalog = rs.get_catalog(query)

print(catalog.datasets)

data = catalog.datasets[0].remote_access()

def raw_to_masked_float(var, data):
    # Values come back signed. If the _Unsigned attribute is set, we need to convert
    # from the range [-127, 128] to [0, 255].
    if hasattr(var, "_Unsigned") and var._Unsigned:
        data = data & 255

    # Mask missing points
    data = np.ma.array(data, mask=data==0)

    # Convert to float using the scale and offset
    return data * var.scale_factor + var.add_offset

def polar_to_cartesian(az, rng):
    az_rad = np.deg2rad(az)[:, None]
    x = rng * np.sin(az_rad)
    y = rng * np.cos(az_rad)
    return x, y

print(data.variables.keys())


sweep = 0

reflectivity_labels = ['Reflectivity_HI', 'Reflectivity']
for reflectivity_label in reflectivity_labels:
    try:
        ref_var = data.variables[reflectivity_label]
        break
    except:
        continue

ref_data = ref_var[sweep]

distance_labels = ['distanceR_HI', 'distanceR']
for distance_label in distance_labels:
    try:
        rng = data.variables[distance_label][:]
        break
    except:
        continue

azimuth_labels = ['azimuthR_HI', 'azimuthR']
for azimuth_label in azimuth_labels:
    try:
        az = data.variables[azimuth_label][sweep]
        break
    except:
        continue

ref = raw_to_masked_float(ref_var, ref_data)
x, y = polar_to_cartesian(az, rng)

ref_norm, ref_cmap = mpplots.ctables.registry.get_with_steps('NWSReflectivity', 5, 5)

def new_map(fig, lon, lat):
    # Create projection centered on the radar. This allows us to use x
    # and y relative to the radar.
    proj = ccrs.LambertConformal(central_longitude=lon, central_latitude=lat)

    # New axes with the specified projection
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96], projection=proj)

    # Add coastlines and states
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=2)
    ax.add_feature(cfeature.STATES.with_scale('50m'))
    
    return ax


fig = plt.figure(figsize=(10, 10))
ax = new_map(fig, data.StationLongitude, data.StationLatitude)
ax.pcolormesh(x, y, ref, cmap=ref_cmap, norm=ref_norm, zorder=0)

fig.savefig("test.png")