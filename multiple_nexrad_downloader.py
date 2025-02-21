import requests
import datetime
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from metpy.io import Level2File
from metpy.plots import ctables
from pyproj import Proj
from scipy.interpolate import griddata

# AWS S3 bucket configuration
BUCKET_NAME = "noaa-nexrad-level2"
NEXRAD_STATION_LIST_URL = "https://www.ncei.noaa.gov/access/homr/file/nexrad-stations.txt"

# Fetch the list of NEXRAD stations
response = requests.get(NEXRAD_STATION_LIST_URL)
lines = response.text.splitlines()

# Extract headers (skip first two dashed lines)
indices = [
    (0, 8), (9, 13), (14, 19), (20, 50), (51, 71),
    (72, 74), (75, 105), (106, 115), (116, 126), (127, 133),
    (134, 139), (140, 188)
]
keys = ["NCDCID", "ICAO", "WBAN", "NAME", "COUNTRY", "STATE", "COUNTY",
        "LAT", "LON", "ELEV", "UTC", "STNTYPE"]

radar_sites = []
for line in lines[3:]:
    entry = {key: line[start:end].strip() for key, (start, end) in zip(keys, indices)}
    radar_sites.append(entry)

# Define bounding box and target datetime
lon_min, lat_min, lon_max, lat_max = -106.647, 25.840, -93.517, 36.500
dt = datetime.datetime(2022, 2, 1, 8, 55, 2)
show_stations = False

# Filter stations within the bounding box
filtered_sites = []
for site in radar_sites:
    try:
        lat, lon = float(site["LAT"]), float(site["LON"])
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            filtered_sites.append(site)
    except ValueError:
        continue

if not filtered_sites:
    print("No radar sites found in the bounding box.")
    exit()

# Set up S3 clients/resources
s3_client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
s3_resource = boto3.resource('s3', config=Config(signature_version=UNSIGNED, user_agent_extra='Resource'))
bucket = s3_resource.Bucket(BUCKET_NAME)

# Get reflectivity colortable from MetPy
ref_norm, ref_cmap = ctables.registry.get_with_steps('NWSReflectivity', 5, 5)

# Create a master grid covering the bounding box
grid_res = 0.01  # degrees; adjust resolution as needed
grid_lon = np.arange(lon_min, lon_max, grid_res)
grid_lat = np.arange(lat_min, lat_max, grid_res)
grid_lon2d, grid_lat2d = np.meshgrid(grid_lon, grid_lat)
combined_ref = np.full(grid_lon2d.shape, np.nan)
combined_rel = np.full(grid_lon2d.shape, np.inf)  # lower distance is more reliable

processed_station_data = []
# Loop over each station, regrid its data, and update the master grid using reliability
for station in filtered_sites:
    icao = station['ICAO']
    try:
        station_lat = float(station['LAT'])
        station_lon = float(station['LON'])
    except ValueError:
        continue

    # Construct the S3 prefix for this station's data
    prefix = f"{dt.year}/{dt.month:02d}/{dt.day:02d}/{icao}/"
    response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)
    if "Contents" not in response:
        continue

    # Collect available files by parsing timestamps from the filenames
    available_files = []
    for obj in response.get("Contents", []):
        key = obj["Key"]
        parts = key.split("_")
        if len(parts) > 1 and parts[1].isdigit():
            available_files.append((key, int(parts[1])))
    if not available_files:
        continue

    target_time = int(f"{dt.hour:02d}{dt.minute:02d}{dt.second:02d}")
    closest_file = min(available_files, key=lambda x: abs(x[1] - target_time))
    closest_key = closest_file[0]

    # Retrieve the file
    objs = list(bucket.objects.filter(Prefix=closest_key))
    if not objs:
        continue
    obj = objs[0]
    try:
        f = Level2File(obj.get()['Body'])
    except Exception as e:
        print(f"Error reading {obj.key}: {e}")
        continue

    if not f.sweeps or len(f.sweeps[0]) == 0:
        continue
    sweep = 0
    try:
        # Extract azimuth angles (one per ray)
        az = np.array([ray[0].az_angle for ray in f.sweeps[sweep]])
    except Exception as e:
        print(f"Error extracting azimuth angles from {obj.key}: {e}")
        continue

    try:
        # Extract reflectivity data and range information
        ref_hdr = f.sweeps[sweep][0][4][b'REF'][0]
        ref_range = np.arange(ref_hdr.num_gates) * ref_hdr.gate_width + ref_hdr.first_gate
        ref_data = np.array([ray[4][b'REF'][1] for ray in f.sweeps[sweep]])
    except Exception as e:
        print(f"Error extracting reflectivity from {obj.key}: {e}")
        continue

    # Skip if no valid data in this sweep
    if ref_data.size == 0 or np.all(np.isnan(ref_data)):
        continue
    
    print(f"Adding to data: {obj.key}")

    # Define a station-specific projection (polar coordinates relative to station)
    proj_station = Proj(proj="aeqd", datum="WGS84", lat_0=station_lat, lon_0=station_lon)
    # Convert polar (range, azimuth) to x,y coordinates (in km)
    xlocs = ref_range * np.sin(np.deg2rad(az[:, np.newaxis]))
    ylocs = ref_range * np.cos(np.deg2rad(az[:, np.newaxis]))

    # Adjust shapes if necessary
    x_diff = xlocs.shape[1] - ref_data.shape[1]
    if x_diff > 0:
        xlocs = xlocs[:, :-x_diff]
    data_diff = ref_data.shape[1] - xlocs.shape[1]
    if data_diff > 0:
        ref_data = ref_data[:, :-data_diff]

    # Convert x,y (km) to geographic coordinates (lat/lon)
    lon_locs, lat_locs = proj_station(xlocs * 1000, ylocs * 1000, inverse=True)
    # Apply the bounding box filter for this station's grid
    mask = (lon_locs < lon_min) | (lon_locs > lon_max) | (lat_locs < lat_min) | (lat_locs > lat_max)
    ref_data = np.ma.masked_where(mask, ref_data)
    if np.all(ref_data.mask):
        continue

    # Compute a reliability metric: distance from the radar station (km)
    reliability = np.sqrt(xlocs**2 + ylocs**2)
    
    # Flatten the station arrays for interpolation
    points = np.column_stack((lon_locs.flatten(), lat_locs.flatten()))
    ref_flat = ref_data.flatten()
    rel_flat = reliability.flatten()

    # Interpolate radar data and reliability onto the master grid
    interp_ref = griddata(points, ref_flat, (grid_lon2d, grid_lat2d), method='linear')
    interp_rel = griddata(points, rel_flat, (grid_lon2d, grid_lat2d), method='linear')
    
    # Where new data is available and has a better (lower) reliability value, update the master grid
    valid_mask = ~np.isnan(interp_rel)
    update_mask = valid_mask & (interp_rel < combined_rel)
    combined_ref[update_mask] = interp_ref[update_mask]
    combined_rel[update_mask] = interp_rel[update_mask]
    
    processed_station_data.append([station_lon, station_lat, icao])


# Create geographic plot using the combined grid
fig, ax = plt.subplots(1, 1, figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
mesh = ax.pcolormesh(grid_lon2d, grid_lat2d, combined_ref, cmap=ref_cmap, norm=ref_norm,
                     shading='auto', transform=ccrs.PlateCarree(), alpha=0.8)

if show_stations:
  for station in processed_station_data:
      station_lon, station_lat, icao = station
      ax.plot(station_lon, station_lat, marker='o', color='blue', markersize=5, transform=ccrs.PlateCarree())
      ax.text(station_lon, station_lat, icao, fontsize=8, transform=ccrs.PlateCarree(),
              verticalalignment='bottom', color='blue')

# Add map features and gridlines
ax.add_feature(cfeature.BORDERS, linestyle=":", edgecolor="black")
ax.add_feature(cfeature.COASTLINE, edgecolor="black")
ax.add_feature(cfeature.STATES, linestyle="--", edgecolor="gray")
gl = ax.gridlines(draw_labels=True, linestyle="--", linewidth=0.5, color="gray")
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 10, 'color': 'black'}
gl.ylabel_style = {'size': 10, 'color': 'black'}
ax.set_xlim(lon_min, lon_max)
ax.set_ylim(lat_min, lat_max)

plt.suptitle("Composite Radar Reflectivity from Overlapping Nexrad Stations", fontsize=20)
plt.colorbar(mesh, ax=ax, label='Reflectivity (dBZ)')
plt.tight_layout()
plt.savefig('nexrad_composite.png', dpi=300, bbox_inches='tight')
