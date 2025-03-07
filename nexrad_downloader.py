import requests
import datetime
import boto3
import pyart
from botocore import UNSIGNED
from botocore.config import Config
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from metpy.plots import add_timestamp, ctables
from mpl_toolkits.axes_grid1 import make_axes_locatable
from metpy.io import Level2File
from pyproj import Proj, Transformer

# AWS S3 bucket configuration
BUCKET_NAME = "noaa-nexrad-level2"
NEXRAD_STATION_LIST_URL = "https://www.ncei.noaa.gov/access/homr/file/nexrad-stations.txt"

# Fetch the list of NEXRAD stations
response = requests.get(NEXRAD_STATION_LIST_URL)
lines = response.text.splitlines()

# Extract headers (ignore the first two lines, which contain dashes)
indices = [
    (0, 8), (9, 13), (14, 19), (20, 50), (51, 71), (72, 74), (75, 105),
    (106, 115), (116, 126), (127, 133), (134, 139), (140, 188)
]
keys = ["NCDCID", "ICAO", "WBAN", "NAME", "COUNTRY", "STATE", "COUNTY",
        "LAT", "LON", "ELEV", "UTC", "STNTYPE"]

# Parse radar sites
radar_sites = []
for line in lines[3:]:
    entry = {key: line[start:end].strip() for key, (start, end) in zip(keys, indices)}
    radar_sites.append(entry)

# Define bounding box for filtering (adjust as needed)
lon_min, lat_min, lon_max, lat_max = -106.647, 25.840, -93.517, 36.500
dt = datetime.datetime(2022, 2, 1, 8, 55, 2)  # Example target datetime

# Filter radar sites based on bounding box
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

# Choose a radar site
test_site = filtered_sites[0]
radarsite = test_site['ICAO']

# Get the radar station's latitude and longitude
radar_lat = float(test_site["LAT"])
radar_lon = float(test_site["LON"])

# Define projection centered at the radar location
proj = Proj(proj="aeqd", datum="WGS84", lat_0=radar_lat, lon_0=radar_lon)

# Initialize S3 client (public NOAA bucket does not require authentication)
s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

# List files in the radar's folder for the given date
prefix = f"{dt.year}/{dt.month:02d}/{dt.day:02d}/{radarsite}/"
response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)

# Extract available timestamps from filenames
available_files = []
for obj in response.get("Contents", []):
    key = obj["Key"]
    parts = key.split("_")
    print(key)
    if len(parts) > 1 and parts[1].isdigit():
        available_files.append((key, int(parts[1])))

if not available_files:
    print(f"No radar data found for {radarsite} on {dt.date()}")
    exit()

# Find the closest available timestamp
target_time_str = f"{dt.hour:02d}{dt.minute:02d}{dt.second:02d}"
target_time = int(target_time_str)
closest_file = min(available_files, key=lambda x: abs(x[1] - target_time))
closest_key = closest_file[0]

s3 = boto3.resource('s3', config=Config(signature_version=UNSIGNED, user_agent_extra='Resource'))
bucket = s3.Bucket(BUCKET_NAME)
print("Closest key:", closest_key)
for obj in bucket.objects.filter(Prefix=closest_key):
    print(obj.key)

    # Use MetPy to read the file
    f = Level2File(obj.get()['Body'])

    # --- Check for valid sweeps before processing ---
    if not f.sweeps or len(f.sweeps[0]) == 0:
        print(f"Skipping file {obj.key}: no valid sweep data found.")
        continue

    sweep = 0
    try:
        # Extract azimuth angles; if a ray is missing data, this may raise an error.
        az = np.array([ray[0].az_angle for ray in f.sweeps[sweep]])
    except (IndexError, AttributeError) as e:
        print(f"Error extracting azimuth angles from {obj.key}: {e}")
        continue

    try:
        ref_hdr = f.sweeps[sweep][0][4][b'REF'][0]
        ref_range = np.arange(ref_hdr.num_gates) * ref_hdr.gate_width + ref_hdr.first_gate
        ref = np.array([ray[4][b'REF'][1] for ray in f.sweeps[sweep]])

        rho_hdr = f.sweeps[sweep][0][4][b'RHO'][0]
        rho_range = (np.arange(rho_hdr.num_gates + 1) - 0.5) * rho_hdr.gate_width + rho_hdr.first_gate
        rho = np.array([ray[4][b'RHO'][1] for ray in f.sweeps[sweep]])

        phi_hdr = f.sweeps[sweep][0][4][b'PHI'][0]
        phi_range = (np.arange(phi_hdr.num_gates + 1) - 0.5) * phi_hdr.gate_width + phi_hdr.first_gate
        phi = np.array([ray[4][b'PHI'][1] for ray in f.sweeps[sweep]])

        zdr_hdr = f.sweeps[sweep][0][4][b'ZDR'][0]
        zdr_range = (np.arange(zdr_hdr.num_gates + 1) - 0.5) * zdr_hdr.gate_width + zdr_hdr.first_gate
        zdr = np.array([ray[4][b'ZDR'][1] for ray in f.sweeps[sweep]])
    except (IndexError, KeyError) as e:
        print(f"Error extracting radar field data from {obj.key}: {e}")
        continue

    # Get the NWS reflectivity colortable from MetPy
    ref_norm, ref_cmap = ctables.registry.get_with_steps('NWSReflectivity', 5, 5)

    # Create figure and axes for plotting
    fig, axes = plt.subplots(2, 2, figsize=(15, 15), subplot_kw={'projection': ccrs.PlateCarree()})

    # Plot the data
    for var_data, var_range, colors, lbl, ax in zip(
            (ref, rho, zdr, phi),
            (ref_range, rho_range, zdr_range, phi_range),
            (ref_cmap, 'plasma', 'viridis', 'viridis'),
            ('REF (dBZ)', 'RHO', 'ZDR (dBZ)', 'PHI'),
            axes.flatten()):
        
        # Remove invalid data
        data = np.ma.array(var_data)
        data[np.isnan(data)] = np.ma.masked

        # Convert azimuth and range to x,y coordinates
        xlocs = var_range * np.sin(np.deg2rad(az[:, np.newaxis]))
        ylocs = var_range * np.cos(np.deg2rad(az[:, np.newaxis]))

        # Dynamically remove excess columns so shapes match
        x_diff = xlocs.shape[1] - data.shape[1]
        y_diff = ylocs.shape[1] - data.shape[1]
        data_diff = data.shape[1] - xlocs.shape[1]
        if x_diff > 0:
            xlocs = xlocs[:, :-x_diff]
        if y_diff > 0:
            ylocs = ylocs[:, :-y_diff]
        if data_diff > 0:
            data = data[:, :-data_diff]

        # Convert x,y (in kilometers) to latitude/longitude
        lon_locs, lat_locs = proj(xlocs * 1000, ylocs * 1000, inverse=True)  # km to meters

        # Apply bounding box filter
        mask = (lon_locs < lon_min) | (lon_locs > lon_max) | (lat_locs < lat_min) | (lat_locs > lat_max)
        data = np.ma.masked_where(mask, data)

        # Only apply reflectivity normalization for REF
        norm = ref_norm if lbl == "REF (dBZ)" else None

        # Plot data using pcolormesh
        a = ax.pcolormesh(lon_locs, lat_locs, data, cmap=colors, norm=norm, shading='auto',
                          transform=ccrs.PlateCarree())

        # Add map features
        ax.add_feature(cfeature.BORDERS, linestyle=":", edgecolor="black")
        ax.add_feature(cfeature.COASTLINE, edgecolor="black")
        ax.add_feature(cfeature.STATES, linestyle="--", edgecolor="gray")

        # Add gridlines with latitude/longitude labels
        gl = ax.gridlines(draw_labels=True, linestyle="--", linewidth=0.5, color="gray")
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 10, 'color': 'black'}
        gl.ylabel_style = {'size': 10, 'color': 'black'}

        # Set lat/lon bounds to the bounding box
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)

        # Add colorbar
        cbar = fig.colorbar(a, ax=ax, orientation='vertical', pad=0.05)
        cbar.set_label(lbl)

        add_timestamp(ax, f.dt, y=0.02, high_contrast=False)

    plt.suptitle(f'{radarsite} Level 2 Data', fontsize=20)
    plt.tight_layout()
    plt.savefig('nexrad_out.png', dpi=300, bbox_inches='tight')
