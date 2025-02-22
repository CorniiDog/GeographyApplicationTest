import requests
import datetime
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from metpy.plots import ctables
from scipy.interpolate import griddata
from multiprocessing import Pool, cpu_count, BoundedSemaphore
import util_toolbox

# ----- User Configuration -----
# Select parameter: 'REF', 'RHO', 'PHI', or 'ZDR'
SELECTED_PARAM = 'REF'  # Change as needed

# Get the NWS reflectivity colortable from MetPy (only called once)
ref_norm, ref_cmap = ctables.registry.get_with_steps('NWSReflectivity', 5, 5)

# Determine plot colormap and normalization based on the selected parameter.
if SELECTED_PARAM == 'REF':
    plot_cmap = ref_cmap
    plot_norm = ref_norm
    plot_label = 'Reflectivity (dBZ)'
    units = ' dBZ'
elif SELECTED_PARAM == 'RHO':
    plot_cmap = 'plasma'
    plot_norm = None
    plot_label = 'RHO'
    units = ''
elif SELECTED_PARAM == 'ZDR':
    plot_cmap = 'viridis'
    plot_norm = None
    plot_label = 'Differential Reflectivity (dB)'
    units = ' dB'
elif SELECTED_PARAM == 'PHI':
    plot_cmap = 'viridis'
    plot_norm = None
    plot_label = 'Differential Phase (°)'
    units = '°'
else:
    plot_cmap = ref_cmap
    plot_norm = ref_norm
    plot_label = SELECTED_PARAM

plot_title = f"Composite Radar {plot_label} from Overlapping Nexrad Stations"

# Global configuration and constants
BUCKET_NAME = "noaa-nexrad-level2"
NEXRAD_STATION_LIST_URL = "https://www.ncei.noaa.gov/access/homr/file/nexrad-stations.txt"
dt = datetime.datetime(2007, 8, 27, 1, 35, 2)
lon_min, lat_min, lon_max, lat_max = -106.647, 25.840, -93.517, 36.500
show_stations = True

# Option: set the percentage of available cores to use (e.g., 0.5 for 50%)
CORE_PERCENTAGE = 0.9
num_processes = max(1, int(cpu_count() * CORE_PERCENTAGE))
print(f"Using {num_processes} out of {cpu_count()} cores.")

# Create master grid for composite plotting
grid_res = 0.01  # degrees; adjust resolution as needed
grid_lon = np.arange(lon_min, lon_max, grid_res)
grid_lat = np.arange(lat_min, lat_max, grid_res)
grid_lon2d, grid_lat2d = np.meshgrid(grid_lon, grid_lat)
combined_param = np.full(grid_lon2d.shape, np.nan)
combined_rel = np.full(grid_lon2d.shape, np.inf)  # lower value means higher reliability

# Fetch the list of NEXRAD stations (single HTTP request)
response = requests.get(NEXRAD_STATION_LIST_URL)
lines = response.text.splitlines()

# Parse station metadata from file using fixed-width indices
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

# Filter stations within the bounding box
filtered_sites = []
for site in radar_sites:
    try:
        lat, lon = float(site["LAT"]), float(site["LON"])
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            print("Site Found:", site['ICAO'])
            filtered_sites.append(site)
    except ValueError:
        continue

if not filtered_sites:
    print("No radar sites found in the bounding box.")
    exit()

processed_station_data = []

# Global semaphore to limit concurrent HTTP requests.
def init_pool(sem):
    global http_semaphore
    http_semaphore = sem
    

def process_station(station, grid_lon2d, grid_lat2d):
    """
    Processes a single station's radar data by retrieving and interpolating the
    selected parameter onto the master grid. Returns a tuple
    (interp_param, interp_rel, station_lon, station_lat, icao) if successful.
    """
    try:
        station_lat = float(station['LAT'])
        station_lon = float(station['LON'])
    except ValueError:
        return None
    icao = station['ICAO']

    # Set up S3 clients in this process
    s3_client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    s3_resource = boto3.resource('s3', config=Config(signature_version=UNSIGNED, user_agent_extra='Resource'))
    bucket = s3_resource.Bucket(BUCKET_NAME)

    prefix = f"{dt.year}/{dt.month:02d}/{dt.day:02d}/{icao}/"
    with http_semaphore:
        response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)
    if "Contents" not in response:
        print(f"No content from response for {prefix}")
        return None
    
    available_files = []
    #print(response.get("Contents", []))
    for obj in response.get("Contents", []):
        key = obj["Key"]
        parts = key.split("_")
        if len(parts) > 1 and parts[1].isdigit():
            available_files.append((key, int(parts[1])))
    if not available_files:
        print(f"Cannot find any available files for {prefix}")
        return None

    target_time = int(f"{dt.hour:02d}{dt.minute:02d}{dt.second:02d}")
    closest_file = min(available_files, key=lambda x: abs(x[1] - target_time))
    closest_key = closest_file[0]

    with http_semaphore:
        objs = list(bucket.objects.filter(Prefix=closest_key))
    if not objs:
        print(f"Cannot find objects for {obj.key}")
        return None
    obj = objs[0]

    from metpy.io import Level2File
    try:
        with http_semaphore:
            f = Level2File(obj.get()['Body'])
    except Exception as e:
        print(f"Error reading {obj.key}: {e}")
        return None

    if not f.sweeps or len(f.sweeps[0]) == 0:
        print(f"Returning because of fsweeps for {obj.key}")
        return None
    sweep = 0
    try:
        az = np.array([ray[0].az_angle for ray in f.sweeps[sweep]])
    except Exception as e:
        print(f"Error extracting azimuth angles from {obj.key}: {e}")
        return None

    try:
        param_bytes = SELECTED_PARAM.encode('utf-8')
        hdr = f.sweeps[sweep][0][4][param_bytes][0]
        # if SELECTED_PARAM == 'ZDR':
        #     print(dir(hdr))
        #     print(repr(hdr))


        if SELECTED_PARAM == 'REF':
            param_range = np.arange(hdr.num_gates) * hdr.gate_width + hdr.first_gate
        else:
            param_range = (np.arange(hdr.num_gates + 1) - 0.5) * hdr.gate_width + hdr.first_gate
        # Extract raw data
        param_data = np.array([ray[4][param_bytes][1] for ray in f.sweeps[sweep]])
        
    except Exception as e:
        print(f"Error extracting {SELECTED_PARAM} from {obj.key}: {e}")
        return None

    if param_data.size == 0 or np.all(np.isnan(param_data)):
        print(f"Missing data for {obj.key}")
        return None
    
    # Compute maximum value from this station's data (for outlier detection)
    max_val = np.nanmax(param_data)
    print(f"Station {obj.key} max {SELECTED_PARAM} value: {max_val:.2f}{units}")

    data = param_data.flatten()
    util_toolbox.text_box_plot(data)

    print(f"Processing station {obj.key} for parameter {SELECTED_PARAM}")

    from pyproj import Proj
    proj_station = Proj(proj="aeqd", datum="WGS84", lat_0=station_lat, lon_0=station_lon)
    xlocs = param_range * np.sin(np.deg2rad(az[:, np.newaxis]))
    ylocs = param_range * np.cos(np.deg2rad(az[:, np.newaxis]))

    x_diff = xlocs.shape[1] - param_data.shape[1]
    if x_diff > 0:
        xlocs = xlocs[:, :-x_diff]
        ylocs = ylocs[:, :-x_diff]  # Ensure ylocs is sliced similarly

    data_diff = param_data.shape[1] - xlocs.shape[1]
    if data_diff > 0:
        param_data = param_data[:, :-data_diff]

    lon_locs, lat_locs = proj_station(xlocs * 1000, ylocs * 1000, inverse=True)
    mask = (lon_locs < lon_min) | (lon_locs > lon_max) | (lat_locs < lat_min) | (lat_locs > lat_max)
    param_data = np.ma.masked_where(mask, param_data)
    if np.all(param_data.mask):
        print(f"No data within lat-lon range for {obj.key}")
        return None

    reliability = np.sqrt(xlocs**2 + ylocs**2)
    points = np.column_stack((lon_locs.flatten(), lat_locs.flatten()))
    param_flat = param_data.flatten()
    rel_flat = reliability.flatten()

    interp_param = griddata(points, param_flat, (grid_lon2d, grid_lat2d), method='linear')
    interp_rel = griddata(points, rel_flat, (grid_lon2d, grid_lat2d), method='linear')

    return (interp_param, interp_rel, station_lon, station_lat, icao)

if __name__ == '__main__':
    # Create a semaphore to limit concurrent HTTP requests (allow 4 at a time)
    http_sem = BoundedSemaphore(4)
    args = [(station, grid_lon2d, grid_lat2d) for station in filtered_sites]

    with Pool(num_processes, initializer=init_pool, initargs=(http_sem,)) as pool:
        results = pool.starmap(process_station, args)

    for res in results:
        if res is None:
            continue
        interp_param, interp_rel, station_lon, station_lat, icao = res
        valid_mask = ~np.isnan(interp_rel)
        update_mask = valid_mask & (interp_rel < combined_rel)
        combined_param[update_mask] = interp_param[update_mask]
        combined_rel[update_mask] = interp_rel[update_mask]
        processed_station_data.append([station_lon, station_lat, icao])
        print(f"Added station {icao} at ({station_lon}, {station_lat})")

    # Create geographic plot using the combined grid
    fig, ax = plt.subplots(1, 1, figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    mesh = ax.pcolormesh(grid_lon2d, grid_lat2d, combined_param, cmap=plot_cmap, norm=plot_norm,
                         shading='auto', transform=ccrs.PlateCarree(), alpha=0.8)

    if show_stations:
        for station in processed_station_data:
            station_lon, station_lat, icao = station
            ax.plot(station_lon, station_lat, marker='o', color='blue', markersize=5, transform=ccrs.PlateCarree())
            ax.text(station_lon, station_lat, icao, fontsize=8, transform=ccrs.PlateCarree(),
                    verticalalignment='bottom', color='blue')

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

    plt.suptitle(plot_title, fontsize=20)
    plt.colorbar(mesh, ax=ax, label=plot_label)
    plt.tight_layout()
    plt.savefig('nexrad_composite.png', dpi=300, bbox_inches='tight')
