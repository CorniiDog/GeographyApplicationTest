# [CTRL] + [SHIFT] + [P]
# Select Interpreter
# Create a .venv within the project
# pip install -r requirements.txt
# Documentation: https://blaylockbk.github.io/goes2go/_build/html/

# To allow vpn service:
# sudo nano /etc/openvpn/server.conf 
# or nano /etc/openvpn/client/airvpn.conf
#
# Add at bottom of the config for the openvpn service:
"""
route nwws-oi-cprk.weather.gov 255.255.255.255 net_gateway
route api.weather.gov 255.255.255.255 net_gateway
route www.ncdc.noaa.gov 255.255.255.255 net_gateway
route tgftp.nws.noaa.gov 255.255.255.255 net_gateway
route nomads.ncep.noaa.gov 255.255.255.255 net_gateway
route www.noaa.gov 255.255.255.255 net_gateway
route ocean.weather.gov 255.255.255.255 net_gateway
route naturalearth.s3.amazonaws.com 255.255.255.255 net_gateway
route amazonaws.com 255.255.255.255 net_gateway
"""

# To run this in background so that it works while the ssh session is terminated:
# ---
#
# python main.py > output.log 2>&1 & disown

# To see storage in current dir:
# ---
#
# sudo apt install exa -y
# exa -la --icons

# To view running processes easily:
# ---
#
# sudo apt install htop
# htop

# alternative way to view running processes easily:
# ---
#
# sudo apt install btop
# btop

# Other holdings for the future:
# https://www.ncei.noaa.gov/access/homr/

# https://www.ncei.noaa.gov/access/homr/file/nexrad-stations.txt

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import contextily as ctx
import geopandas as gpd
from goes2go.data import goes_nearesttime, goes_latest
from toolbox.cartopy_tools_OLD import common_features, pc
from toolbox.wind import spddir_to_uv
from paint.standard2 import cm_wind
import matplotlib.patheffects as path_effects
from scipy.ndimage import zoom
import cv2
from cv2 import dnn_superres
import xarray as xr
from shapely.geometry import Point
import datetime
from rasterio.transform import rowcol
import json
import os
import pyart

import forecast_database

dir_to_use = "/mnt/LightDir"

os.makedirs(dir_to_use, exist_ok=True)

forecast_database.DB_PATH = "forecasts.db"
forecast_database.FORECAST_DIR = os.path.join(dir_to_use, "forecasts")
forecast_database.USE_CACHE = True
forecast_database.VALIDATE_DATA = True # Enable to True for quick validation of incoming data


# Find bounding box of a state
state_of_interest = "Texas"

# 'state' : If region is a state
# 'country' : If region is a country
state_type = "state"

# 'C': Contiguous United States (alias 'CONUS')
# 'F': Full Disk (alias 'FULL')
# 'M': Mesoscale (alias 'MESOSCALE')
satellite_domain = 'C' 

latlon_additional_buffer = 0.1

# Load GOES-16 data
#dt = datetime.datetime(2020, 11, 16, 18, 0, 0)
#dt = datetime.datetime(2021,2, 1, 8, 55) # 2021 winter storm
#dt = datetime.datetime(2022,2, 1, 8, 55)
dt = datetime.datetime(2004,12, 1, 8, 55)
#dt = datetime.datetime(2024, 12, 29, 18, 0) # 2024 storm
#dt = datetime.datetime.now()

buffer_time = pd.Timedelta(minutes=10)

show_state_outlines = True
show_roads = True
show_wind = True
show_city_names = True
show_weather_stations = True

news_header = 'temperature'

# All RGB Recipes: https://blaylockbk.github.io/goes2go/_build/html/reference_guide/index.html#rgb-recipes
# Useful recipes for clouds: 'DayCloudPhase' 'NighttimeMicrophysics' 'DayNightCloudMicroCombo'
# Other recipes: AirMass Ash DayCloudConvection DayCloudPhase DayConvection DayLandCloud DayLandCloudFire
# DaySnowFog DifferentialWaterVapor Dust FireTemperature NaturalColor NightFogDifference NighttimeMicrophysics
# RocketPlume SulfurDioxide TrueColor WaterVapor
rgb_recipe = 'NaturalColor'
cities_border_buffer_pct = 0.1
pct_dark_to_consider_night = 0.8 # For 'DayNightCloudMicroCombo'


## Find and locate state information
with open("state_bboxes.json", "r") as f:
    state_bboxes = json.load(f)

state_found = False
for state_info in state_bboxes:
    if state_type in state_info.keys():
        if state_info[state_type].lower() == state_of_interest.lower():
            bounds = state_info["bounds"]
            lon_min, lat_min, lon_max, lat_max = bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]
            state_found = True
            print(f"Identified {state_info[state_type]}.", "Bounds:", state_info["bounds"])

if not state_found:
    raise(f"State not found: {state_of_interest}")

g = goes_nearesttime(dt, product='ABI', satellite='goes16', domain=satellite_domain)
# More info: https://www.star.nesdis.noaa.gov/goes/documents/ABIQuickGuide_DayNightCloudMicroCombo.pdf

def compute_darkness(img_rgb):
    return 1 - np.nanmean(np.max(img_rgb, axis=2))

if rgb_recipe == 'DayNightCloudMicroCombo':
    pct_black = compute_darkness(g.rgb.NaturalColor())
    print("Pct black:", f"{pct_black:.5}%")
    if pct_black > pct_dark_to_consider_night:
        rgb_recipe = 'NighttimeMicrophysics'
    else:
        rgb_recipe = 'DayCloudPhase'

lon_min -= latlon_additional_buffer
lat_min -= latlon_additional_buffer
lon_max += latlon_additional_buffer
lat_max += latlon_additional_buffer

middle_lon = np.average([lon_max, lon_min])
middle_lat = np.average([lat_max, lat_min])


############################## TODO: Delete 
# now = datetime.datetime.now() - datetime.timedelta(days=365)
# datetimes = [now - datetime.timedelta(days=365 * i) for i in range(21)][::-1]  # -20 years to 0
# for hist_dt in datetimes:
#     weater_data = forecast_database.get_nearest_station_dt_data(hist_dt, lat_min, lat_max, lon_min, lon_max, timedelta=buffer_time, unique=True)
##############################


bbox_width = np.abs(lon_max - lon_min)
bbox_height = np.abs(lat_min - lat_max)
buffer_width = bbox_width * cities_border_buffer_pct
buffer_height = bbox_height  * cities_border_buffer_pct

def is_recent(dt, days=120):
    return dt >= datetime.datetime.now() - datetime.timedelta(days=days)

days = 120
if is_recent(dt, days):
    print(f"The date {dt} must be at least {days} days from now to obtain weather. Disabling weather station options.")
    show_weather_stations = False

if show_weather_stations:
    weater_data = forecast_database.get_nearest_station_dt_data(dt, lat_min, lat_max, lon_min, lon_max, timedelta=buffer_time, unique=True)


fig = plt.figure(figsize=(15, 12))

# Format date
str_date_16 = f"{pd.to_datetime(g.time_coverage_start.item()):%d-%b-%Y %H:%M UTC}"

# Create axis with Geostationary projection (for wide view)
ax16_wide = fig.add_subplot(1, 2, 1, projection=g.rgb.crs)

# Create zoomed-in axis using a traditional projection (PlateCarree for lat/lon alignment)
ax16_zoom = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())

# Add coastlines, state borders, and country borders
common_features('50m', ax=ax16_wide, STATES=show_state_outlines, color='white', dark=True)
common_features('10m', ax=ax16_zoom, STATES=show_state_outlines, ROADS=show_roads, color='white', dark=True)

img_rgb = getattr(g.rgb, rgb_recipe)()

# Wide view: GOES-East Image
ax16_wide.set_title(f'GOES-East View: {rgb_recipe}', loc='left', fontweight='bold')
ax16_wide.set_title(f'{str_date_16}', loc='right')
ax16_wide.imshow(img_rgb, **g.rgb.imshow_kwargs)

# Zoomed-in view of Texas (corrected projection)
ax16_zoom.set_title(f'Texas Region in True Map Projection', loc='center')
ax16_zoom.imshow(img_rgb, transform=g.rgb.crs, extent=[g.rgb.x.min(), g.rgb.x.max(), g.rgb.y.min(), g.rgb.y.max()], origin='upper')

# Set map extent to Texas using a traditional lat/lon-based projection
ax16_zoom.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

# 2. Add state and country names dynamically
ax16_zoom.add_feature(cfeature.BORDERS, edgecolor='black', linewidth=1)
ax16_zoom.add_feature(cfeature.LAND)
text = ax16_zoom.text(middle_lon, middle_lat, s=state_of_interest, transform=ccrs.PlateCarree(), fontsize=15, color="white", weight="bold", zorder=10, ha='center', va='center')

text.set_path_effects([
    path_effects.Stroke(linewidth=2, foreground='black'),  # White outline
    path_effects.Normal()  # Normal text rendering
])


# 3. Load city data and add city labels
if show_city_names:
    cities_file = os.path.join("populated_places", "ne_10m_populated_places.shp")
    cities = gpd.read_file(cities_file)

    # Create a GeoDataFrame row for Houston
    houston = gpd.GeoDataFrame({
        'NAME': ['Houston'],
        'geometry': [Point(-95.3698, 29.7604)]
    }, crs=cities.crs)

    # Append Houston to the Texas cities GeoDataFrame
    cities = pd.concat([cities, houston], ignore_index=True)

    texas_cities = cities[(cities.geometry.x >= lon_min + buffer_width) & (cities.geometry.x <= lon_max - buffer_width) & 
                        (cities.geometry.y >= lat_min + buffer_height) & (cities.geometry.y <= lat_max - buffer_height)]

    for _, row in texas_cities.iterrows():
        text = ax16_zoom.text(row.geometry.x, row.geometry.y, row['NAME'], transform=ccrs.PlateCarree(),
                    fontsize=6, color="white", weight="bold", zorder=10, ha='center', va='center')
        text.set_path_effects([
            path_effects.Stroke(linewidth=2, foreground='black'),  # White outline
            path_effects.Normal()  # Normal text rendering
        ])
    
if show_weather_stations:
    for key in weater_data.keys():
        print(key)

    for _, row in weater_data.iterrows():
        text = ax16_zoom.text(float(row['Longitude']), float(row['Latitude']), str(row[news_header]), transform=ccrs.PlateCarree(),
                    fontsize=5, color="yellow", weight="bold", zorder=9, ha='center', va='center')
        text.set_path_effects([
            path_effects.Stroke(linewidth=1, foreground='black'),  # White outline
            path_effects.Normal()  # Normal text rendering
        ])

# Draw bounding box on wide view
left, right, bottom, top = ax16_zoom.get_extent()
lons = [left, right, right, left, left]
lats = [top, top, bottom, bottom, top]
ax16_wide.plot(lons, lats, transform=ccrs.PlateCarree())

if dt < datetime.datetime(2021, 5, 1):
    print(f"The date {dt} must be after may 2021 to retreive wind barb information. Disabling wind barbs.")
    show_wind = False

if show_wind:
    gwnd = goes_nearesttime(dt, product='ABI-L2-DMWVC')


    gu, gv = spddir_to_uv(gwnd.wind_speed, gwnd.wind_direction)

    ax16_wide.barbs(
        gwnd.lon.data,
        gwnd.lat.data,
        gu.data,
        gv.data,
        gwnd.wind_speed,
        **cm_wind().cmap_kwargs,
        length=4,
        transform=pc,
    )

    ax16_zoom.barbs(
        gwnd.lon.data,
        gwnd.lat.data,
        gu.data,
        gv.data,
        gwnd.wind_speed,
        **cm_wind().cmap_kwargs,
        length=5,
        transform=pc,
    )

gl = ax16_zoom.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                         linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False

# Adjust layout
plt.subplots_adjust(wspace=0.01)

print("Outputting...")
plt.savefig('output.png', dpi=300, bbox_inches='tight')
#plt.show()
