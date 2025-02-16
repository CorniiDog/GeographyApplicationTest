# [CTRL] + [SHIFT] + [P]
# Select Interpreter
# Create a .venv within the project
# pip install -r requirements.txt
# Documentation: https://blaylockbk.github.io/goes2go/_build/html/

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



# Define the bounding box in lat/lon (Texas region)
lon_min, lat_min, lon_max, lat_max = -106.64719063660635, 25.840437651866516, -93.5175532104321, 36.50050935248352

latlon_additional_buffer = 0.1
# Load GOES-16 data
#dt = datetime.datetime(2020, 11, 16, 18, 0, 0)
# dt = datetime.datetime(2021,2, 1, 8, 55) # 2021 winter storm
#dt = datetime.datetime(2024, 12, 29, 18, 0) # 2024 storm
dt = datetime.datetime.now()

show_state_outlines = True
show_roads = True
show_wind = True
show_city_names = True

# All RGB Recipes: https://blaylockbk.github.io/goes2go/_build/html/reference_guide/index.html#rgb-recipes
# Useful recipes for clouds: 'DayCloudPhase' 'NighttimeMicrophysics' 'DayNightCloudMicroCombo'
rgb_recipe = 'DayNightCloudMicroCombo'
cities_border_buffer_pct = 0.1
pct_dark_to_consider_night = 0.8 # For 'DayNightCloudMicroCombo'

g = goes_nearesttime(dt, product='ABI', satellite='goes16', domain='F')
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


bbox_width = np.abs(lon_max - lon_min)
bbox_height = np.abs(lat_min - lat_max)
buffer_width = bbox_width * cities_border_buffer_pct
buffer_height = bbox_height  * cities_border_buffer_pct

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
text = ax16_zoom.text(-99.5, 31, s="Texas", transform=ccrs.PlateCarree(), fontsize=15, color="white", weight="bold", zorder=10, ha='center', va='center')

text.set_path_effects([
    path_effects.Stroke(linewidth=2, foreground='black'),  # White outline
    path_effects.Normal()  # Normal text rendering
])

# 3. Load city data and add city labels
cities = gpd.read_file('populated_places\\ne_10m_populated_places.shp')

# Create a GeoDataFrame row for Houston
houston = gpd.GeoDataFrame({
    'NAME': ['Houston'],
    'geometry': [Point(-95.3698, 29.7604)]
}, crs=cities.crs)


if show_city_names:

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
    


# Draw bounding box on wide view
left, right, bottom, top = ax16_zoom.get_extent()
lons = [left, right, right, left, left]
lats = [top, top, bottom, bottom, top]
ax16_wide.plot(lons, lats, transform=ccrs.PlateCarree())

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
plt.show()
