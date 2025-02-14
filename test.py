from goes2go.data import goes_nearesttime
from toolbox.cartopy_tools_OLD import common_features
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import datetime
import imageio
import os
import pandas as pd
import numpy as np
import matplotlib.patheffects as path_effects
from shapely.geometry import Point
import geopandas as gpd
import io

# Set up parameters

gif_duration_seconds = 2
end_dt = datetime.datetime.now()
start_dt = end_dt - datetime.timedelta(days=1)
step_delta = datetime.timedelta(hours=1)

datetimes = [end_dt]
while datetimes[-1] > start_dt:
    datetimes.append(datetimes[-1] - step_delta)
datetimes = datetimes[::-1]  # Reverse to get chronological order from start_dt to end_dt

# Directory to store frames
frame_dir = "frames"
os.makedirs(frame_dir, exist_ok=True)

frames = []

for i, dt in enumerate(datetimes):
    G = goes_nearesttime(dt, product='ABI', satellite='goes16', domain='C')    

    # Create figure
    fig, ax = plt.subplots(subplot_kw={'projection': G.rgb.crs})
    ax.imshow(G.rgb.TrueColor(night_IR=True), **G.rgb.imshow_kwargs)
    ax.coastlines()
    common_features('50m', ax=ax, STATES=True, color='white', dark=True)


    # Identify nearest time
    actual_time = pd.to_datetime(G.time_coverage_start.item())
        
    ax.text(
        0.05, 0.05,  # Position (X, Y) in figure coordinates
        actual_time.strftime("%Y-%m-%d %H:%M:%S UTC"),  # Format datetime
        transform=ax.transAxes,
        fontsize=12,
        color='white',
        bbox=dict(facecolor='black', alpha=0.7, edgecolor='none')
    )

    cities = gpd.read_file('populated_places\\ne_10m_populated_places.shp')


    # Create a GeoDataFrame row for Houston
    texas_cities = gpd.GeoDataFrame({
        'NAME': ['Texas'],
        'geometry': [Point(-99.5, 31)]
    }, crs=cities.crs)


    for _, row in texas_cities.iterrows():
        text = ax.text(row.geometry.x, row.geometry.y, row['NAME'], transform=ccrs.PlateCarree(),
                      fontsize=8, color="black", weight="bold", zorder=10, ha='center', va='center')
        text.set_path_effects([
            path_effects.Stroke(linewidth=2, foreground='white'),  # White outline
            path_effects.Normal()  # Normal text rendering
        ])
    
    # Save figure to an in-memory buffer.
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    frame = imageio.imread(buf)
    frames.append(frame)
    buf.close()
    plt.close(fig)

# Create GIF directly from frames in memory.
duration_per_frame = 1000 * gif_duration_seconds / len(frames)
output_gif = "goes_animation.gif"
imageio.mimsave(output_gif, frames, format='GIF', duration=duration_per_frame, loop=0)

print(f"GIF saved as {output_gif}")