import os
import tempfile
import pytz
import numpy as np
import matplotlib.pyplot as plt
import nexradaws
import gzip
import datetime
from metpy.io import Level2File
from metpy.plots import ctables, add_timestamp
from mpl_toolkits.axes_grid1 import make_axes_locatable
import subprocess
from metpy.cbook import get_test_data

# -------------------- STEP 1: SEARCH FOR SCANS ON AWS --------------------


temp_dir = "tempnexrad"

dt_buffer = datetime.timedelta(hours=2)
def get_nexrad_gz_backup(dt, radar_id):

    os.makedirs(temp_dir, exist_ok=True)

    safe_str = dt.strftime("%Y-%m-%d_%H-%M-%S") + radar_id
    new_temp_path = os.path.join(temp_dir, safe_str)
    os.makedirs(new_temp_path, exist_ok=True)

    # Connect to the NEXRAD AWS Interface  
    conn = nexradaws.NexradAwsInterface()

    # Define search parameters (Date, Radar ID)
    start_time = dt - dt_buffer
    end_time   = dt + dt_buffer

    # Get available scans in the selected time range
    scans = conn.get_avail_scans_in_range(start_time, end_time, radar_id)

    if not scans:
      raise ValueError("No scans found!")

    def parse_scan_time(scan):
        # If scan.scan_time is already a datetime, use it directly
        # otherwise parse the ISO string:
        return scan.scan_time

    selected_scan = min(scans, key=lambda s: abs(parse_scan_time(s) - dt))
    print(f"Selected scan: {selected_scan}")

    filename = str(selected_scan).split("/")[-1].replace(">", "")
    return get_test_data(filename, as_file_obj=False)


if __name__ == "__main__":

    dt = datetime.datetime(2006, 12, 4, 1, 30, 0, tzinfo=pytz.UTC)
    radar_id = "KAMA"

    file_path = get_nexrad_gz_backup(dt, radar_id)
    radar_data = Level2File(file_path)

    # Extract scan data
    sweep = 0
    azimuths = np.array([ray[0].az_angle for ray in radar_data.sweeps[sweep]])

    fields = ['REF', 'RHO', 'ZDR', 'PHI']
    data_dict = {}

        
    for field in fields:
        valid_hdr = None
        valid_field_data = []
        for ray in radar_data.sweeps[sweep]:
            field_tuple = ray[4].get(field.encode())
            if field_tuple is not None and len(field_tuple) >= 2:
                if valid_hdr is None:
                    valid_hdr = field_tuple[0]
                valid_field_data.append(field_tuple[1])
        if valid_hdr is not None:
            field_range = np.arange(valid_hdr.num_gates) * valid_hdr.gate_width + valid_hdr.first_gate
            data_dict[field] = (field_range, np.array(valid_field_data))
        else:
            print(f"Warning: Field {field} not found in any ray.")
            data_dict[field] = None


    # -------------------- STEP 5: PLOT RADAR DATA --------------------

    # Get MetPy's NWS reflectivity colormap
    ref_norm, ref_cmap = ctables.registry.get_with_steps('NWSReflectivity', 5, 5)

    # Set up figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    colormaps = {'REF': ref_cmap, 'RHO': 'plasma', 'ZDR': 'viridis', 'PHI': 'viridis'}
    labels = {'REF': 'Reflectivity (dBZ)', 'RHO': 'Correlation Coefficient', 
            'ZDR': 'Differential Reflectivity (dBZ)', 'PHI': 'Differential Phase'}

    for field, ax in zip(fields, axes.flatten()):
        if data_dict[field]:
            field_range, field_data = data_dict[field]

            # Convert azimuth + range to Cartesian x, y
            xlocs = field_range * np.sin(np.deg2rad(azimuths[:, np.newaxis]))
            ylocs = field_range * np.cos(np.deg2rad(azimuths[:, np.newaxis]))

            # Mask invalid data
            data = np.ma.array(field_data)
            data[np.isnan(data)] = np.ma.masked

            # Plot data
            cmap = colormaps[field]
            norm = ref_norm if field == 'REF' else None
            mesh = ax.pcolormesh(xlocs, ylocs, data, cmap=cmap, norm=norm)

            # Add colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(mesh, cax=cax, orientation='vertical', label=labels[field])

            ax.set_aspect('equal', 'datalim')
            ax.set_xlim(-100, 100)
            ax.set_ylim(-100, 100)
            add_timestamp(ax, radar_data.dt, y=0.02, high_contrast=False)
            ax.set_title(labels[field])

    # Save and display
    plt.suptitle(f"NEXRAD Level 2 Data - {radar_id}", fontsize=20)
    plt.tight_layout()
    plt.savefig("nexrad_plot.png")

