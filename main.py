
# [CTRL] + [SHIFT] + [P]
# Select Interpreter
# Create a .venv within the project
# pip install -r requirements.txt
# Documentation: https://blaylockbk.github.io/goes2go/_build/html/

from goes2go.data import goes_nearesttime, goes_latest
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

# Function from Carpenter_Workshop: https://github.com/blaylockbk/Carpenter_Workshop
from toolbox.cartopy_tools_OLD import common_features

G16 = goes_nearesttime('2020-9-22 18')
G17 = goes_nearesttime('2020-9-22 18', satellite=17)

rgb_products = [i for i in dir(G16.rgb) if i[0].isupper()]

for product in rgb_products:

    fig = plt.figure(figsize=(15, 12))
    ax17 = fig.add_subplot(1, 2, 1, projection=G17.rgb.crs)
    ax16 = fig.add_subplot(1, 2, 2, projection=G16.rgb.crs)

    for ax, G in zip([ax17, ax16], [G17, G16]):
        RGB = getattr(G.rgb, product)()

        common_features('50m', STATES=True, ax=ax)
        ax.imshow(RGB, **G.rgb.imshow_kwargs)
        ax.set_title(f"{G.orbital_slot} {product}", loc='left', fontweight='bold')
        ax.set_title(f"{G.t.dt.strftime('%H:%M UTC %d-%b-%Y').item()}", loc="right")
    plt.subplots_adjust(wspace=0.01)
    plt.savefig(f'docs/{product}', bbox_inches='tight')