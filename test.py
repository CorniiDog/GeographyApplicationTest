from goes2go.data import goes_latest, goes_nearesttime
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# Download a GOES ABI dataset
G = goes_nearesttime('2020-9-22 18', satellite=17)
# Make figure on Cartopy axes
ax = plt.subplot(projection=G.rgb.crs )
ax.imshow(G.rgb.TrueColor(night_IR=False), **G.rgb.imshow_kwargs)
ax.coastlines()

plt.show()