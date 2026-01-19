import geopandas as gpd
import os


# Read the shapefile
gdf = gpd.read_file(os.path.join(os.path.dirname(__file__), r"DATA\RAW\Scottish_Govt_DataZoneCentroids_2022\SG_DataZone_Cent_2022.shp"))

# Print the first few rows of the GeoDataFrame
print(gdf.head())





