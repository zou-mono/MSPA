import gdal, ogr, osr, os, gdalnumeric
from shapely.geometry import shape
import numpy as np
from shapely.wkt import dumps, loads
from skimage import measure, io
from skimage.graph import route_through_array
import matplotlib.pyplot as plt
from PIL import Image
import cv2 as cv
from rasterio.mask import mask
import rasterio
import fiona

# img = cv.imread("D:/MSPA/i.tif", cv.IMREAD_LOAD_GDAL)

# srcArray = gdalnumeric.LoadFile("D:/MSPA/i2.tif")

driver = ogr.GetDriverByName('ESRI Shapefile')
dataSource = driver.Open('POLYGONIZED_STUFF.shp', 0)
layer = dataSource.GetLayer()
print(layer.GetFeatureCount())

driver = ogr.GetDriverByName('Esri Shapefile')
# ds = driver.CreateDataSource('result.shp')
# out_layer = ds.CreateLayer('', None, ogr.wkbPolygon)
# # Add one attribute
# out_layer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))
# defn = out_layer.GetLayerDefn()
# print(defn)

src = rasterio.open("result2.tif")
with fiona.open("POLYGONIZED_STUFF.shp", "r") as shapefile:
    shapes = [feature["geometry"] for feature in shapefile]

out_image, out_transform = mask(src, [shapes[1]], crop=True)
no_data=src.nodata

print(no_data)
row, col = np.where(out_image[0] != no_data)

plt.imshow(out_image[0])
plt.show()

for feature in layer:
    geom = feature.GetGeometryRef()
    geojson = geom.ExportToJson()
    # out_image, out_transform = mask(src, [geojson], crop=True)

    g = loads(geom.ExportToWkt())
    feat = geom = None
    # print(out)

layer.ResetReading()

print("over")
