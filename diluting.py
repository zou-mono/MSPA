import sys
from osgeo import gdal, gdalnumeric, ogr, osr
import numpy as np
from shapely.geometry import shape
from shapely.wkt import dumps, loads
from skimage import measure, io
from skimage.graph import route_through_array
import matplotlib.pyplot as plt
import cv2 as cv
from rasterio.mask import mask
import rasterio
import fiona
import time
import copy

def main():
    dataset = gdal.Open("D:/MSPA/i2.tif")

    srcband = dataset.GetRasterBand(1)
    projectionfrom = dataset.GetProjection()
    geotransform = dataset.GetGeoTransform()
    xsize = srcband.XSize
    ysize = srcband.YSize

    rasterArray = srcband.ReadAsArray()

    gtiff = gdal.GetDriverByName('GTiff')
    output_dataset = gtiff.Create('ttt.tif', xsize, ysize, 2, gdal.GDT_Byte)
    output_dataset.SetProjection(projectionfrom)
    output_dataset.SetGeoTransform(geotransform)

    # 修改掩膜
    # rasterArray[rasterArray == 1] = 2
    # rasterArray[rasterArray == 0] = 1
    # rasterArray[rasterArray == 2] = 0
    output_dataset.GetRasterBand(1).WriteArray(rasterArray)
    output_dataset.GetRasterBand(2).SetNoDataValue(0) # 只能在创建的时候使用
    output_dataset.GetRasterBand(2).WriteArray(rasterArray)  # mask图层
    output_dataset.FlushCache()

    res_dataset = gdal.Open("ttt.tif")
    out_band = res_dataset.GetRasterBand(1)
    dst_layername = "POLYGONIZED_STUFF"
    drv = ogr.GetDriverByName("ESRI Shapefile")
    dst_ds = drv.CreateDataSource(dst_layername + ".shp")
    dst_layer = dst_ds.CreateLayer(dst_layername, srs=None)
    # dst_layer.CreateField(ogr.FieldDefn('v', ogr.OFTInteger))

    gdal.Polygonize(out_band, res_dataset.GetRasterBand(2), dst_layer, -1, [], callback=None)
    dst_ds.Destroy()

    driver = ogr.GetDriverByName('ESRI Shapefile')
    dataSource = driver.Open('POLYGONIZED_STUFF.shp', 0)
    layer = dataSource.GetLayer(0)
    print(layer.GetFeatureCount())

    # 提取轮廓
    dataset = gdal.Open("D:/MSPA/i2.tif")
    srcband = dataset.GetRasterBand(1)
    rasterArray = srcband.ReadAsArray()

    res = np.zeros((ysize, xsize), dtype=np.int)
    contours, heriachy = cv.findContours(rasterArray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE) # RETR_EXTERNAL RETR_TREE RETR_FLOODFILL  RETR_EXTERNAL

    for n, contour in enumerate(contours):
        for i in range(contour.shape[0]):
            res[int(contour[i][0][1])][int(contour[i][0][0])] = 1
            # cv.fillPoly(res, pts=contour, color=(255,255,255))

    # cv.drawContours(res, contours, -1, (255, 255, 255), thickness=cv.FILLED)
    # cv.imwrite('contours.jpg', img)
    array2raster('result2.tif', 'D:/MSPA/i2.tif', res, gdal.GDT_Byte)

    # 两个像素点之间的最短路径
    CostSurfacefn = 'D:\MSPA\\cost_test.tif' #'D:\MSPA\\background4.tif' background4.tif D:\MSPA\\cost_test1.tif
    outputPathfn = 'Path2.tif'

    costSurfaceArray = raster2array(CostSurfacefn)  # creates array from cost surface raster

    # 根据栅格边缘提取对应矩阵元素位置
    src = rasterio.open("result2.tif")
    visited = []
    with fiona.open("POLYGONIZED_STUFF.shp", "r") as shapefile:
        shapes = [feature["geometry"] for feature in shapefile]
        # jshapes = shapes.copy()
        res_array = np.zeros_like(costSurfaceArray)

        for i in range(len(shapes)):
            # shape = shapes.pop(0)
            visited.append(i)

            start_image, start_transform = mask(src, [shapes[i]], crop=False)
            no_data = src.nodata
            start_row, start_col = np.where(start_image[0] != no_data)
            if len(start_row) == 0:
                continue

            for j in range(len(shapes)):
                if i == j:
                    continue
                if j in visited:
                    continue
                # if j != 557:
                #     continue

                end_image, end_transform = mask(src, [shapes[j]], crop=False)
                no_data = src.nodata
                end_row, end_col = np.where(end_image[0] != no_data)
                if len(end_row) == 0:
                    continue

                minWeight = sys.maxsize
                minPathArray = []

                # start_sep = diluting(start_row, 1)
                # end_sep = diluting(end_row, 1)
                for m in range(int(len(start_row) / 1000) + 1):
                    startCoord = (start_row[m * 1000 - 1], start_col[m * 1000 - 1])
                    for n in range(int(len(end_row) / 1000) + 1):
                        endCoord = (end_row[n * 1000 - 1], end_col[n * 1000 - 1])

                        start = time.time()
                        pathArray, weight = createPath(CostSurfacefn, costSurfaceArray, startCoord, endCoord)  # creates path array
                        end = time.time()
                        print("Execution Time: ", end - start)

                        if weight < 0:
                            continue

                        if weight < minWeight:
                            minWeight = weight
                            minPathArray = pathArray

                if minWeight < sys.maxsize:
                    res_array = minPathArray + res_array

                print(str(i) + "," + str(j))

                if i == 0 and j == 555:
                    res_array[res_array > 0] = 1
                    array2raster(outputPathfn, CostSurfacefn, res_array, gdal.GDT_Byte)

        array2raster(outputPathfn, CostSurfacefn, res_array, gdal.GDT_Byte)  # converts path array to raster

    print("Over")


def diluting(num, iMax):
    if len(num) / 1000 > iMax:
        sep = len(num)
    else:
        sep = 1000

    return sep

def raster2array(rasterfn):
    raster = gdal.Open(rasterfn)
    band = raster.GetRasterBand(1)
    noDataValue = band.GetNoDataValue()
    # rasterArray[rasterArray == noDataValue] = newValue
    array = band.ReadAsArray()
    return array


def coord2pixelOffset(rasterfn, x, y):
    raster = gdal.Open(rasterfn)
    geotransform = raster.GetGeoTransform()
    originX = geotransform[0]
    originY = geotransform[3]
    pixelWidth = geotransform[1]
    pixelHeight = geotransform[5]
    xOffset = int((x - originX) / pixelWidth)
    yOffset = int((y - originY) / pixelHeight)
    return xOffset, yOffset

def array2raster(newRasterfn, rasterfn, array, datatype):
    raster = gdal.Open(rasterfn)
    geotransform = raster.GetGeoTransform()
    originX = geotransform[0]
    originY = geotransform[3]
    pixelWidth = geotransform[1]
    pixelHeight = geotransform[5]
    cols = array.shape[1]
    rows = array.shape[0]

    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create(newRasterfn, cols, rows, 1, datatype)
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
    outband = outRaster.GetRasterBand(1)
    outband.SetNoDataValue(0)
    outband.WriteArray(array)
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromWkt(raster.GetProjectionRef())
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()
    outRaster = None


def createPath(CostSurfacefn, costSurfaceArray, startCoord, endCoord):
    # coordinates to array index
    startCoordX = startCoord[0]
    startCoordY = startCoord[1]
    # startIndexX, startIndexY = coord2pixelOffset(CostSurfacefn, startCoordX, startCoordY)

    endCoordX = endCoord[0]
    endCoordY = endCoord[1]
    # stopIndexX, stopIndexY = coord2pixelOffset(CostSurfacefn, stopCoordX, stopCoordY)

    # create path
    try:
        indices, weight = route_through_array(costSurfaceArray, (startCoordX, startCoordY), (endCoordX, endCoordY),
                                          geometric=True, fully_connected=True)
        indices = np.array(indices).T
        path = np.zeros_like(costSurfaceArray)
        path[indices[0], indices[1]] = 1
    except Exception as e:
        return [], -1
    else:
        return path, weight



if __name__ == "__main__":
    gdal.AllRegister()
    gdal.UseExceptions()
    main()
