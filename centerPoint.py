import sys
from osgeo import gdal, gdalnumeric, ogr, osr
import numpy as np
from shapely.geometry import shape
from shapely.wkt import dumps, loads
from skimage import measure, io
from skimage.graph import route_through_array
import cv2 as cv
import fiona
import time
import csv
import rasterio
from rasterio.mask import mask

def main():
    input_path = "D:/MSPA/i2.tif"
    input_dataset = gdal.Open(input_path)
    CostSurfacefn_path = 'D:\MSPA\\cost_background.tif'
    oCostfn_path = 'D:\MSPA\\cost_test.tif'  # 'D:\MSPA\\background4.tif' background4.tif D:\MSPA\\cost_test1.tif
    oCost_dataset = gdal.Open(oCostfn_path)
    outputPathfn_path = 'res_Path.tif'

    srcband = input_dataset.GetRasterBand(1)
    projectionfrom = input_dataset.GetProjection()
    geotransform = input_dataset.GetGeoTransform()
    xsize = srcband.XSize
    ysize = srcband.YSize

    rasterArray = srcband.ReadAsArray()

    # 提取轮廓
    dataset = gdal.Open("D:/MSPA/i2.tif")
    srcband = dataset.GetRasterBand(1)
    rasterArray = srcband.ReadAsArray()

    res = np.zeros((ysize, xsize), dtype=np.int)
    contours, heriachy = cv.findContours(rasterArray, cv.RETR_EXTERNAL,
                                         cv.CHAIN_APPROX_NONE)  # RETR_EXTERNAL RETR_TREE RETR_FLOODFILL  RETR_EXTERNAL

    cv.drawContours(res, contours, -1, (255, 255, 255), thickness=cv.FILLED)
    res[res == 255] = 1
    array2raster('No_Hole_Raster.tif', 'D:/MSPA/i2.tif', res, gdal.GDT_Byte)

    # 提取矢量多边形
    res_dataset = gdal.Open("No_Hole_Raster.tif")
    out_band = res_dataset.GetRasterBand(1)
    dst_layername = "No_Hole_Poly"
    drv = ogr.GetDriverByName("ESRI Shapefile")
    dst_ds = drv.CreateDataSource(dst_layername + ".shp")
    dst_layer = dst_ds.CreateLayer(dst_layername, srs=None)

    gdal.Polygonize(out_band, res_dataset.GetRasterBand(1), dst_layer, -1, [], callback=None)
    dst_ds.Destroy()

    # # 修改背景图层栅格值
    # src = rasterio.open("oCostfn_path.tif")
    # with fiona.open("No_Hole_Raster.shp", "r") as shapefile:
    #     shapes = [feature["geometry"] for feature in shapefile]
    #     # srcband = oCost_dataset.GetRasterBand(1)
    #     # rasterArray = srcband.ReadAsArray()
    #     img, transform = mask(src, shapes, crop=False)
    #
    # gtiff = gdal.GetDriverByName('GTiff')
    # output_dataset = gtiff.Create(CostSurfacefn_path, xsize, ysize, 2, gdal.GDT_Byte)
    # output_dataset.SetProjection(projectionfrom)
    # output_dataset.SetGeoTransform(geotransform)

    # # rasterArray[rasterArray == 2] = 0
    # output_dataset.GetRasterBand(1).WriteArray(rasterArray)
    # output_dataset.GetRasterBand(2).SetNoDataValue(0)  # 只能在创建的时候使用
    # output_dataset.GetRasterBand(2).WriteArray(rasterArray)  # mask图层
    # output_dataset.FlushCache()

    # 两个像素点之间的最短路径

    costSurfaceArray = raster2array(oCost_dataset)  # creates array from cost surface raster

    # 根据多边形中心点提取对应矩阵元素位置
    # src = rasterio.open(input_path)
    visited = []
    with fiona.open("No_Hole_Poly.shp", "r") as shapefile:
        shapes = [feature["geometry"] for feature in shapefile]
        res_array = np.zeros_like(costSurfaceArray)

        with open("result.csv", 'w', newline="") as out_csv:
            csvwriter = csv.writer(out_csv)

            for i in range(len(shapes)):
                visited.append(i)

                for j in range(len(shapes)):
                    if i == j:
                        continue
                    if j in visited:
                        continue
                    # if j != 42:
                    #     continue

                    start_point = shape(shapes[i]).representative_point()
                    end_point = shape(shapes[j]).representative_point()
                    startCoord = coord2pixelOffset(oCost_dataset, start_point.x, start_point.y)
                    endCoord = coord2pixelOffset(oCost_dataset, end_point.x, end_point.y)

                    # start = time.time()
                    minPathArray, weight = createPath(CostSurfacefn_path, costSurfaceArray, startCoord, endCoord)  # creates path array
                    # end = time.time()
                    # print("Execution Time: ", end - start)

                    if len(minPathArray) > 0:
                        res_array = minPathArray + res_array

                    csvwriter.writerow([str(i), str(j), str(weight)])
                    out_csv.flush()
                    print(str(i) + "," + str(j))

                    if i == 1 and j == 500:
                        res_array[res_array > 0] = 1
                        array2raster(outputPathfn_path, CostSurfacefn_path, res_array, gdal.GDT_Byte)

            array2raster(outputPathfn_path, CostSurfacefn_path, res_array, gdal.GDT_Byte)  # converts path array to raster
            out_csv.close()

    print("Over")


def raster2array(raster):
    band = raster.GetRasterBand(1)
    noDataValue = band.GetNoDataValue()
    # rasterArray[rasterArray == noDataValue] = newValue
    array = band.ReadAsArray()
    return array


def coord2pixelOffset(raster, x, y):
    # raster = gdal.Open(rasterfn)
    geotransform = raster.GetGeoTransform()
    originX = geotransform[0]
    originY = geotransform[3]
    pixelWidth = geotransform[1]
    pixelHeight = geotransform[5]
    xOffset = int((x - originX) / pixelWidth)
    yOffset = int((y - originY) / pixelHeight)
    return yOffset, xOffset


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
                                              geometric=False, fully_connected=True)
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
