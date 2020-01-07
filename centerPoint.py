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
from skimage.graph import MCP, MCP_Geometric, MCP_Connect, MCP_Flexible


def main():
    input_path = "D:\\MSPA\\i2.tif"
    input_dataset = gdal.Open(input_path)
    CostSurfacefn_path = 'D:\\MSPA\\cost_background.tif'
    oCostfn_path = 'D:\\MSPA\\cost_test1.tif'  # 'D:\MSPA\\background4.tif' background4.tif D:\MSPA\\cost_test1.tif
    oCost_dataset = gdal.Open(oCostfn_path, gdal.GDT_Float32)
    outputPathfn_path = 'res\\res_Path.tif'

    srcband = input_dataset.GetRasterBand(1)
    projectionfrom = input_dataset.GetProjection()
    geotransform = input_dataset.GetGeoTransform()
    xsize = srcband.XSize
    ysize = srcband.YSize

    # 提取轮廓，把空洞填充
    dataset = gdal.Open("D:/MSPA/i2.tif")
    srcband = dataset.GetRasterBand(1)
    rasterArray = srcband.ReadAsArray()

    res = np.zeros((ysize, xsize), dtype=np.int)
    contours, heriachy = cv.findContours(rasterArray, cv.RETR_EXTERNAL,
                                         cv.CHAIN_APPROX_NONE)  # RETR_EXTERNAL RETR_TREE RETR_FLOODFILL  RETR_EXTERNAL

    cv.drawContours(res, contours, -1, (255, 255, 255), thickness=cv.FILLED)
    res[res == 255] = 1
    array2raster('res/No_Hole_Raster.tif', 'D:/MSPA/i2.tif', res, gdal.GDT_Byte, 0)

    # 提取矢量多边形
    res_dataset = gdal.Open("res/No_Hole_Raster.tif")
    out_band = res_dataset.GetRasterBand(1)
    dst_layername = "res/No_Hole_Poly"
    drv = ogr.GetDriverByName("ESRI Shapefile")
    dst_ds = drv.CreateDataSource(dst_layername + ".shp")
    dst_layer = dst_ds.CreateLayer(dst_layername, srs=None)

    gdal.Polygonize(out_band, res_dataset.GetRasterBand(1), dst_layer, -1, [], callback=None)
    dst_ds.Destroy()

    # 修改背景图层栅格值
    srcband = oCost_dataset.GetRasterBand(1)
    rasterArray = srcband.ReadAsArray().astype(np.float)
    rasterArray[rasterArray == 0] = 0.5

    with fiona.open("res/No_Hole_Poly.shp", "r") as shapefile:
        src = rasterio.open(oCostfn_path)
        no_data = src.nodata
        shapes = [feature["geometry"] for feature in shapefile]
        img, transform = mask(src, shapes, crop=False)
        row, col = np.where(img[0] != no_data)

        # res_array = np.zeros_like(rasterArray)
        np.put(rasterArray, [row * xsize + col], 0.1)

        # out_meta = src.meta
        # with rasterio.open("RGB.byte.masked.tif", "w", **out_meta) as dest:
        #     dest.write(img)

        array2raster(CostSurfacefn_path, input_path, rasterArray, gdal.GDT_Float32, no_data)
        Cost_dataset = gdal.Open(CostSurfacefn_path)
        costSurfaceArray = raster2array(Cost_dataset)  # creates array from cost surface raster

    # 两个像素点之间的最短路径

    # 根据多边形中心点提取对应矩阵元素位置
    # src = rasterio.open(input_path)
    visited = []
    with fiona.open("res/No_Hole_Poly.shp", "r") as shapefile:
        shapes = [feature["geometry"] for feature in shapefile]
        res_array = np.zeros_like(costSurfaceArray)

        with open("res/result.csv", 'w', newline="") as out_csv:
            csvwriter = csv.writer(out_csv)

            for i in range(len(shapes)):
                start_point = shape(shapes[i]).representative_point()
                startCoord = coord2pixelOffset(oCost_dataset, start_point.x, start_point.y)

                mcp_class = MCP
                m = mcp_class(costSurfaceArray, fully_connected=True)
                costs, traceback_array = m.find_costs([startCoord])  # 直接计算从源点出发到其他所有点的最短路径

                for j in range(i, len(shapes)):
                    if i == j:
                        continue
                    # if j != 42:
                    #     continue
                    end_point = shape(shapes[j]).representative_point()
                    endCoord = coord2pixelOffset(oCost_dataset, end_point.x, end_point.y)

                    # start = time.time()
                    weight = costs[endCoord]
                    indices = m.traceback(endCoord)
                    indices = np.array(indices).T
                    minPathArray = np.zeros_like(costSurfaceArray)
                    minPathArray[indices[0], indices[1]] = 1

                    # end = time.time()
                    # print("Execution Time: ", end - start)

                    if len(minPathArray) > 0:
                        res_array = minPathArray + res_array

                    csvwriter.writerow([str(i), str(j), str(weight)])
                    out_csv.flush()
                    print(str(i) + "," + str(j))

                    if i == 0 and j == 100:
                        res_array[res_array > 0] = 1
                        array2raster(outputPathfn_path, CostSurfacefn_path, res_array, gdal.GDT_Byte, 0)

            res_array[res_array == 0] = 1
            array2raster(outputPathfn_path, CostSurfacefn_path, res_array, gdal.GDT_Byte,
                         0)  # converts path array to raster
            out_csv.close()


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


def array2raster(newRasterfn, rasterfn, array, datatype, nodata):
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
    outband.SetNoDataValue(nodata)
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

        # start = time.time()
        indices = np.array(indices).T
        path = np.zeros_like(costSurfaceArray)
        path[indices[0], indices[1]] = 1
        # end = time.time()
        # print("Execution Time: ", end - start)
    except Exception as e:
        return [], -1
    else:
        return path, weight


if __name__ == "__main__":
    gdal.AllRegister()
    gdal.UseExceptions()
    main()
    print("Over")
