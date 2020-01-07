import sys
from osgeo import gdal, ogr, osr
import numpy as np
from shapely.geometry import shape, LineString, mapping
from shapely.wkt import dumps, loads
from skimage import measure, io
from skimage.graph import route_through_array
import cv2 as cv
import fiona
from collections import OrderedDict
import time
import csv
import jsonlines
import rasterio
from rasterio.mask import mask
from skimage.graph import MCP, MCP_Geometric

input_path = "D:\\MSPA\\2018输入数据.tif"
input_dataset = gdal.Open(input_path)
CostSurfacefn_path = 'D:\\MSPA\\cost_background.tif'
oCostfn_path = 'D:\\MSPA\\cost_test1.tif'  # 'D:\MSPA\\background4.tif' background4.tif D:\MSPA\\cost_test1.tif
oCost_dataset = gdal.Open(oCostfn_path, gdal.GDT_Float32)
outputPathfn_path = 'res\\res_Path.tif'
outputPathfn_path1 = 'res\\res_Path1.tif'
outputData_Path = "res\\result.jsonl"
outputPath_shape = "res\\res_sp.shp"


def main():
    srcband = input_dataset.GetRasterBand(1)
    # projectionfrom = input_dataset.GetProjection()
    # geotransform = input_dataset.GetGeoTransform()
    xsize = srcband.XSize
    ysize = srcband.YSize

    # 提取轮廓，把空洞填充
    srcband = input_dataset.GetRasterBand(1)
    rasterArray = srcband.ReadAsArray()
    noDataValue = srcband.GetNoDataValue()
    rasterArray[rasterArray == noDataValue] = 0

    res = np.zeros((ysize, xsize), dtype=np.int)
    contours, heriachy = cv.findContours(rasterArray, cv.RETR_EXTERNAL,
                                         cv.CHAIN_APPROX_NONE)  # RETR_EXTERNAL RETR_TREE RETR_FLOODFILL  RETR_EXTERNAL

    cv.drawContours(res, contours, -1, (255, 255, 255), thickness=cv.FILLED)
    res[res == 255] = 1
    array2raster('res/No_Hole_Raster.tif', input_path, res, gdal.GDT_Byte, 0)

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
    rasterArray[rasterArray == 0] = 0.5  # 空数据设置为0.5，不允许出现0

    with fiona.open("res/No_Hole_Poly.shp", "r") as shapefile:
        src = rasterio.open(oCostfn_path)
        no_data = src.nodata
        shapes = [feature["geometry"] for feature in shapefile]

        # s = filter(lambda x: shape(x).area <= 25000, shapes)  # 用面积过滤矢量图斑
        # shapes_f = list(s)

        # img, transform = mask(src, shapes_f, crop=False)
        # row, col = np.where(img[0] != no_data)
        # np.put(rasterArray, [row * xsize + col], 0.1)  # 面积小于阈值的斑块赋值为0.1

        img, transform = mask(src, shapes, crop=False)
        row, col = np.where(img[0] != no_data)
        np.put(rasterArray, [row * xsize + col], 0.1)  # 绿地斑块赋值为0.1

        array2raster(CostSurfacefn_path, input_path, rasterArray, gdal.GDT_Float32, no_data)
        Cost_dataset = gdal.Open(CostSurfacefn_path)
        costSurfaceArray = raster2array(Cost_dataset)  # creates array from cost surface raster

    # 用面积过滤矢量图斑
    s = filter(lambda x: shape(x).area > 25000, shapes)
    shapes = list(s)

    # with fiona.open("res/No_Hole_Poly.shp", "r") as shapefile:
    #     shapes = [feature["geometry"] for feature in shapefile]
    res_array = np.zeros_like(costSurfaceArray)

    with jsonlines.open(outputData_Path, mode='w', flush=True) as out_json:
        for i in range(len(shapes)):
            # if i != 198:
            #     continue
            start_point = shape(shapes[i]).representative_point()  # 根据多边形中心点提取对应矩阵元素位置
            startCoord = coord2pixelOffset(input_dataset, start_point.x, start_point.y)

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

                weight = costs[endCoord]
                outputPath = m.traceback(endCoord)
                indices = np.array(outputPath).T
                minPathArray = np.zeros_like(costSurfaceArray)
                minPathArray[indices[0], indices[1]] = 1

                if len(minPathArray) > 0:
                    res_array = minPathArray + res_array

                outputData = {
                    "source": i,
                    "end": j,
                    "weight": weight,
                    "shortestPath": outputPath
                }

                out_json.write(outputData)
                print(str(i) + "," + str(j))

                # outputPathfn_path1 = str(i) + ".tif"
                # res_array[res_array > 0] = 1
                # array2raster(outputPathfn_path1, CostSurfacefn_path, res_array, gdal.GDT_Byte, 0)

                # if i == 0 and j == 10:
                # jsonlines2shapefile(outputData_Path)
                # array2raster(outputPathfn_path, CostSurfacefn_path, res_array, gdal.GDT_Int32, 0)
                # res_array[res_array > 0] = 1
                # array2raster(outputPathfn_path1, CostSurfacefn_path, res_array, gdal.GDT_Byte, 0)

        array2raster(outputPathfn_path, CostSurfacefn_path, res_array, gdal.GDT_Int32, 0)
        res_array[res_array > 0] = 1
        array2raster(outputPathfn_path1, CostSurfacefn_path, res_array, gdal.GDT_Byte, 0)
        jsonlines2shapefile(outputData_Path)


class StreamArray(list):
    """
    Converts a generator into a list object that can be json serialisable
    while still retaining the iterative nature of a generator.

    IE. It converts it to a list without having to exhaust the generator
    and keep its contents in memory.
    """

    def __init__(self, generator):
        self.generator = generator
        self._len = 1

    def __iter__(self):
        self._len = 0
        for item in self.generator:
            yield item
            self._len += 1

    def __len__(self):
        """
        Json parser looks for a this method to confirm whether or not it can
        be parsed
        """
        return self._len


def jsonlines2shapefile(json_path):
    schema = {
        'geometry': 'LineString',
        'properties': OrderedDict([
            ('id', 'int'),
            ('source', 'int'),
            ('end', 'int'),
            ('weight', 'float')])
    }

    output_sp = fiona.open(outputPath_shape, 'w', 'ESRI Shapefile', schema)

    with jsonlines.open(json_path, mode='r') as in_json:
        icount = 0
        for obj in in_json:
            output_sp.write({
                'geometry': geometry_linestring(obj['shortestPath']),
                'properties': OrderedDict([
                    ('id', icount),
                    ('source', obj['source']),
                    ('end', obj['end']),
                    ('weight', obj['weight'])])
            })

    output_sp.close()


def geometry_linestring(Paths):
    t = map(lambda path: pixelOffset2coord(input_dataset, path[1], path[0]), Paths)
    # m = map(lambda x, y: x + y, [1, 3, 5, 7, 9], [2, 4, 6, 8, 10])
    t = LineString(list(t))
    return mapping(t)


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


def pixelOffset2coord(raster, xOffset, yOffset):
    # raster = gdal.Open(rasterfn)
    geotransform = raster.GetGeoTransform()
    originX = geotransform[0]
    originY = geotransform[3]
    pixelWidth = geotransform[1]
    pixelHeight = geotransform[5]
    coordX = originX + pixelWidth * xOffset
    coordY = originY + pixelHeight * yOffset
    return coordX, coordY


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
