import click
from osgeo import gdal, ogr, osr
import numpy as np
from shapely.geometry import shape, LineString, mapping
from skimage.graph import route_through_array
import cv2 as cv
import fiona
import time
import jsonlines
import rasterio
from rasterio.mask import mask
from skimage.graph import MCP, MCP_Geometric
from coordHandle import *

# input_path = "data\\2018输入数据2.tif"
CostSurfacefn_path = 'res/cost_background.tif'
# Costfn_path = 'data\\cost_test2.tif'  # 'D:\MSPA\\background4.tif' background4.tif D:\MSPA\\cost_test1.tif
outputPathfn_path = 'res/res_Path.tif'
outputPathfn_path1 = 'res/res_Path1.tif'
outputData_Path = "res/result.jsonl"
outputPath_shape = "res/res_sp.shp"


@click.command()
@click.option('--input-path', '-i',
              help='Input File, need the full path. For example, data/2018输入数据2.tif',
              required=True)
@click.option(
    '--cost-path', '-c',
    help='Background File contains cost matrix, need the full path.',
    required=True)
def main(input_path, cost_path):
    start = time.time()
    input_dataset = gdal.Open(input_path)
    Cost_dataset = gdal.Open(cost_path, gdal.GDT_Float32) # .GDT_Float32

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
    # rasterArray[rasterArray > 0] = 255

    res = np.zeros((ysize, xsize), dtype=np.uint8) # 这里用int有bug
    # res = np.ones(rasterArray.shape, dtype=np.uint8) * 255
    contours, heriachy = cv.findContours(rasterArray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)  # RETR_EXTERNAL RETR_TREE RETR_FLOODFILL  RETR_EXTERNAL

    cv.drawContours(res, contours, -1, (255, 255, 255), thickness=cv.FILLED)
    # cv.imwrite('contours.jpg', res)
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
    srcband = Cost_dataset.GetRasterBand(1)
    rasterArray = srcband.ReadAsArray().astype(np.float)
    rasterArray[rasterArray == 0] = 0.5  # 空数据设置为0.5，不允许出现0

    with fiona.open("res/No_Hole_Poly.shp", "r") as shapefile:
        src = rasterio.open(cost_path)
        no_data = src.nodata
        shapes = [feature["geometry"] for feature in shapefile]

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
                # if j != 686:
                #     continue
                end_point = shape(shapes[j]).representative_point()
                endCoord = coord2pixelOffset(Cost_dataset, end_point.x, end_point.y)

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

                ## for debug
                # if i == 0 and j == 1:
                #     array2raster(outputPathfn_path, CostSurfacefn_path, res_array, gdal.GDT_Int32, 0)
                #     res_array[res_array > 0] = 1
                #     array2raster(outputPathfn_path1, CostSurfacefn_path, res_array, gdal.GDT_Byte, 0)
                # jsonlines2shapefile(input_dataset, "res/result.jsonl", outputData_Path)

        array2raster(outputPathfn_path, CostSurfacefn_path, res_array, gdal.GDT_Int32, 0)
        res_array[res_array > 0] = 1
        array2raster(outputPathfn_path1, CostSurfacefn_path, res_array, gdal.GDT_Byte, 0)
        # jsonlines2shapefile(input_dataset, outputData_Path, )

    end = time.time()
    print("Finished! Execution Time: ", end - start)


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

    # input_dataset = gdal.Open("data/2018输入数据2.tif")
    # jsonlines2shapefile(input_dataset, "res/result.jsonl")
    main()
