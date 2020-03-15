import click
from osgeo import gdal, ogr, osr
import numpy as np
from shapely.geometry import shape, LineString, mapping
import fiona
import time
import jsonlines
from coordHandle import *
from skimage.graph import MCP
from fileHandle import jsonlines2shapefile

outputData_Path = "res/result.jsonl"
outputPathfn_path = 'res/res_Path.tif'
outputPathfn_path1 = 'res/res_Path1.tif'
outputData_Path = "res/result.jsonl"
outputPath_shape = "res/res_sp.shp"

@click.command()
@click.option('--input-shp', '-i',
              help='Input point shapefile, need the full path. For example, data/No_Hole_Poly2_点.shp',
              required=True)
@click.option(
    '--cost-path', '-c',
    help='Background File contains cost matrix, need the full path. For example, data/cost_test2.tif',
    required=True)
def main(input_shp, cost_path):
    Cost_dataset = gdal.Open(cost_path, gdal.GDT_Float32)
    costSurfaceArray = raster2array(Cost_dataset)

    res_array = np.zeros_like(costSurfaceArray)

    with fiona.open(input_shp, "r") as shapefile:
        shapes = [feature["geometry"] for feature in shapefile]

    with jsonlines.open(outputData_Path, mode='w', flush=True) as out_json:
        for i in range(len(shapes)):
            # if i != 198:
            #     continue
            start_point = shape(shapes[i])  # 根据多边形中心点提取对应矩阵元素位置
            startCoord = coord2pixelOffset(Cost_dataset, start_point.x, start_point.y)

            mcp_class = MCP
            m = mcp_class(costSurfaceArray, fully_connected=True)
            costs, traceback_array = m.find_costs([startCoord])  # 直接计算从源点出发到其他所有点的最短路径

            for j in range(i, len(shapes)):
                if i == j:
                    continue
                # if j != 686:
                #     continue
                end_point = shape(shapes[j])
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

        array2raster(outputPathfn_path, cost_path, res_array, gdal.GDT_Int32, 0)
        res_array[res_array > 0] = 1
        array2raster(outputPathfn_path1, cost_path, res_array, gdal.GDT_Byte, 0)
        # jsonlines2shapefile(input_dataset, outputData_Path, )

if __name__ == "__main__":
    gdal.AllRegister()
    gdal.UseExceptions()

    main()