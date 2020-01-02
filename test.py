import gdal, ogr, osr, os
from shapely.geometry import shape
import numpy as np
from shapely.wkt import dumps, loads
from skimage import measure, io
from skimage.graph import route_through_array
import matplotlib.pyplot as plt
from PIL import Image
import cv2 as cv


def main():

    dataset = gdal.Open("D:/MSPA/i.tif")
    # print(dataset.GetDescription())
    # print(dataset.RasterCount)  #波段数

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
    dst_layer = dst_ds.CreateLayer(dst_layername, srs = None)

    gdal.Polygonize(out_band, res_dataset.GetRasterBand(2), dst_layer, -1, [], callback=None)

    driver = ogr.GetDriverByName('ESRI Shapefile')
    dataSource = driver.Open('POLYGONIZED_STUFF.shp', 0)
    layer = dataSource.GetLayer(0)
    print(layer.GetFeatureCount())

    # 提取轮廓
    dataset = gdal.Open("D:/MSPA/i.tif")
    srcband = dataset.GetRasterBand(1)
    rasterArray = srcband.ReadAsArray()
    # cv.imshow("binary", rasterArray)

    contours = measure.find_contours(rasterArray, 0)
    res = np.zeros((ysize, xsize), dtype=np.int)

    for n, contour in enumerate(contours):
        # plt.plot(contour[:, 1], ysize - contour[:, 0], linewidth=1, color='black')
        for i in range(contour.shape[0]):
            if i == 6:
                startCoord = (int(contour[i][0]), int(contour[i][1]))
            if i == 108:
                stopCoord = (int(contour[i][0]), int(contour[i][1]))

            res[int(contour[i][0])][int(contour[i][1])] = 1

    array2raster('result.tif', 'D:/MSPA/i.tif', res, gdal.GDT_Byte)

    # image = cv.imread('rk.jpg')
    # gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # ret, binary = cv.threshold(gray,127,255, cv.THRESH_BINARY)
    # cv.imshow("binary image", binary)
    # cv.imwrite('rk_binary.jpg', binary)

    res = np.zeros((ysize, xsize), dtype=np.int)
    contours, heriachy = cv.findContours(rasterArray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE) # RETR_EXTERNAL RETR_TREE
    for n, contour in enumerate(contours):
        # plt.plot(contour[:, 1], ysize - contour[:, 0], linewidth=1, color='black')
        for i in range(contour.shape[0]):
            res[int(contour[i][0][1])][int(contour[i][0][0])] = 1

    array2raster('result.tif', 'D:/MSPA/i.tif', res, gdal.GDT_Byte)

    # img = cv.imread("D:/MSPA/i2.tif", cv.IMREAD_LOAD_GDAL)
    # # cv.imshow("binary", rasterArray)
    # rasterArray[rasterArray == 1] = 255
    # # img[:,:,2] = 255
    # cv.imwrite('binary.jpg', img)
    #
    # for i, contour in enumerate(contours):
    #     cv.drawContours(img, contours, i, (0, 0, 255), 1)
    # print(i)
    #
    # color = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    # cv.imshow("contours", img)
    # cv.imwrite('contours.jpg', img)
    # for i, contour in enumerate(contours):
    #     cv.drawContours(img, contours, i, (0, 0, 255), -1)
    # # cv.imshow("pcontours", img)
    # cv.imwrite('pcontours.jpg', img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()


    # 两个像素点之间的最短路径
    CostSurfacefn = 'background.tif' #background4.tif D:\MSPA\\cost_test1.tif
    outputPathfn = 'Path.tif'

    costSurfaceArray = raster2array(CostSurfacefn)  # creates array from cost surface raster
    pathArray = createPath(CostSurfacefn, costSurfaceArray, startCoord, stopCoord)  # creates path array
    array2raster(outputPathfn, CostSurfacefn, pathArray, gdal.GDT_Byte)  # converts path array to raster


    # 边缘简化
    driver = ogr.GetDriverByName('ESRI Shapefile')
    dataSource = driver.Open('POLYGONIZED_STUFF.shp', 0)
    layer = dataSource.GetLayer(0)
    print(layer.GetFeatureCount())

    driver = ogr.GetDriverByName('Esri Shapefile')
    ds = driver.CreateDataSource('result.shp')
    out_layer = ds.CreateLayer('', None, ogr.wkbPolygon)
    # Add one attribute
    out_layer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))
    defn = out_layer.GetLayerDefn()
    print(defn)

    for feature in layer:
        geom = feature.GetGeometryRef()
        g = loads(geom.ExportToWkt())
        r = g.simplify(0.007, True)
        out = ogr.CreateGeometryFromWkt(dumps(r))
        feat = ogr.Feature(defn)
        feat.SetGeometry(out)
        out_layer.CreateFeature(feat)
        feat = geom = None
        # print(out)

    layer.ResetReading()

    print("Over")


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


def createPath(CostSurfacefn, costSurfaceArray, startCoord, stopCoord):
    # coordinates to array index
    startCoordX = startCoord[0]
    startCoordY = startCoord[1]
    # startIndexX, startIndexY = coord2pixelOffset(CostSurfacefn, startCoordX, startCoordY)

    stopCoordX = stopCoord[0]
    stopCoordY = stopCoord[1]
    # stopIndexX, stopIndexY = coord2pixelOffset(CostSurfacefn, stopCoordX, stopCoordY)

    # create path
    indices, weight = route_through_array(costSurfaceArray, (startCoordX, startCoordY), (stopCoordX, stopCoordY),
                                          geometric=False, fully_connected=True)
    indices = np.array(indices).T
    path = np.zeros_like(costSurfaceArray)
    path[indices[0], indices[1]] = 1
    return path


if __name__ == "__main__":
    gdal.UseExceptions()
    main()
