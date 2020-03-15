import fiona
from collections import OrderedDict
import jsonlines
import gdal
from coordHandle import pixelOffset2coord
from shapely.geometry import LineString, mapping
import click

@click.command()
@click.option('--input-path', '-i',
              help='Input File, need the full path. For example, data\\2018输入数据2.tif',
              required=True)
@click.option('--jsonl-path', '-j',
              help='Input jsonl File, need the full path. For example, "res/result.jsonl"',
              required=True)
@click.option('--output-path', '-o',
              help='Output shapefile, need the full path. For example "res/res_sp.shp"',
              required=True)
@click.option('--distance', '-d',
              help='Distance for simplify algorithm. For example 100',
              default=100)
def jsonlines2shapefile(input_path, jsonl_path, output_path, distance):
    input_dataset = gdal.Open(input_path)

    schema = {
        'geometry': 'LineString',
        'properties': OrderedDict([
            ('id', 'int'),
            ('source', 'int'),
            ('end', 'int'),
            ('weight', 'float')])
    }

    output_sp = fiona.open(output_path, 'w', 'ESRI Shapefile', schema)

    with jsonlines.open(jsonl_path, mode='r') as in_json:
        icount = 0
        for obj in in_json:
            output_sp.write({
                'geometry': geometry_linestring(input_dataset, obj['shortestPath'], distance),
                'properties': OrderedDict([
                    ('id', icount),
                    ('source', obj['source']),
                    ('end', obj['end']),
                    ('weight', obj['weight'])])
            })
            icount = icount + 1
            print(icount)

    output_sp.close()


def geometry_linestring(input_dataset, Paths, distance):
    t = map(lambda path: pixelOffset2coord(input_dataset, path[1], path[0]), Paths)
    # m = map(lambda x, y: x + y, [1, 3, 5, 7, 9], [2, 4, 6, 8, 10])
    t = LineString(list(t))
    t = t.simplify(distance)
    return mapping(t)


if __name__ == "__main__":
    gdal.AllRegister()
    gdal.UseExceptions()

    # input_dataset = gdal.Open("data/2018输入数据2.tif")
    jsonlines2shapefile()
