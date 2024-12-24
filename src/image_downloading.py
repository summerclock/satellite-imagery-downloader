import cv2
import requests
import numpy as np
import threading
from osgeo import gdal
from pyproj import Proj
from tqdm import tqdm


def download_tile(url, headers, channels):
    try:
        response = requests.get(url, headers=headers)
        arr =  np.asarray(bytearray(response.content), dtype=np.uint8)
    except requests.exceptions.RequestException as e:
        print(f"Exception while downloading tile: {e}")
        return download_tile(url, headers, channels)
    except Exception as e:
        print(f"Unhandled exception while downloading tile: {e}")
        return None
    
    if channels == 3:
        return cv2.imdecode(arr, 1)
    return cv2.imdecode(arr, -1)


# Mercator projection 
# https://developers.google.com/maps/documentation/javascript/examples/map-coordinates
def project_with_scale(lat, lon, scale):
    siny = np.sin(lat * np.pi / 180)
    siny = min(max(siny, -0.9999), 0.9999)
    x = scale * (0.5 + lon / 360)
    y = scale * (0.5 - np.log((1 + siny) / (1 - siny)) / (4 * np.pi))
    return x, y


def download_image(lat1: float, lon1: float, lat2: float, lon2: float,
    zoom: int, url: str, headers: dict, out_path, tile_size: int = 256, channels: int = 3, max_threads=8):
    """
    Downloads a map region. Returns an image stored as a `numpy.ndarray` in BGR or BGRA, depending on the number
    of `channels`.

    Parameters
    ----------
    `(lat1, lon1)` - Coordinates (decimal degrees) of the top-left corner of a rectangular area

    `(lat2, lon2)` - Coordinates (decimal degrees) of the bottom-right corner of a rectangular area

    `zoom` - Zoom level

    `url` - Tile URL with {x}, {y} and {z} in place of its coordinate and zoom values

    `headers` - Dictionary of HTTP headers

    `tile_size` - Tile size in pixels

    `channels` - Number of channels in the output image. Also affects how the tiles are converted into numpy arrays.
    """

    scale = 1 << zoom

    # Find the pixel coordinates and tile coordinates of the corners
    tl_proj_x, tl_proj_y = project_with_scale(lat1, lon1, scale)
    br_proj_x, br_proj_y = project_with_scale(lat2, lon2, scale)

    tl_pixel_x = int(tl_proj_x * tile_size)
    tl_pixel_y = int(tl_proj_y * tile_size)
    br_pixel_x = int(br_proj_x * tile_size)
    br_pixel_y = int(br_proj_y * tile_size)

    tl_tile_x = int(tl_proj_x)
    tl_tile_y = int(tl_proj_y)
    br_tile_x = int(br_proj_x)
    br_tile_y = int(br_proj_y)

    img_w = abs(tl_pixel_x - br_pixel_x)
    img_h = br_pixel_y - tl_pixel_y
    creation_options = ['TILED=YES', 'BLOCKXSIZE=256', 'BLOCKYSIZE=256']
    raster_ds = gdal.GetDriverByName('GTiff').Create(out_path, img_w, img_h, channels, gdal.GDT_Byte, creation_options)
    write_lock = threading.Lock()


    def build_row(tile_y):
        with tqdm(total=(br_tile_x - tl_tile_x + 1), leave=True, desc=f'Row {tile_y}') as row_pbar:
            for tile_x in range(tl_tile_x, br_tile_x + 1):
                tile = download_tile(url.format(x=tile_x, y=tile_y, z=zoom), headers, channels)
                # bgr to rgb
                if channels == 3:
                    tile = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
                if tile is not None:
                    # Find the pixel coordinates of the new tile relative to the image
                    tl_rel_x = tile_x * tile_size - tl_pixel_x
                    tl_rel_y = tile_y * tile_size - tl_pixel_y
                    br_rel_x = tl_rel_x + tile_size
                    br_rel_y = tl_rel_y + tile_size

                    # Define where the tile will be placed on the image
                    img_x_l = max(0, tl_rel_x)
                    img_x_r = min(img_w + 1, br_rel_x)
                    img_y_l = max(0, tl_rel_y)
                    img_y_r = min(img_h + 1, br_rel_y)

                    # Define how border tiles will be cropped
                    cr_x_l = max(0, -tl_rel_x)
                    cr_x_r = tile_size + min(0, img_w - br_rel_x)
                    cr_y_l = max(0, -tl_rel_y)
                    cr_y_r = tile_size + min(0, img_h - br_rel_y)
                    
                    write_lock.acquire()
                    raster_ds.GetRasterBand(1).WriteArray(tile[cr_y_l:cr_y_r, cr_x_l:cr_x_r, 0], img_x_l, img_y_l)
                    if channels >= 3:
                        raster_ds.GetRasterBand(2).WriteArray(tile[cr_y_l:cr_y_r, cr_x_l:cr_x_r, 1], img_x_l, img_y_l)
                        raster_ds.GetRasterBand(3).WriteArray(tile[cr_y_l:cr_y_r, cr_x_l:cr_x_r, 2], img_x_l, img_y_l)
                    if channels == 4:
                        raster_ds.GetRasterBand(4).WriteArray(tile[cr_y_l:cr_y_r, cr_x_l:cr_x_r, 3], img_x_l, img_y_l)
                    write_lock.release()
                row_pbar.update(1)

    threads = []
    with tqdm(total=(br_tile_y - tl_tile_y + 1), leave=True, desc="Overall") as pbar:
        for tile_y in range(tl_tile_y, br_tile_y + 1):
            while len(threads) >= max_threads:
                for thread in threads:
                    if not thread.is_alive():
                        threads.remove(thread)
                        pbar.update(1)
            
            thread = threading.Thread(target=build_row, args=[tile_y])
            thread.start()
            threads.append(thread)
        
        for thread in threads:
            thread.join()
            pbar.update(1)

    transformer = Proj.from_proj(Proj(init='epsg:4326'), Proj(init='epsg:3857'))
    tl_geo_x, tl_geo_y = transformer.transform(lon1, lat1)
    br_geo_x, br_geo_y = transformer.transform(lon2, lat2)
    geotransform = (tl_geo_x, (br_geo_x - tl_geo_x) / img_w, 0, tl_geo_y, 0, (br_geo_y - tl_geo_y) / img_h)
    raster_ds.SetGeoTransform(geotransform)
    raster_ds.SetProjection('EPSG:3857')
    raster_ds.FlushCache()
    raster_ds = None


def image_size(lat1: float, lon1: float, lat2: float,
    lon2: float, zoom: int, tile_size: int = 256):
    """ Calculates the size of an image without downloading it. Returns the width and height in pixels as a tuple. """

    scale = 1 << zoom
    tl_proj_x, tl_proj_y = project_with_scale(lat1, lon1, scale)
    br_proj_x, br_proj_y = project_with_scale(lat2, lon2, scale)

    tl_pixel_x = int(tl_proj_x * tile_size)
    tl_pixel_y = int(tl_proj_y * tile_size)
    br_pixel_x = int(br_proj_x * tile_size)
    br_pixel_y = int(br_proj_y * tile_size)

    return abs(tl_pixel_x - br_pixel_x), br_pixel_y - tl_pixel_y
