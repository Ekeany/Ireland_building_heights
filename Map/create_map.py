import glob
import os
import subprocess
from osgeo import gdal


def create_dem_list(directory):
    
    dem_list = []
    for file in os.listdir(directory):
        dem_list.append(directory+file)

    return dem_list


if __name__ == "__main__":

    input_dir = '/home/ubuntu/newvolume/cleaned_combined_tiles/'
    ouput_dir = '/home/ubuntu/newvolume/the_map/'

    dem_list =  create_dem_list(input_dir)
    vrt = gdal.BuildVRT(ouput_dir+'building_heights.vrt', dem_list)
    gdal.Translate(ouput_dir+'building_heights.tif', vrt, xRes=10, yRes=-10)
    vrt = None