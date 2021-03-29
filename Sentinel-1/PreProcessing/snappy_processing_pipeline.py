from __future__ import division

from osgeo import gdal
from zipfile import ZipFile
import subprocess
import os
import numpy as np
from tqdm import tqdm
import re
import sys



def set_no_data_values_for_gdal(filename):
    
    source = gdal.Open(filename)
    
    band1 = source.GetRasterBand(1)
    arr1 = band1.ReadAsArray()
    
    band2 = source.GetRasterBand(2)
    arr2 = band2.ReadAsArray()
    
    [cols, rows] = arr1.shape

    driver = gdal.GetDriverByName("GTiff")
    driver.Register()
    outdata = driver.Create(filename, xsize=rows, ysize=cols, bands=2, eType=gdal.GDT_Int16)
    
    outdata.SetGeoTransform(source.GetGeoTransform())##sets same geotransform as input
    outdata.SetProjection(source.GetProjection())##sets same projection as input
    
    outdata.GetRasterBand(1).WriteArray(arr1)
    outdata.GetRasterBand(1).SetDescription('VV')
    outdata.GetRasterBand(1).SetNoDataValue(-9999)##if you want these values transparent
    
    outdata.GetRasterBand(2).WriteArray(arr2)
    outdata.GetRasterBand(2).SetDescription('VH')
    outdata.GetRasterBand(2).SetNoDataValue(-9999)##if you want these values transparent
    
    outdata.FlushCache() ##saves to disk!!
    
    outdata = None
    band1   = None
    band2   = None
    source  = None
    del arr2
    del arr1


def filter_output(output_string, output_path):

    cleaned_output = [line for line in output_string.split('\n') if line.strip() != '']
    for elmt in cleaned_output:
        if output_path in elmt:
            return elmt



def check_if_files_have_been_processed(input_path, output_path):

    input_files  = os.listdir(input_path)
    output_files = os.listdir(output_path)

    num_output_files = len(output_files)
    if num_output_files > 0:

        print('Skipping the first ' + str(num_output_files) + ' files as they have already been processed')
        return input_files[num_output_files:]

    else:
        return input_files




if __name__ == "__main__":

    # python snappy_processing_pipeline.py /home/ubuntu/newvolume/level1/ /home/ubuntu/newvolume/level2/ /home/ubuntu/params/srtm_35_02.tif

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    externel_dem = sys.argv[3]

    print('The input path is: '  + str(input_path))
    print('The output path is: ' + str(output_path))

    files = check_if_files_have_been_processed(input_path, output_path)

    for filename in tqdm(files):
        
        if filename.endswith(".zip"):
            
            file_path = str(input_path) + str(filename)

            try:
                output_name = subprocess.check_output(['python','snappy_pipeline.py', file_path, output_path, externel_dem])
                output_name = filter_output(output_name, output_path)
                set_no_data_values_for_gdal(output_name)
            
            except:
                print('process failed for: ' + filename)
