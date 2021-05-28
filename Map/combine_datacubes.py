import shutil
import os
from osgeo import gdal
import numpy as np
import pandas as pd
from tqdm import tqdm


def extract_tiles(folder_path):
    
    '''
    extract the filepaths and tile names for all
    of the building height tiles
    '''
    filepaths = []
    filenames = []
    for root, folders, files in os.walk(folder_path):
        
        for folder in folders:
            
            path_to_folder = os.path.join(folder_path, folder)
            
            for file_ in os.listdir(path_to_folder):

                if file_.endswith(".tif"):

                    file_path = os.path.join(path_to_folder, file_)
                    
                    filepaths.append(file_path)
                    filenames.append(file_)
                    
                    
    return pd.DataFrame(zip(filepaths,filenames), columns=['Filepath','Filename'])


def get_mean_of_arrays(list_arrays):
    '''
    gets the mean of array with numpy
    used to get the average value over a tile that appears
    mulitple times to get average even with nan
    '''
    array = np.array(list_arrays)
    array[array<0] = np.nan
    return np.nanmean(array, axis=0)


def get_mean_of_all_files(subset):
    
    filepaths = subset['Filepath'].values
    
    files = []
    for file_ in filepaths:
        
        source = gdal.Open(file_)
        raster_band = source.GetRasterBand(1)
        scene = raster_band.ReadAsArray()
        
        scene = scene.astype('float')
        files.append(scene)
        
    return get_mean_of_arrays(files), source

    

def save_gdal(filenpath, source, array, rows, cols):
    """
    save the predcitions in the same projection 
    """

    driver = gdal.GetDriverByName("GTiff")
    driver.Register()
    outdata = driver.Create(filenpath, xsize=rows, ysize=cols, bands=1, eType=gdal.GDT_Int16)

    outdata.SetGeoTransform(source.GetGeoTransform())##sets same geotransform as input
    outdata.SetProjection(source.GetProjection())##sets same projection as input

    outdata.GetRasterBand(1).WriteArray(array)
    outdata.GetRasterBand(1).SetDescription('Building Heights')
    outdata.GetRasterBand(1).SetNoDataValue(-9999)##if you want these values transparent
    
    outdata.FlushCache() ##saves to disk!!
    
    outdata = None
    array   = None
    source  = None


def combine_multiple_tiles_save_to_dir(files, output_folder):

    all_filenames = files['Filename'].unique()

    for file_ in tqdm(all_filenames):
        
        subset = files[files['Filename'] == file_]
        
        if len(subset) > 1:
            
            mean_of_tile, source = get_mean_of_all_files(subset)
            rows, cols = mean_of_tile.shape
            output_path = output_folder + file_
            save_gdal(output_path, source, mean_of_tile, rows, cols)
            
        else:
            
            filepath = subset['Filepath'].values[0]
            shutil.copy(filepath, output_folder)



if __name__ == "__main__":

    output_folder = '/home/ubuntu/newvolume/combined_tiles/'
    input_folder  = '/home/ubuntu/newvolume/tiles/'

    files = extract_tiles(input_folder)
    files = files[files['Filename']!='merged_tile.tif']
    combine_multiple_tiles_save_to_dir(files, output_folder)