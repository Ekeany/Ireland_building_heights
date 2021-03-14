from osgeo import gdal
import numpy as np
import random
import pandas as pd
from tqdm import tqdm
import sys
import re

def compute_height_distribtuion(building_height_path):
    '''
    calculates the distribution of pixel values given a 
    raster image.
    
    plots a histogram and returns dictionary of pixel value 
    and percentage it occurs
    '''
    
    print('Getting the building height distribution for this raster....')
    total_building_height = gdal.Open(building_height_path)
    height_data = total_building_height.GetRasterBand(1).ReadAsArray()
    
    flat_height = height_data.flatten()
    # remove all 0s and no data values
    flat_height = flat_height[flat_height > 0]
    
    total_num_pixels = len(flat_height)
    height_dist = {}
    for height in tqdm(np.unique(flat_height)):
        
        num_pixels = sum(flat_height == height)
        height_dist[height] = round(num_pixels/total_num_pixels,3)

    
    del total_building_height
    
    return height_dist
    

    
def stratified_height_sample(building_height_path, height_stratified, percentage):
    '''
    given a raster it will return a stratified sample of pixel
    values given dictionary with the percentage of each pixel
    value to sample.
    
    {1:0.3,
     2:0.5
     3:0.2}
    '''
    
    building_height = gdal.Open(building_height_path)
    building_height = building_height.GetRasterBand(1).ReadAsArray()
    
    numrows = len(building_height)    # 3 rows in your example
    numcols = len(building_height[0]) # 2 columns in your example
    length  = numrows*numcols

    n_samples = round(length*percentage)
    print('Taking a stratified sample of ' + str(100*percentage) + 
          ' or ' + str(n_samples) + ' samples out of ' + str(length))

    save_coords = np.empty((0, 2))
    save_height = np.empty((0, 1))
    for height, value in tqdm(height_stratified.items()):
        
        X, Y = np.where(building_height == height)
        coords = np.column_stack((X, Y))
        
        take = len(coords)
        # take all buildings over 20 meteres
        if height <= 20:
            take = round(value*n_samples)
            idx_to_take = np.random.randint(len(coords), size=take)
            coords = coords[idx_to_take,:]
        
        
        save_coords = np.concatenate((save_coords, coords), axis=0)
        save_height = np.append(save_height, np.full((take, ), height))
        
        save_coords = save_coords.astype(int)
        save_height = save_height.astype(int)
        
    del building_height
        
    return pd.DataFrame(np.column_stack((save_coords, save_height)), columns=['X','Y','Height'])



def collect_data_from_image(gdal_obj, stratified_sampler):
    
    '''
    given a gdal object and the stratified sample dataframe
    this funciton will get these samples from every band in gdal obj
    and store them in a dataframe
    '''
    
    num_bands = gdal_obj.RasterCount
    
    X_coords = stratified_sampler['X']
    Y_coords = stratified_sampler['Y']
    
    data_store = pd.DataFrame([])
    for band in range(1, num_bands):
        
        raster_band = gdal_obj.GetRasterBand(band)
        col_name = raster_band.GetDescription()
        raster_array = raster_band.ReadAsArray()
        
        
        values = raster_array[X_coords, Y_coords]
        data_store[col_name] = values
        
    return data_store


def mask_building_height_with_settlement_map():
    pass


def loop_trhough_directory():
    pass


def check_if_all_paths_are_from_the_same_chip(list_arg_paths):


    def extract_tile(path_string):

        '''
        extract the corresponding tile from the image or directory path
        '''
        result = re.search('/X(.*)/', path_string)
        return result.group(1)

    
    tiles = []
    for path in list_arg_paths:
        tile = extract_tile(path)
        tiles.append(tile)

    if len(set(tiles)) != 1:
        print('One of the tiles does not match !!!  ' + str(tiles))
        return False

    else:
        return True


if __name__ == "__main__":

    building_height_file = sys.argv[1]
    sentinel_1_directory = sys.argv[2]
    sentinel_2_directory = sys.argv[3]
    settlement_map_file = sys.argv[4]

    print('The path to the building height file: '  + str(building_height_file))
    print('The Sentinel 1 path is: '  + str(sentinel_1_directory))
    print('The Sentinel 2 path is: ' + str(sentinel_2_directory))
    print('The path to the Settlement map path is: ' + str(settlement_map_file))

    list_arg_paths = [building_height_file, sentinel_1_directory,
                      sentinel_2_directory, settlement_map_file]


    flag = check_if_all_paths_are_from_the_same_chip(list_arg_paths)

    if not flag:
        sys.exit()