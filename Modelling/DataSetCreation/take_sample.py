from multiprocessing import Pool
from osgeo import gdal
import numpy as np
import random
import pandas as pd
from tqdm import tqdm
import glob
import sys
import re
import os


def compute_height_distribtuion(building_height_path, directory):
    '''
    calculates the distribution of pixel values given a 
    raster image.
    
    plots a histogram and returns dictionary of pixel value 
    and percentage it occurs
    '''
    
    print('Getting the building height distribution for this raster: ' + str(directory) + '....')
    total_building_height = gdal.Open(building_height_path)
    height_data = total_building_height.GetRasterBand(1).ReadAsArray()
    
    flat_height = height_data.flatten()
    # remove all 0s and no data values
    flat_height = flat_height[flat_height > 0]
    
    total_num_pixels = len(flat_height)
    height_dist = {}
    for height in np.unique(flat_height):
        
        num_pixels = sum(flat_height == height)
        height_dist[height] = round(num_pixels/total_num_pixels,3)

    
    del total_building_height
    
    return height_dist
    

    
def stratified_height_sample(building_height_path,height_stratified,percentage,
                            settle_map_path, apply_settlement_mask=False):
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

    if apply_settlement_mask:
        building_height = mask_building_height_with_settlement_map(building_height, settle_map_path)
    
    
    flat_height = building_height.flatten()
    # remove all 0s and no data values
    flat_height = flat_height[flat_height > 0]  
    length  = len(flat_height)


    n_samples = round(length*percentage)
    print('Taking a stratified sample of ' + str(100*percentage) + 
          '% Or ' + str(n_samples) + ' samples out of ' + str(length))

    save_coords = np.empty((0, 2))
    save_height = np.empty((0, 1))
    for height, value in height_stratified.items():
        
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


def get_file_name(file_path):
    file_name = os.path.basename(file_path)
    clean_name = file_name.replace('.tif','')
    
    name_pieces = clean_name.split('_')

    if name_pieces[-1] == 'STM':
        return name_pieces[-2]

    elif name_pieces[-2] == 'TXT':
        return name_pieces[-1]

    else:
        return clean_name


def collect_data_sample_from_file(file_path, stratified_sampler, feature_name=False):
    
    '''
    given a gdal object and the stratified sample dataframe
    this funciton will get these samples from every band in gdal obj
    and store them in a dataframe
    '''
    gdal_obj = gdal.Open(file_path)
    num_bands = gdal_obj.RasterCount
    
    X_coords = stratified_sampler['X']
    Y_coords = stratified_sampler['Y']

    file_name = get_file_name(file_path)
    for band in range(1, num_bands):
        
        raster_band = gdal_obj.GetRasterBand(band)
        col_name = raster_band.GetDescription()
        raster_array = raster_band.ReadAsArray()
        
        
        values = raster_array[X_coords, Y_coords]
        stratified_sampler[col_name+'_'+file_name] = values
    
    del gdal_obj


    
    return stratified_sampler



def mask_building_height_with_settlement_map(height_raster, settle_map_path):
    
    settlement = gdal.Open(settle_map_path)

    settlement_data = settlement.GetRasterBand(1).ReadAsArray()
    masked_height_data = np.where(settlement_data>2, height_raster, 0)

    return masked_height_data




def take_samples(directory, stratified_sample):
    '''
    loops through all .tif files in a directory
    and applies the collect data sample function
    '''

    for file_ in os.listdir(directory):

        if file_.endswith(".tif"):
            file_path = os.path.join(directory,file_)
            stratified_sample = collect_data_sample_from_file(file_path, stratified_sample)


    return stratified_sample



if __name__ == "__main__":


    building_height_directory = 'C:/Users/egnke/PythonCode/MetEireann/Dublin_Height_Data/tiled/'
    sentinel_1_directory = 'C:/Users/egnke/PythonCode/MetEireann/Sentinel-1-Data/Sentinel-1/Texture/Desc/'
    sentinel_2_directory = 'C:/Users/egnke/PythonCode/MetEireann/Sentienl-2-Data/Processed_Data/morphology/'
    settlement_map_dir = 'C:/Users/egnke/PythonCode/MetEireann/Settlement_Map/tiled/'
    percentage = 0.1


    sub_dirs = ['X0002_Y0002','X0002_Y0003','X0003_Y0002','X0003_Y0003' ]


    list_of_dfs = []
    for sub_dir in tqdm(sub_dirs):


        building_height_dir = building_height_directory + sub_dir
        sentinel_1_dir  = sentinel_1_directory + sub_dir
        sentinel_2_dir  = sentinel_2_directory + sub_dir
        settle_map_dir  = settlement_map_dir + sub_dir

        #  get single tif file in each tiles dircectory
        building_height = glob.glob(building_height_dir+"/*.tif")[0].replace('\\','/')
        settlement_map  = glob.glob(settle_map_dir+"/*.tif")[0].replace('\\','/')



        height_distribution = compute_height_distribtuion(building_height, sub_dir)
        stratified_sample  = stratified_height_sample(building_height, 
                                                    height_distribution, 
                                                    percentage,
                                                    settlement_map,
                                                    apply_settlement_mask=False)



        stratified_sample = take_samples(sentinel_1_dir, stratified_sample)
        stratified_sample = take_samples(sentinel_2_dir, stratified_sample)

        stratified_sample['tile'] = sub_dir
        list_of_dfs.append(stratified_sample)

    
    startified_sample = pd.concat(list_of_dfs)
    startified_sample.to_csv('building_height_sample.csv', index=False)




    
