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


def collect_data_sample_from_file(file_path, stratified_sampler, add_to_name):
    
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
    for band in range(1, num_bands+1):
        
        raster_band = gdal_obj.GetRasterBand(band)
        col_name = raster_band.GetDescription()
        raster_array = raster_band.ReadAsArray()
        
        
        values = raster_array[X_coords, Y_coords]
        if not add_to_name:
            stratified_sampler[col_name+'_'+file_name] = values
        else:
            stratified_sampler[col_name + '_' + file_name + '_' + add_to_name] = values
    
    del gdal_obj

    return stratified_sampler


def mask_building_height_with_settlement_map(height_raster, settle_map_path):
    
    settlement = gdal.Open(settle_map_path)

    settlement_data = settlement.GetRasterBand(1).ReadAsArray()
    masked_height_data = np.where(settlement_data>2, height_raster, 0)

    return masked_height_data




def take_samples(directory, stratified_sample, add_to_name=''):
    '''
    loops through all .tif files in a directory
    and applies the collect data sample function
    '''

    for file_ in os.listdir(directory):

        if file_.endswith(".tif"):
            file_path = os.path.join(directory,file_)
            stratified_sample = collect_data_sample_from_file(file_path, stratified_sample, add_to_name)


    return stratified_sample


def create_directory_to_save_to(output_dir):
    '''
    checks to see if directory exists if it doesn't it
    will create it.
    '''
    if output_dir is not None and not os.path.exists(output_dir):
        print(f'creating the directory {output_dir} as it does not exist.')
        os.makedirs(output_dir)



def start_points(size, chunk_size, overlap=0):
    '''
    split an array into equal parts including overlap. used
    as the starting points then we add the size to each point to get
    the desired result.
    
    chunk_size: is the size of the stride length to take 
                i.e chunk_size = 2 [0,1,2,3] -> [0,2]
    overlap: is a float 0.5=50% of the overlap to take between chunk_sizes
            chunk_size:4, np.arange(10), overlap:0.5 [0, 4, 6] -> [0, 2, 4, 6]
    '''
    
    if(overlap>1 or overlap<0): raise ValueError('The Overlap Parameter must be between 0-1!')
    
    points = [0]
    stride = int(chunk_size * (1-overlap))
    counter = 1
    while True:
        
        pt = stride * counter
        
        if pt + chunk_size >= size:
            points.append(size - chunk_size)
            break
        else:
            points.append(pt)
        
        counter += 1
        
    return points



def save_numpy_out(filename, array):
    '''
    saves a numpy array out for later 
    filename: 'file.npy'
    '''
    with open(filename, 'wb') as f:
        np.save(f, array)



def create_csv_with_tiles_and_split_points(image_raster, split_width, 
                                           split_height, overlap,
                                           tile_id, file_output_path, 
                                           segmented_tiles_dir=None):
    '''
    Create a csv file that has the overlapping split points for each tile 
    X0002_X0002, X0002_X0003, X0003_X0002, X0003_X0003
    these split points are only for the sub images that have building heights
    in them.
    '''
    
    # check if the image is binary
    is_binary = np.array_equal(image_raster, image_raster.astype(bool))  
    # create directory if it doesn't exist
    create_directory_to_save_to(segmented_tiles_dir)
    
    img_w, img_h = image_raster.shape
    
    X_start_points = start_points(size=img_w,
                                  chunk_size=split_width,
                                  overlap=overlap)
    
    Y_start_points = start_points(size=img_h,
                                  chunk_size=split_height,
                                  overlap=overlap)
    
    rows = []
    for i in Y_start_points:
        for j in X_start_points:
            
            X1 = i
            X2 = (i+split_height)
            Y1 = j
            Y2 = (j+split_width)
            split_img = image_raster[X1:X2, Y1:Y2]
            
            if not is_binary:
                # check if not binary to calucate the area covered by buildings
                split_binary = np.where(split_img >=1, 1, 0)
                ratio_no_height = np.sum(split_binary)/(split_width*split_height)
                
            else:
                ratio_no_height = np.sum(split_img)/(split_width*split_height)
            
            
            if ratio_no_height > 0.05:
                
                rows.append([tile_id, X1, X2, Y1, Y2])
                
                if segmented_tiles_dir is not None:
                    # if a directory is passed then save the data out.
                    new_filename = f'Y_{tile_id}_{X1}_{X2}_{Y1}_{Y2}.npy'
                    new_filepath = os.path.join(segmented_tiles_dir, new_filename)
                    # expand dimension to include channel info from (250,250) -> (250,250,1)
                    split_img = np.expand_dims(split_img, axis=2)
                    save_numpy_out(new_filepath, split_img)
                    
       
    
    new_data = pd.DataFrame(rows, columns=['Tile_id','X1','X2','Y1','Y2'])
    
    if check_if_csv_file_exists(file_output_path):
        old_data = pd.read_csv(file_output_path)
        new_data = pd.concat([new_data, old_data], ignore_index=True, sort=False)
        new_data.to_csv(file_output_path ,index=False)
    
    else:
        print(f'No file found creating a new one at {file_output_path}')
        new_data.to_csv(file_output_path ,index=False)
        
        
    print(f'Successfuly Segmented {tile_id}')


def check_if_csv_file_exists(path):
        '''
        example if path='mydirectory/myfile.csv' exists
        return true.
        '''
        return os.path.exists(path)


def extract_morphology_operater(filepath):
    '''
    exctracts the morphology operator from the file name
    'C:/home/ubuntu/TEXTURE_HL_TXT_BHT.tif' -> _BHT
    need to remove this when comapring features to band description
    '''
    file_name = os.path.basename(filepath)
    clean_name = file_name.replace('.tif','')
    
    return '_'+clean_name.split('_')[-1]



def stack_images_into_volume(list_of_img_rasters):
    '''
    stacks numpy arrays together along z-axis to create
    multichannel image.
    '''
    return np.dstack(list_of_img_rasters)



def extract_bands(directory, img_bands, important_features):
    '''
    given a list of band names extract the relevant tif files from the directory 
    and stack them into a mutli channel image.
    '''
    

    for tif in glob.glob(directory + "/*.tif"):
        
        filepath = tif.replace('\\','/')
        morph = extract_morphology_operater(filepath)
    
        gdal_file = gdal.Open(filepath)
                
        for band in range(gdal_file.RasterCount):
            band += 1
            gdal_band = gdal_file.GetRasterBand(band)
            name = gdal_band.GetDescription() + morph

            if any(feat in name for feat in important_features):
                raster_band = gdal_band.ReadAsArray()
                img_bands.append(raster_band)
                
    return img_bands



def extract_bands_and_merge(sentinel_1_asc_dir, sentinel_2_dir, important_features):
    
    img_band_sent_2 = []
    img_band_sent_1 = []
    
    img_band_sent_2 = extract_bands(sentinel_2_dir, 
                                    img_band_sent_2, 
                                    important_features)
    
    img_band_sent_1 = extract_bands(sentinel_1_asc_dir, 
                                    img_band_sent_1,
                                    important_features)
    
    
    img = img_band_sent_1 + img_band_sent_2
    
    return stack_images_into_volume(img)


def split_image_from_csv_coords(csv_filepath, img, tile_id, output_dir):
    '''
    using the csv of tile coordinates created from create_csv_with_tiles_and_split_points()
    split the X features or images sentinel-1 and 2 so that they align with the target variable
    or building heights.s
    '''
    
    if not check_if_csv_file_exists(csv_filepath): raise ValueError('The file does not exist !')
    
    # create directory if it doesn't exist
    create_directory_to_save_to(output_dir)
    
    # filter to the appropriate tile
    coords = pd.read_csv(csv_filepath)
    coords = coords[coords['Tile_id'] == tile_id]
    
    for index, row in coords.iterrows():

        X1 = row['X1']
        X2 = row['X2']
        Y1 = row['Y1']
        Y2 = row['Y2']
        tile_id = row['Tile_id']
        
        if img.ndim > 2:
            split_img = img[X1:X2, Y1:Y2, :]
            
        else:
            split_img = img[X1:X2, Y1:Y2]
            
            
        filename = f'X_{tile_id}_{X1}_{X2}_{Y1}_{Y2}.npy'
        filepath = os.path.join(output_dir, filename)
        save_numpy_out(filepath, split_img)
        
        
    print(f'Succesfully subseted X feature tile {tile_id} variables using csv coords, saved output to {output_dir}')

        


if __name__ == "__main__":


    building_height_directory = 'C:/Users/egnke/PythonCode/MetEireann/Dublin_Height_Data/tiled/'
    sentinel_1_directory_desc = 'C:/Users/egnke/PythonCode/MetEireann/Sentinel-1-Data/Sentinel-1/Texture/Desc/'
    sentinel_1_directory_asc = 'C:/Users/egnke/PythonCode/MetEireann/Sentinel-1-Data/Sentinel-1/Texture/Asc/'
    sentinel_2_directory = 'C:/Users/egnke/PythonCode/MetEireann/Sentienl-2-Data/Processed_Data/morphology/'
    settlement_map_dir = 'C:/Users/egnke/PythonCode/MetEireann/Settlement_Map/tiled/'
    percentage = 0.25

 
    important_features = ['2020-2020_001-365_HL_TSA_SEN2L_NDV_STM_B0007_GRD', '2020-2020_001-365_HL_TSA_SEN2L_NDV_STM_B0007_ERO', 
                     '2020-2020_001-365_HL_TSA_VVVHP_BVH_STM_B0011_OPN', '2020-2020_001-365_HL_TSA_SEN2L_BNR_STM_B0002_CLS', 
                     '2020-2020_001-365_HL_TSA_VVVHP_BVH_STM_B0003_DIL', '2020-2020_001-365_HL_TSA_VVVHP_BVH_STM_B0003_GRD', 
                     '2020-2020_001-365_HL_TSA_VVVHP_BVH_STM_B0011_CLS', '2020-2020_001-365_HL_TSA_SEN2L_TCG_STM_B0008_OPN', 
                     '2020-2020_001-365_HL_TSA_SEN2L_BNR_STM_B0004_OPN', '2020-2020_001-365_HL_TSA_VVVHP_BVV_STM_B0001_DIL', 
                     '2020-2020_001-365_HL_TSA_VVVHP_BVH_STM_B0012_DIL', '2020-2020_001-365_HL_TSA_VVVHP_BVH_STM_B0006_GRD', 
                     '2020-2020_001-365_HL_TSA_SEN2L_SW1_STM_B0002_ERO', '2020-2020_001-365_HL_TSA_VVVHP_BVH_STM_B0007_DIL', 
                     '2020-2020_001-365_HL_TSA_VVVHP_BVH_STM_B0002_GRD', '2020-2020_001-365_HL_TSA_SEN2L_NDW_STM_B0001_DIL', 
                     '2020-2020_001-365_HL_TSA_SEN2L_BNR_STM_B0005_ERO', '2020-2020_001-365_HL_TSA_VVVHP_BVV_STM_B0005_GRD', 
                     '2020-2020_001-365_HL_TSA_SEN2L_GRN_STM_B0004_ERO', '2020-2020_001-365_HL_TSA_VVVHP_BVV_STM_B0002_DIL', 
                     '2020-2020_001-365_HL_TSA_VVVHP_BVV_STM_B0012_DIL', '2020-2020_001-365_HL_TSA_VVVHP_BVH_STM_B0001_DIL', 
                     '2020-2020_001-365_HL_TSA_SEN2L_SW1_STM_B0012_OPN', '2020-2020_001-365_HL_TSA_SEN2L_NIR_STM_B0003_ERO', 
                     '2020-2020_001-365_HL_TSA_VVVHP_BVH_STM_B0003_ERO', '2020-2020_001-365_HL_TSA_SEN2L_TCW_STM_B0012_ERO', 
                     '2020-2020_001-365_HL_TSA_VVVHP_BVH_STM_B0005_DIL', '2020-2020_001-365_HL_TSA_SEN2L_NIR_STM_B0007_ERO', 
                     '2020-2020_001-365_HL_TSA_SEN2L_BNR_STM_B0004_ERO', '2020-2020_001-365_HL_TSA_SEN2L_TCG_STM_B0006_ERO', 
                     '2020-2020_001-365_HL_TSA_VVVHP_BVV_STM_B0005_OPN', '2020-2020_001-365_HL_TSA_VVVHP_BVV_STM_B0007_DIL', 
                     '2020-2020_001-365_HL_TSA_SEN2L_NDB_STM_B0007_DIL', '2020-2020_001-365_HL_TSA_SEN2L_NIR_STM_B0004_ERO', 
                     '2020-2020_001-365_HL_TSA_SEN2L_NDW_STM_B0013_ERO', '2020-2020_001-365_HL_TSA_VVVHP_BVV_STM_B0001_CLS', 
                     '2020-2020_001-365_HL_TSA_SEN2L_GRN_STM_B0004_GRD', '2020-2020_001-365_HL_TSA_VVVHP_BVV_STM_B0001_OPN', 
                     '2020-2020_001-365_HL_TSA_SEN2L_NDW_STM_B0004_DIL', '2020-2020_001-365_HL_TSA_SEN2L_NIR_STM_B0012_OPN', 
                     '2020-2020_001-365_HL_TSA_SEN2L_SW2_STM_B0002_ERO', '2020-2020_001-365_HL_TSA_SEN2L_TCB_STM_B0009_DIL', 
                     '2020-2020_001-365_HL_TSA_SEN2L_NIR_STM_B0002_ERO', '2020-2020_001-365_HL_TSA_VVVHP_BVH_STM_B0006_CLS', 
                     '2020-2020_001-365_HL_TSA_SEN2L_NDW_STM_B0013_OPN', '2020-2020_001-365_HL_TSA_VVVHP_BVV_STM_B0003_DIL', 
                     '2020-2020_001-365_HL_TSA_SEN2L_TCW_STM_B0007_GRD', '2020-2020_001-365_HL_TSA_VVVHP_BVV_STM_B0011_CLS', 
                     '2020-2020_001-365_HL_TSA_VVVHP_BVV_STM_B0012_CLS', '2020-2020_001-365_HL_TSA_SEN2L_RE2_STM_B0011_GRD', 
                     '2020-2020_001-365_HL_TSA_VVVHP_BVH_STM_B0004_GRD', '2020-2020_001-365_HL_TSA_SEN2L_TCB_STM_B0012_CLS', 
                     '2020-2020_001-365_HL_TSA_VVVHP_BVH_STM_B0008_GRD', '2020-2020_001-365_HL_TSA_VVVHP_BVV_STM_B0012_GRD', 
                     '2020-2020_001-365_HL_TSA_SEN2L_RE1_STM_B0001_CLS', '2020-2020_001-365_HL_TSA_SEN2L_TCG_STM_B0003_OPN', 
                     '2020-2020_001-365_HL_TSA_SEN2L_TCG_STM_B0003_BHT', '2020-2020_001-365_HL_TSA_SEN2L_SW1_STM_B0003_ERO', 
                     '2020-2020_001-365_HL_TSA_SEN2L_NIR_STM_B0002_CLS', '2020-2020_001-365_HL_TSA_SEN2L_NDB_STM_B0011_CLS', 
                     '2020-2020_001-365_HL_TSA_VVVHP_BVH_STM_B0009_ERO', '2020-2020_001-365_HL_TSA_VVVHP_BVV_STM_B0007_CLS', 
                     '2020-2020_001-365_HL_TSA_SEN2L_SW1_STM_B0007_ERO', '2020-2020_001-365_HL_TSA_SEN2L_TCB_STM_B0013_ERO', 
                     '2020-2020_001-365_HL_TSA_SEN2L_SW1_STM_B0004_ERO', '2020-2020_001-365_HL_TSA_SEN2L_RE1_STM_B0011_ERO', 
                     '2020-2020_001-365_HL_TSA_VVVHP_BVV_STM_B0001_GRD', '2020-2020_001-365_HL_TSA_SEN2L_RE1_STM_B0011_OPN', 
                     '2020-2020_001-365_HL_TSA_VVVHP_BVH_STM_B0001_CLS', '2020-2020_001-365_HL_TSA_SEN2L_TCB_STM_B0004_ERO', 
                     '2020-2020_001-365_HL_TSA_SEN2L_NIR_STM_B0001_ERO', '2020-2020_001-365_HL_TSA_VVVHP_BVV_STM_B0007_OPN', 
                     '2020-2020_001-365_HL_TSA_VVVHP_BVH_STM_B0007_GRD', '2020-2020_001-365_HL_TSA_SEN2L_RE2_STM_B0008_ERO']


    sub_dirs = ['X0002_Y0002','X0002_Y0003','X0003_Y0002']


    list_of_dfs = []
    for sub_dir in tqdm(sub_dirs):

        print('\n')
        building_height_dir = building_height_directory + sub_dir
        sentinel_1_desc_dir  = sentinel_1_directory_desc + sub_dir
        sentinel_1_asc_dir  = sentinel_1_directory_asc + sub_dir
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



        stratified_sample = take_samples(sentinel_1_asc_dir, stratified_sample, 'asc')
        stratified_sample = take_samples(sentinel_1_desc_dir, stratified_sample, 'desc')
        stratified_sample = take_samples(sentinel_2_dir, stratified_sample)

        stratified_sample['tile'] = sub_dir
        list_of_dfs.append(stratified_sample)

    
    startified_sample = pd.concat(list_of_dfs)
    startified_sample.to_csv('building_height_sample_asc.csv', index=False)
    
    ##########################################################################################################
    ##########################################################################################################

    # getting the test dataset in the same format as unet.
    test_dir = 'X0003_Y0003'
    csv_output_path = 'C:/Users/egnke/PythonCode/Met_Eireann_git/Ireland_building_heights/Modelling/DataSetCreation/Pixel-Wise-Data/pixel_building_height_testing_coords.csv'
    segmented_tiles_dir_Y = 'C:/Users/egnke/PythonCode/Met_Eireann_git/Ireland_building_heights/Modelling/DataSetCreation/Pixel-Wise-Data/y_test/'
    segmented_tiles_dir_X = 'C:/Users/egnke/PythonCode/Met_Eireann_git/Ireland_building_heights/Modelling/DataSetCreation/Pixel-Wise-Data/X_test/'

    building_height_dir = building_height_directory + test_dir
    sentinel_1_asc_dir  = sentinel_1_directory_asc + test_dir
    sentinel_2_dir  = sentinel_2_directory + test_dir
    settle_map_dir  = settlement_map_dir + test_dir

    building_height = glob.glob(building_height_dir+"/*.tif")[0].replace('\\','/')
    build_height = gdal.Open(building_height)
    height_data = build_height.GetRasterBand(1).ReadAsArray()

    create_csv_with_tiles_and_split_points(height_data, 250, 250, 0,
                                           test_dir, csv_output_path, 
                                           segmented_tiles_dir=segmented_tiles_dir_Y)

    img = extract_bands_and_merge(sentinel_1_asc_dir, sentinel_2_dir, important_features)

    split_image_from_csv_coords(csv_output_path, img, test_dir, segmented_tiles_dir_X)


    
    







    
