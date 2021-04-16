from osgeo import gdal
import pandas as pd
import numpy as np
import glob
import cv2
import os



def check_if_csv_file_exists(path):
        '''
        example if path='mydirectory/myfile.csv' exists
        return true.
        '''
        return os.path.exists(path)
    
    

def stack_images_into_volume(list_of_img_rasters):
    '''
    stacks numpy arrays together along z-axis to create
    multichannel image.
    '''
    return np.dstack(list_of_img_rasters)



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
                    new_filepath = os.path.join(segmented_tiles_dir_building, new_filename)
                    save_numpy_out(new_filepath, split_img)
                    
                    #output_view_gdal_height(split, vmin=0, MAX_HEIGHT=10)
       
    
    new_data = pd.DataFrame(rows, columns=['Tile_id','X1','X2','Y1','Y2'])
    
    if check_if_csv_file_exists(file_output_path):
        old_data = pd.read_csv(file_output_path)
        new_data = pd.concat([new_data, old_data], ignore_index=True, sort=False)
        new_data.to_csv(file_output_path ,index=False)
    
    else:
        print(f'No file found creating a new one at {file_output_path}')
        new_data.to_csv(file_output_path ,index=False)
        
        
    print(f'Successfuly Segmented {tile_id}')




if __name__ == "__main__":


    building_height_directory = 'C:/Users/egnke/PythonCode/MetEireann/Dublin_Height_Data/tiled/'
    sentinel_1_directory_desc = 'C:/Users/egnke/PythonCode/MetEireann/Sentinel-1-Data/Sentinel-1/Interpolation/Desc/'
    sentinel_1_directory_asc = 'C:/Users/egnke/PythonCode/MetEireann/Sentinel-1-Data/Sentinel-1/Interpolation/Asc/'
    sentinel_2_directory = 'C:/Users/egnke/PythonCode/MetEireann/Sentienl-2-Data/Processed_Data/interpolation/'
    settlement_map_dir = 'C:/Users/egnke/PythonCode/MetEireann/Settlement_Map/tiled/'

    csv_output_path = os.path.join(os.getcwd(), 'Unet-Data', 'building_height_coords.csv')
    segmented_tiles_dir_building = os.path.join(os.getcwd(), 'Unet-Data', 'Y')

    sub_dirs = ['X0002_Y0002','X0002_Y0003','X0003_Y0002','X0003_Y0003']


    for sub_dir in sub_dirs:

        building_height_dir = building_height_directory + sub_dir
        sentinel_1_desc_dir  = sentinel_1_directory_desc + sub_dir
        sentinel_1_asc_dir  = sentinel_1_directory_asc + sub_dir
        sentinel_2_dir  = sentinel_2_directory + sub_dir
        settle_map_dir  = settlement_map_dir + sub_dir

        # read in building height file and get raster
        building_height = glob.glob(building_height_dir+"/*.tif")[0].replace('\\','/')
        build_height = gdal.Open(building_height)
        height_data = build_height.GetRasterBand(1).ReadAsArray()

        create_csv_with_tiles_and_split_points(height_data, 250, 250, 0,
                                                sub_dir, csv_output_path, 
                                                segmented_tiles_dir=segmented_tiles_dir_building)