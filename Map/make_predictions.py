from osgeo import gdal
from functools import reduce
import pickle
import numpy as np
import pandas as pd
import re
import os


def load_model(pkl_filename):
    '''
    load model object from pickle file
    '''

    with open(pkl_filename, 'rb') as file:
        pickle_model = pickle.load(file)
        
    return pickle_model


def get_file_name(file_path):
    '''
    gets the folder name of the morhpology operater for example if 
    we are in the dilated folder -> DIL erosion -> ERO
    '''
    file_name = os.path.basename(file_path)
    clean_name = file_name.replace('.tif','')
    
    name_pieces = clean_name.split('_')

    if name_pieces[-1] == 'STM':
        return name_pieces[-2]

    elif name_pieces[-2] == 'TXT':
        return name_pieces[-1]

    else:
        return clean_name



def extract_band_names_values(file_path, columns):
    '''
    given a tif file extract the band names
    and values
    '''

    gdal_obj = gdal.Open(file_path)
    num_bands = gdal_obj.RasterCount

    file_name = get_file_name(file_path)
    dfs = []
    names = []
    for band in range(1, num_bands+1):

        raster_band = gdal_obj.GetRasterBand(band)
        col_name = raster_band.GetDescription()

        if (col_name + '_' + file_name) in columns:
            raster_array = raster_band.ReadAsArray()
            dfs.append(convert_2d_array_to_dataframe(raster_array, col_name + '_' + file_name))
            names.append(col_name+'_'+file_name)


    if len(dfs) > 0:
        return merge_list_of_dataframes(dfs), names

    else:
        return None, None
   


def convert_2d_array_to_dataframe(raster, colname):
    
    '''
    converts a raster to dataframe where each pixel value is a row and the x y coordinates are there
    
    np.array([[1,2,3],[4,5,6],[7,8,9]]) ->  y,x,value
                                            0,0,1
                                            0,1,2
                                            0,2,3
                                            1,0,4
                                            1,1,5
                                            1,2,6
                                            2,0,7
                                            2,1,8
                                            2,2,9
    '''
    
    return (pd.DataFrame(raster)
             .stack()
             .rename_axis(['y', 'x'])
             .reset_index(name=colname))


def convert_df_to_numpy(df, colname):
    '''
    returns the convert_2d_array_to_dataframe datframe back to its original form
    '''
    df.sort_values(by=['y','x'], inplace=True)
    return df.pivot('y', 'x', colname).values



def flatten_list_of_lists(array):
    return [item for sublist in array for item in sublist]


def loop_through_tile_folder(directory, columns):
    '''
    loops through every file in a tile folder X0001_Y0001
    and combines the raster bands from each morphology tif file 
    in the folder together and also documents the names of each band or feature.
    '''
    
    tile_dfs = []
    tile_feature_names = []
    for file_ in os.listdir(directory):
        
        if file_.endswith(".tif"):
            
            file_path = os.path.join(directory,file_)
            df, names = extract_band_names_values(file_path, columns)
            
            if df is not None:
                tile_dfs.append(df)
                tile_feature_names.append(names)
    
    tile_feature_names = flatten_list_of_lists(tile_feature_names)
    
    return merge_list_of_dataframes(tile_dfs), tile_feature_names



def make_segment_wide_prediction(img_volume, settlement_map, model, features, feature_order):
    '''
    loop through an image pixel by pixel and make predictions
    save these predictions in a numpy array
    '''

    channel, width, height = img_volume.shape
    predictions = np.zeros((width, height))
    
    for point_X in range(width):
        for point_Y in range(height):
        
            # if building height map non zero than make a prediction
            if settlement_map[point_X,point_Y] > 0:
                
                pixel = img_volume[:,point_X,point_Y]
                pixel_data = pd.DataFrame(data=[pixel], columns=features)[feature_order]
                predictions[point_X, point_Y] = model.predict(pixel_data)
    
    
    return predictions



def make_predictions_using_df(tile_df, settlement_raster, model, model_features):
    '''
    makes predictions by converting the image to a df make predicitions
    then convert the df back to an image..
    '''

    # convert settlement to df
    settle_df = convert_2d_array_to_dataframe(settlement_raster, 'settlement_value')
    merged_df = pd.merge(tile_df, settle_df, on=['y','x'], how='inner')

    # filter data to where buildings are 
    buildings = merged_df[merged_df['settlement_value'] > 2]

    # make predictions
    X = buildings.drop(['y','x','settlement_value'], axis=1)
    buildings['building_heights'] = model.predict(X[model_features])

    # merge predicitons back to df and fill in nas with a building height of zero
    predictions_df = pd.merge(tile_df, buildings[['y','x','building_heights']], 
                              on=['y','x'], how='left')

    predictions_df['building_heights'] = predictions_df['building_heights'].fillna(0)
    predictions_df['building_heights'] = (np.where((predictions_df.isin([-9999]).any(axis=1)) & 
                                                    (predictions_df['building_heights'] != 0), 
                                                    -100, predictions_df['building_heights']))
    
    img = convert_df_to_numpy(predictions_df, 'building_heights')

    return img



def make_predictions(folder_path_tiles, folder_path_settle, 
                    output_folder, model, model_features):
    '''
    find coresspinding settlement map
    
    folder_path_tiles: path to the folder that stores all the tiles
                       for the morpholgy output.
                       
    folder_path_settle: path to the folder that stores all the tiles
                        for the settlement map.
    '''
    
    for root, folders, files in os.walk(folder_path_tiles):
        
        for folder in folders:

            path_to_folder = os.path.join(folder_path_tiles, folder)
            tile = extract_what_tile(path_to_folder)
            settlement_tile, source = extract_settlement_tile(folder_path_settle, tile)

            if settlement_tile is not None:

                tile_df, tile_feature_names = loop_through_tile_folder(path_to_folder, columns=model_features)

                predictions = make_predictions_using_df(tile_df, settlement_tile, model, model_features)
                
                [cols, rows] = settlement_tile.shape
                
                filename = output_folder + tile + '.tif'
                save_predictions(filename, source, predictions, rows, cols)
        
        
        

def save_predictions(filename, source, predictions, rows, cols):
    """
    save the predcitions in the same projection 
    """

    driver = gdal.GetDriverByName("GTiff")
    driver.Register()
    outdata = driver.Create(filename, xsize=rows, ysize=cols, bands=1, eType=gdal.GDT_Int16)

    outdata.SetGeoTransform(source.GetGeoTransform())##sets same geotransform as input
    outdata.SetProjection(source.GetProjection())##sets same projection as input

    outdata.GetRasterBand(1).WriteArray(predictions)
    outdata.GetRasterBand(1).SetDescription('Building Heights')
    outdata.GetRasterBand(1).SetNoDataValue(-9999)##if you want these values transparent
    
    outdata.FlushCache() ##saves to disk!!
    
    outdata = None
    predictions   = None
    source  = None
        

        

def extract_settlement_tile(folder_path_settle, tile):
    
    '''
    given a tile eg. X0002_Y0003 find the corresponding
    settlement tile and return as array
    '''

    for root, folders, files in os.walk(folder_path_settle):
        
        for folder in folders:
            
            path_to_folder = os.path.join(folder_path_settle, folder)
            
            if tile in path_to_folder:

                for file_ in os.listdir(path_to_folder):
            
                    if file_.endswith(".tif"):

                        file_path = os.path.join(path_to_folder, file_)
                    

                    source = gdal.Open(file_path)
                    raster_band = source.GetRasterBand(1)
                    settle_img = raster_band.ReadAsArray()
                    
                    return settle_img, source

            else:
                return None, None
            
    

def merge_list_of_dataframes(list_of_dfs):
    '''
    converts a list of dfs into a single df with columns
    x,y,val1 + x,y,val2 -> x,y,val1,val2 
    '''
    return reduce(lambda x, y: pd.merge(x, y, on = ['y','x']), list_of_dfs)



def extract_what_tile(file_path):
    '''
    given a file path this function will extract the tile name
    '''

    x = re.search(r'X\d{4}_Y\d{4}', file_path)
    
    if not None:
        return x[0]
    
    else:
        return 0

    
if __name__ == "__main__":

    model_path = 'C:/Users/egnke/PythonCode/Ireland_building_heights/Map/final_model.pkl'
    output_folder = 'C:/Users/egnke/PythonCode/Ireland_building_heights/Map/test_preds/'
    folder_to_sentinel2_tiles = 'C:/Users/egnke/PythonCode/MetEireann/Sentienl-2-Data/Processed_Data/morphology/'
    folder_to_settlement_tiles = 'C:/Users/egnke/PythonCode/MetEireann/Settlement_Map/tiled/'

    model_features = ['2020-2020_001-365_HL_TSA_SEN2L_NDV_STM_B0007_GRD', '2020-2020_001-365_HL_TSA_SEN2L_NDV_STM_B0007_ERO', 
                     '2020-2020_001-365_HL_TSA_SEN2L_BNR_STM_B0002_CLS', '2020-2020_001-365_HL_TSA_SEN2L_TCG_STM_B0008_OPN',
                     '2020-2020_001-365_HL_TSA_SEN2L_BNR_STM_B0004_OPN', '2020-2020_001-365_HL_TSA_SEN2L_SW1_STM_B0002_ERO', 
                     '2020-2020_001-365_HL_TSA_SEN2L_NDW_STM_B0001_DIL', '2020-2020_001-365_HL_TSA_SEN2L_BNR_STM_B0005_ERO', 
                     '2020-2020_001-365_HL_TSA_SEN2L_GRN_STM_B0004_ERO', '2020-2020_001-365_HL_TSA_SEN2L_SW1_STM_B0012_OPN', 
                     '2020-2020_001-365_HL_TSA_SEN2L_NIR_STM_B0003_ERO', '2020-2020_001-365_HL_TSA_SEN2L_TCW_STM_B0012_ERO', 
                     '2020-2020_001-365_HL_TSA_SEN2L_NIR_STM_B0007_ERO', '2020-2020_001-365_HL_TSA_SEN2L_BNR_STM_B0004_ERO',
                     '2020-2020_001-365_HL_TSA_SEN2L_TCG_STM_B0006_ERO', '2020-2020_001-365_HL_TSA_SEN2L_NDB_STM_B0007_DIL',
                     '2020-2020_001-365_HL_TSA_SEN2L_NIR_STM_B0004_ERO', '2020-2020_001-365_HL_TSA_SEN2L_NDW_STM_B0013_ERO',
                     '2020-2020_001-365_HL_TSA_SEN2L_GRN_STM_B0004_GRD', '2020-2020_001-365_HL_TSA_SEN2L_NDW_STM_B0004_DIL',
                     '2020-2020_001-365_HL_TSA_SEN2L_NIR_STM_B0012_OPN', '2020-2020_001-365_HL_TSA_SEN2L_SW2_STM_B0002_ERO', 
                     '2020-2020_001-365_HL_TSA_SEN2L_TCB_STM_B0009_DIL', '2020-2020_001-365_HL_TSA_SEN2L_NIR_STM_B0002_ERO',
                     '2020-2020_001-365_HL_TSA_SEN2L_NDW_STM_B0013_OPN', '2020-2020_001-365_HL_TSA_SEN2L_TCW_STM_B0007_GRD', 
                     '2020-2020_001-365_HL_TSA_SEN2L_RE2_STM_B0011_GRD', '2020-2020_001-365_HL_TSA_SEN2L_TCB_STM_B0012_CLS', 
                     '2020-2020_001-365_HL_TSA_SEN2L_RE1_STM_B0001_CLS', '2020-2020_001-365_HL_TSA_SEN2L_TCG_STM_B0003_OPN', 
                     '2020-2020_001-365_HL_TSA_SEN2L_TCG_STM_B0003_BHT', '2020-2020_001-365_HL_TSA_SEN2L_SW1_STM_B0003_ERO', 
                     '2020-2020_001-365_HL_TSA_SEN2L_NIR_STM_B0002_CLS', '2020-2020_001-365_HL_TSA_SEN2L_NDB_STM_B0011_CLS', 
                     '2020-2020_001-365_HL_TSA_SEN2L_SW1_STM_B0007_ERO', '2020-2020_001-365_HL_TSA_SEN2L_TCB_STM_B0013_ERO', 
                     '2020-2020_001-365_HL_TSA_SEN2L_SW1_STM_B0004_ERO', '2020-2020_001-365_HL_TSA_SEN2L_RE1_STM_B0011_ERO', 
                     '2020-2020_001-365_HL_TSA_SEN2L_RE1_STM_B0011_OPN', '2020-2020_001-365_HL_TSA_SEN2L_TCB_STM_B0004_ERO', 
                     '2020-2020_001-365_HL_TSA_SEN2L_NIR_STM_B0001_ERO',  '2020-2020_001-365_HL_TSA_SEN2L_RE2_STM_B0008_ERO']

    model = load_model(model_path)

    make_predictions(folder_to_sentinel2_tiles, folder_to_settlement_tiles, 
                     output_folder, model, model_features)


    