from osgeo import gdal
import pickle
import numpy as np
import re


def load_model(pkl_filename):
    '''
    load model object from pickle file
    '''

    with open(pkl_filename, 'rb') as file:
        pickle_model = pickle.load(file)
        
    return pickle_modelf


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



def extract_band_names_values(file_path):
    '''
    given a tif file extract the band names
    and values
    '''

    gdal_obj = gdal.Open(file_path)
    num_bands = gdal_obj.RasterCount

    file_name = get_file_name(file_path)
    rasters = []
    names = []
    for band in range(1, num_bands+1):

        raster_band = gdal_obj.GetRasterBand(band)
        col_name = raster_band.GetDescription()
        raster_array = raster_band.ReadAsArray()

        rasters.append(raster_array)
        names.append(col_name+'_'+file_name)
        
    return rasters, names
   

def flatten_list_of_lists(array):
    return [item for sublist in array for item in sublist]


def loop_through_tile_folder(file_path):
    '''
    loops through every file in a tile folder X0001_Y0001
    and combines the raster bands from each morphology tif file 
    in the folder together and also documents the names of each band or feature.
    '''
    
    tile_rasters = []
    tile_feature_names = []
    for file_ in os.listdir(directory):
        
        if file_.endswith(".tif"):
            
            file_path = os.path.join(directory,file_)
            rasters, names = extract_band_names_values(file_path)
            
            tile_rasters.append(rasters)
            tile_feature_names.append(names)
    
    tile_rasters = flatten_list_of_lists(tile_rasters)
    tile_feature_names = flatten_list_of_lists(tile_feature_names)
    
    return np.array(tile_rasters), tile_feature_names



def make_segment_wide_prediction(img_volume, settlement_map, model, features, feature_order):
    '''
    loop through an image pixel by pixel and make predictions
    save these predictions in a numpy array
    '''
    width, height, channels = img_volume.shape
    predictions = np.zeros((width, height, 1))
    
    for point_X in range(width):
        for point_Y in range(height):
        
            # if building height map non zero than make a prediction
            if settlement_map[point_X,point_Y,0] > 0:
                
                pixel = img_volume[point_X,point_Y,:]
                pixel_data = pd.DataFrame(data=[pixel], columns=features)[feature_order]
                predictions[point_X, point_Y,0] = model.predict(pixel_data)
    
    
    return predictions



def loop_through_all_tiles_in_folder(folder_path_tiles, folder_path_settle, model, model_features):
    '''
    find coresspinding settlement map
    
    folder_path_tiles: path to the folder that stores all the tiles
                       for the morpholgy output.
                       
    folder_path_settle: path to the folder that stores all the tiles
                        for the settlement map.
    '''
    
    for folder in os.listdir(folder_path_tiles):
        
        tile = extract_what_tile(folder)
        settlement_tile, source = extract_settlement_tile(folder_path_settle, tile)
        tile_rasters, tile_feature_names = loop_through_tile_folder(folder)
        
        prediction = make_segment_wide_prediction(tile_rasters, 
                                                  settlement_tile, 
                                                  model, tile_feature_names, 
                                                  model_features)
        
        [cols, rows] = settlement_tile.shape
        
        
        

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

    for folder in os.listdir(folder_path_settle):
        
        if tile in folder:
            
            for file_ in os.listdir(folder):
        
                if file_.endswith(".tif"):

                    file_path = os.path.join(folder, file_)
                    
    source = gdal.Open(file_path)
    raster_band = source.GetRasterBand(1)
    settle_img = raster_band.ReadAsArray()
    
    return settle_img, source
            
    

def extract_what_tile(file_path):
    '''
    given a file path this function will extract the tile name
    '''
    
    x = re.search(r'X\d{4}_Y\d{4}', file_path)
    
    if not None:
        return x[0]
    
    else:
        return 0