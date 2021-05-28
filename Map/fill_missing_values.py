import os
from osgeo import gdal
import numpy as np
import pandas as pd
from statistics import mean
from tqdm import tqdm



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



def average_nearest_neighbours(a, x, y, k=10):
    '''
    finds the average value of the k nearest non zero or missing neighbours
    
    '''
    tmp_array = a.copy()

    # convert missing values -100 to nan
    tmp_array[tmp_array < 0] = 0
    tmp_array[int(x), int(y)] = 0
    
    #where_are_NaNs = np.isnan(tmp_array)
    #tmp_array[where_are_NaNs] = 0
    
    r,c = np.nonzero(tmp_array)

    closest_idx = np.argsort(((r - x)**2 + (c - y)**2))[:k]
    
    values = []
    for idx in closest_idx:
        x_, y_ = r[idx], c[idx]
        values.append(float(a[x_, y_]))
        
    if len(values) > 0:
        return mean(values)

    else:
        return 4.95



def fill_missing_values(filepath):
    
    product = gdal.Open(filepath)
    band_data = product.GetRasterBand(1).ReadAsArray()
    band_data = band_data.astype('float')

    tile_df = convert_2d_array_to_dataframe(band_data, 'building_height_preds')
    missing_values = tile_df[tile_df['building_height_preds'] < 0]

    if len(missing_values) > 0:

        missing_values['building_height_filled'] = (missing_values.apply(lambda row: 
                                                        average_nearest_neighbours(band_data, row['x'], 
                                                                                   row['y'], k=10)
                                                                        , axis=1))

        tile_df = pd.merge(tile_df, missing_values[['y','x','building_height_filled']], 
                        on=['y','x'], how='left')


        tile_df['building_height_filled'] = (tile_df['building_height_filled']
                                            .fillna(tile_df['building_height_preds']))

        return convert_df_to_numpy(tile_df, colname='building_height_filled'), product


    else:

        return band_data, product




def save_gdal(filepath, source, array, rows, cols):
    """
    save the predcitions in the same projection 
    """

    driver = gdal.GetDriverByName("GTiff")
    driver.Register()
    outdata = driver.Create(filepath, xsize=rows, ysize=cols, bands=1, eType=gdal.GDT_Int16)

    outdata.SetGeoTransform(source.GetGeoTransform())##sets same geotransform as input
    outdata.SetProjection(source.GetProjection())##sets same projection as input

    outdata.GetRasterBand(1).WriteArray(array)
    outdata.GetRasterBand(1).SetDescription('Building Heights')
    outdata.GetRasterBand(1).SetNoDataValue(-9999)##if you want these values transparent
    
    outdata.FlushCache() ##saves to disk!!
    
    outdata = None
    array   = None
    source  = None



if __name__ == "__main__":


    input_folder  = '/home/ubuntu/newvolume/combined_tiles/'
    output_folder = '/home/ubuntu/newvolume/cleaned_combined_tiles/'

    for filename in tqdm(os.listdir(input_folder)):
        
        if filename.endswith(".tif"):
            
            filepath = os.path.join(input_folder, filename)
            outputpath = os.path.join(output_folder, filename)

            filled_scene, source = fill_missing_values(filepath)
            
            rows, cols = filled_scene.shape
            save_gdal(outputpath, source, filled_scene, rows, cols)
