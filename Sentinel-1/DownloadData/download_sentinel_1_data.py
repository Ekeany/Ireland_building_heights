from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
import numpy as np
import pandas as pd
from datetime import date
from pathlib import Path
from shapely.geometry import Polygon
import os
import sys

import warnings
warnings.filterwarnings("ignore")


def get_products(orbit, start_date, end_date):
    
    
    if orbit == 'DESC':

        print('Only Taking Descending Orbit')
        products = api.query(area=footprint,
                             date=(start_date, end_date),
                             platformname='Sentinel-1',
                             producttype='GRD',
                             sensoroperationalmode='IW',
                             orbitdirection='DESCENDING'
                            )


    elif orbit == 'ASC':

        print('Only Taking Ascending Orbit')
        products = api.query(area=footprint,
                             date=(start_date, end_date),
                             platformname='Sentinel-1',
                             producttype='GRD',
                             sensoroperationalmode='IW',
                             orbitdirection='ASCENDING'
                            )
        
    else:

        products = api.query(area=footprint,
                             date=(start_date, end_date),
                             platformname='Sentinel-1',
                             producttype='GRD',
                             sensoroperationalmode='IW'
                            )
        
    
    
    return products



def get_polygon(string_):
    
    between_brackets = string_[string_.find("(")+3:string_.find(")))")]
    
    long_lat = []
    for str_coord in between_brackets.split(','):

        long_lat.append(str_coord.lstrip().split(' '))
    
    long_lat = [list(map(float, sublist)) for sublist in long_lat]
    
    poly = []
    for elem in long_lat:
        x, y = elem
        poly.append((x,y))

    
    return Polygon(poly)


def get_interesction(poly1, poly2):
    # dividing by the area for completely interected polygons
    try:
        return poly2.intersection(poly1).area
    
    except:
        return -1


def normalize_the_intersection(df, column):
    maximum =  df[column].max()
    df[column] = df[column]/maximum
    return df
    


def filter_sentinel_data_by_area(df):
    
    og_length = len(df)
    
    filtered_df = df[df['intersection_area'] == 1.0]
    
    new_length = len(filtered_df)
    print('Filtering by area ....')
    print('Original Length was:' + str(og_length))
    print('New Length is: ' + str(new_length))
    
    return filtered_df

def get_year_and_month(x):
    date_str = str(x).split('-')
    year = date_str[0]
    month = date_str[1]
    day = date_str[2][:2]
    return year, month, day


def only_take_five_sensors_for_each_month(df):
    
    '''
    For each month approximatly 16 sensors records we subset this to only 5
    these 5 records are eqaully scpaed throught the month though.
    '''
    
    def pct25(s):
        return s.iloc[int(len(s) * .25)]
    
    def pct50(s):
        return s.iloc[int(len(s) * .50)]


    def pct75(s):
        return s.iloc[int(len(s) * .75)]
    
    
    dates_to_keep = df.sort_values(by='beginposition').groupby('month').agg({'day': ['first', pct25, pct50, pct75 ,'last']}).unstack().reset_index()
    dates_to_keep.rename(columns={0: "day"}, inplace=True)
    

    filtered_df =  df.merge(dates_to_keep[['month','day']], on=['month','day'], how='inner')

    print('Filtering by time 5 equally spaced products per month ....')
    print('Original Length was:' + str(len(df)))
    print('New Length is: ' + str(len(filtered_df)))

    return filtered_df


def create_directory_to_save_to(output_path):
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        



def create_append_to_queue(filename, object_to_write):

    # Open the file in append & read mode ('a+')
    with open(filename, "a+") as file_object:
        # Move read cursor to the start of file.
        file_object.seek(0)
        # If file is not empty then append '\n'
        data = file_object.read(100)
        if len(data) > 0 :
            file_object.write("\n")
        # Append text at the end of file
        file_object.write(object_to_write)
        


def download_data(df, api, queue_name):
               

    def download_(df, queue_name):

        for _, row in df.iterrows():
            api.download(row['uuid'], directory_path=row['save_to'])
            create_append_to_queue(queue_name, row['save_to']+'/'+row['title']+'/'+row['title']+'.SAFE'+' QUEUED')
    

    
    def filter_out_already_downloaded(df, queue_name):

        already_downloaded = check_if_file_queue_exits(queue_name)
        already_downloaded = [line.replace(' QUEUED','') for line in already_downloaded]

        if len(already_downloaded) > 0:
            df = df[~df['title'].isin(already_downloaded)]

        return df

    df = filter_out_already_downloaded(df, queue_name)
    print('Begining Download: ' + str(len(df)) + ' products found')
    download_(df, queue_name)




def check_if_file_queue_exits(filename):
    
    files = []

    if Path(filename).is_file():

        with open(filename) as f:
            lines = f.readlines()
        
        for line in lines:
            file_ = line.split('/')[-1].replace('.SAFE','').replace('\n','')
            files.append(file_)

    return files



if __name__ == '__main__':


    orbit = sys.argv[1]
    time_filter = sys.argv[2]
    output_path = sys.argv[3]
    print('Orbit Selected: '   + str(orbit))
    print('Filtering by Time: ' + str(time_filter))
    print('Output Path: ' + str(output_path))

    #Download
    api = SentinelAPI(user='ekeany',
                  password='carrontlieve1',
                  api_url='https://scihub.copernicus.eu/dhus',
                  show_progressbars=True)


    footprint = geojson_to_wkt(read_geojson('Dublin_Area.geojson'))

    start_date = date(2020,1,1)
    end_date = date(2020,12,31)

    products = get_products(orbit=orbit, start_date=start_date, end_date=end_date)
    products_df = api.to_dataframe(products)


    # Filter by Area
    dublin_poly = Polygon([(-6.6663, 53.5769),(-6.4968, 53.1487),(-5.8961, 53.2305),(-6.0599, 53.6596)])
    
    products_df['intersection_area'] = products_df.footprint.apply(lambda x: get_interesction(get_polygon(x), dublin_poly))
    products_df = normalize_the_intersection(products_df, 'intersection_area')

    filtered_df = filter_sentinel_data_by_area(products_df)


    # filter number of total records 
    if time_filter == 'yes':

        filtered_df['year'], filtered_df['month'], filtered_df['day'] = zip(*filtered_df['beginposition'].apply(lambda date_: get_year_and_month(date_)))
        filtered_df = only_take_five_sensors_for_each_month(filtered_df)


    # create directory and save prodcuts
    create_directory_to_save_to(output_path)
    filtered_df['save_to'] = filtered_df.apply(lambda x: output_path, axis=1)

    download_data(filtered_df, api, output_path + '/Dublin_Queue.txt')
