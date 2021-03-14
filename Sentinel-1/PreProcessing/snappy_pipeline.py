from __future__ import division

from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
from zipfile import ZipFile
import snappy 
import os
import numpy as np
import jpy
import re
import sys


def create_product_name(source):

    
    def find_satelite(s):

        if 'S1B' in s:
            return 'S1B'

        else:
            return 'S1A'

    
    def find_date_string(s):

        letters = sum(c.isalpha() for c in s)

        if letters/len(s) < 0.1:
            return s.split('T')[0]

        else:
            return None

    
    def asc_or_desc_orbit(source):

        orbit = source.getMetadataRoot().getElement('Abstracted_Metadata').getAttribute('PASS').getData()

        if str(orbit) == 'DESCENDING':
            return 'ID_SIG'

        else:
            return 'IA_SIG'

        
    def find_date(s):
        list_ = [find_date_string(i) for i in s.split('_')]
        return next(item for item in list_ if item is not None)

    s = source.getName()
    string = ''
    name = (string + find_date(s) + '_LEVEL2_' + 
            find_satelite(s) + asc_or_desc_orbit(source))
    

    return name



def do_terrain_correction(source):
    #print("\nGRD terrain correction...")
    parameters = snappy.HashMap()

    parameters.put('demName', 'SRTM 3Sec')
    parameters.put('demResamplingMethod', 'BILINEAR_INTERPOLATION')
    parameters.put('imgResamplingMethod', 'BILINEAR_INTERPOLATION')
    parameters.put('pixelSpacingInMeter', 10.0)
    parameters.put('mapProjection', 'AUTO:42001')
    parameters.put('nodataValueAtSea', False)
    parameters.put('saveSelectedSourceBand', True)
    
    output = snappy.GPF.createProduct('Terrain-Correction', parameters, source)
    return output


def do_radiometric_calibration(source):
    #print("\nGRD radiometric calibration...")
    parameters = snappy.HashMap()
    parameters.put('outputSigmaBand', True)
    parameters.put('sourceBands','Intensity_VH,Intensity_VV')
    parameters.put('selectedPolarisations','VH,VV')
    parameters.put('outputImageScaleInDb', False)
    
    output = snappy.GPF.createProduct('Calibration', parameters, source)
    return output


def do_remove_GRD_border_noise(source):
    #print('\nGRD border noise removal...')
    parameters = snappy.HashMap()
    parameters.put('Remove-GRD-Border-Noise', True)
    output = snappy.GPF.createProduct('Remove-GRD-Border-Noise', parameters, source)
    return output


def read_product(path):
    #print('\nGRD reading product...')
    output = snappy.ProductIO.readProduct(path)
    return output


def apply_orbit_file(source):
    #print('\napply orbit file...')
    parameters = snappy.HashMap()
    parameters.put('Apply-Orbit-File', True)
    output = snappy.GPF.createProduct('Apply-Orbit-File', parameters, source)
    return output


def do_thermal_noise_removal(source):
    #print('\nthermal noise removal...')
    parameters = snappy.HashMap()
    parameters.put('removeThermalNoise', True)
    output = snappy.GPF.createProduct('ThermalNoiseRemoval', parameters, source)
    return output


def convert_dB(source):
    #print('\nconverting to dB...')
    parameters = snappy.HashMap()
    parameters.put('sourceBands', 'Sigma0_VH,Sigma0_VV')
    output = snappy.GPF.createProduct('LinearToFromdB', parameters, source)
    return output


def create_subset(source):
    
    geom = geojson_to_wkt(read_geojson('Dublin_Area.geojson'))
    
    #print("\ncreate subset...")
    parameters = snappy.HashMap()
    parameters.put('copyMetadata', True)
    parameters.put('geoRegion', geom)
    subset = snappy.GPF.createProduct('Subset', parameters, source)
    return subset


def do_convert_data_type(source, source_bands='Intensity_VH,Intensity_VV'):

    parameters = snappy.HashMap()
    parameters.put('sourceBands',source_bands)
    parameters.put('targetDataType','int16')

    output = snappy.GPF.createProduct('Convert-Datatype', parameters, source)
    return output


def reproject(source):
    
    coords = 'PROJCS["TM65 / Irish Grid",GEOGCS["TM65",DATUM["TM65",SPHEROID["Airy Modified 1849",6377340.189,299.3249646,AUTHORITY["EPSG","7002"]],TOWGS84[482.5,-130.6,564.6,-1.042,-0.214,-0.631,8.15],AUTHORITY["EPSG","6299"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4299"]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",53.5],PARAMETER["central_meridian",-8],PARAMETER["scale_factor",1.000035],PARAMETER["false_easting",200000],PARAMETER["false_northing",250000],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","29902"]]'
    
    #print("\nreprojecting coordinate system...")
    parameters = snappy.HashMap()
    parameters.put('crs', coords)
    parameters.put('resampling', 'Nearest')
    reprojProduct = snappy.GPF.createProduct('Reproject', parameters, source)
    return reprojProduct


def change_no_data_value(source, no_data_value, bands=[]):
    
    for band in bands:
        source.getBand(band).setGeophysicalNoDataValue(no_data_value)
        source.getBand(band).setNoDataValue(no_data_value)
        source.getBand(band).setNoDataValueUsed(True)
        
    return source



def do_band_maths(source, scale=10000, bands=['Sigma0_VH_db','Sigma0_VV_db']):
    
    #print("\ndoing band maths...")
    
    snappy.GPF.getDefaultInstance().getOperatorSpiRegistry().loadOperatorSpis()
    
    BandDescriptor = jpy.get_type('org.esa.snap.core.gpf.common.BandMathsOp$BandDescriptor')
    
    vh_band = BandDescriptor()
    vh_band.name = 'Sigma0_VH_Scaled'
    vh_band.type = 'int16'
    vh_band.expression = '(' + str(bands[0]) + '*' + str(scale) + ')'
    
    vv_band = BandDescriptor()
    vv_band.name = 'Sigma0_VV_Scaled'
    vv_band.type = 'int16'
    vv_band.expression = '(' + str(bands[1]) + '*' + str(scale) + ')'
    
    targetBands = jpy.array('org.esa.snap.core.gpf.common.BandMathsOp$BandDescriptor', 2)
    targetBands[0] = vv_band
    targetBands[1] = vh_band
    
    parameters = snappy.HashMap()
    parameters.put('targetBands', targetBands)
    
    band_maths = snappy.GPF.createProduct('BandMaths', parameters, source)
    
    return band_maths



def do_save_product(source, output_dir ,file_type='GeoTIFF-BigTIFF'):
    
    #print("\nsaving file...")
    output_name = create_product_name(source)
    outputfile = output_dir + output_name
    snappy.ProductIO.writeProduct(source, outputfile, file_type)
    source.closeIO()
    
    return outputfile + '.tif'


def do_terrain_correction_offline(source, externel_dem):
    print("\nGRD terrain correction...")
    parameters = snappy.HashMap()
    
    parameters.put('demName', 'External DEM')
    parameters.put('externalDEMFile', externel_dem)
    parameters.put('externalDEMNoDataValue', -32768.0)
    parameters.put('externalDEMApplyEGM', False)
    parameters.put('demResamplingMethod', 'BILINEAR_INTERPOLATION')
    parameters.put('imgResamplingMethod', 'BILINEAR_INTERPOLATION')
    parameters.put('pixelSpacingInMeter', 10.0)
    parameters.put('mapProjection', 'AUTO:42001')
    parameters.put('nodataValueAtSea', False)
    parameters.put('saveSelectedSourceBand', True)
    
    output = snappy.GPF.createProduct('Terrain-Correction', parameters, source)
    return output



if __name__ == "__main__":

    file_path = sys.argv[1]
    output_path = sys.argv[2]
    externel_dem = sys.argv[3]

    s1 = read_product(file_path)
    s1 = create_subset(s1)

    s1 = apply_orbit_file(s1)
    s1 = do_thermal_noise_removal(s1)
    s1 = do_remove_GRD_border_noise(s1)
    s1 = do_radiometric_calibration(s1)

    if '.tif' in externel_dem:
        s1 = do_terrain_correction_offline(s1, externel_dem=externel_dem)

    else:
        s1 = do_terrain_correction(s1)

    s1 = convert_dB(s1)

    s1 = do_band_maths(s1, scale=1000, bands=['Sigma0_VH_db','Sigma0_VV_db'])
    output_name = do_save_product(s1, output_dir=output_path ,file_type='GeoTIFF')
    print(output_name)