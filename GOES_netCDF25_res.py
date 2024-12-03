# %%

## load in the relavant packages


from csat2 import misc, ECMWF, GOES
from csat2.ECMWF.ECMWF import _calc_eis
from csat2.misc import fileops
import numpy as np
from advection_functions.air_parcel import AirParcel25
from datetime import datetime, timedelta
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from advection_functions import advection_funcs,GOES_AMSR_colocation,LWP_correction_funcs,plotting
from scipy.stats import binned_statistic_2d
import xarray as xr
from netCDF4 import Dataset
import pvlib
import RSS
import cftime
import pandas as pd
import os
import logging
import netCDF4 as nc
import warnings

# %%


def dqf_filter(dqf_data):
    # Good quality data points have a flag value of 0
    return dqf_data == 0


# make a netCDF file that is all the GOES CONUS data for 2020 at 0.25 degree resolution of 0.25 degree grid

t_step = 30
year = 2022
start_date = datetime(year,1,1,0,0,0)

end_date = datetime(year, 12, 31, 23, 59)
channel = 7
logging.basicConfig(filename='process.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define constants
nan_fill_value = np.nan


# %%

gran = GOES.Granule.fromtext(f'G17.{start_date.year}{start_date.timetuple().tm_yday:03d}.{start_date.strftime("%H%M")}.RadC')

glon, glat = gran.get_lonlat(channel=channel) ## the lon and lat is the same for every time step so we can just use the first one

glon_360 = advection_funcs.adjust_longitudes(glon) # adjust the longitudes to be [0,360] instead of [-180,180]  

glon_flat = glon_360.flatten()
glat_flat = glat.flatten()



lon_min, lon_max = glon_360.min(), glon_360.max()
lat_min, lat_max = glat.min(),glat.max()

# Create the latitude and longitude arrays with a 0.25° increment
lons = np.arange(np.trunc(lon_min), np.ceil(lon_max) + 0.25, 0.25)
lats = np.arange(np.trunc(lat_min), np.ceil(lat_max) + 0.25, 0.25)

lon_bins = np.arange(np.trunc(lon_min) - 0.125, np.ceil(lon_max) + 0.25 + 0.125, 0.25)
lat_bins = np.arange(np.trunc(lat_min) - 0.125, np.ceil(lat_max) + 0.25 + 0.125, 0.25)
expected_shape = (len(lon_bins) - 1, len(lat_bins) - 1)

lon_grid, lat_grid = np.meshgrid(lons, lats)

nan_fill_value = np.nan
output_dir = f'/disk1/Data/GOES/Geoff/{year}/v2'


# %%

# Ensure output directory exists

warnings.filterwarnings('ignore', category=RuntimeWarning)
os.makedirs(output_dir, exist_ok=True)
# Set up logging
logging.basicConfig(
    filename='/disk1/Data/GOES/Geoff/process1.log', 
    level=logging.DEBUG, 
    format='%(asctime)s - %(levelname)s - %(message)s')



# Define a function to process data for a single day
def process_day(date):
    times = []
    CSM_25_list = []
    CTP_25_list = []
    CER_25_list = []
    COD_25_list = []
    CF_25_list = []

    # Adjusted expected shape (lon, lat)
    expected_shape = (len(lon_bins) - 1, len(lat_bins) - 1)

    # Loop over all timesteps for the given day
    current_time = date
    while current_time < date + timedelta(days=1):
        print(current_time)
        logging.info(f"Processing time step: {current_time}")
        
        gran = GOES.Granule.fromtext(f'G17.{current_time.year}{current_time.timetuple().tm_yday:03d}.{current_time.strftime("%H%M")}.RadC')

        # Initialize variables for the current time step
        CSM_25 = np.full(expected_shape, nan_fill_value)
        CTP_25 = np.full(expected_shape, nan_fill_value)
        CER_25 = np.full(expected_shape, nan_fill_value)
        COD_25 = np.full(expected_shape, nan_fill_value)

        try:
            BCM = gran.get_product_data(product='L2-ACM', dqf_filter=dqf_filter).flatten() ==1 #binary cloud mask = 1 when cloud is present
            LWC = gran.get_product_data(product='L2-ACTP', dqf_filter=dqf_filter).flatten() ==1 #liquid water cloud = 1 when liq cloud is present
            LWC_mask = BCM & LWC # True when cloud is present and is a liquid cloud
        except Exception as e:
            gran.download(product = 'L2-ACM')
            gran.download(product = 'L2-ACTP')
            print(f"Error processing CF data: {e}")
            LWC_mask = np.full(expected_shape, nan_fill_value)


        # Define a helper function for data retrieval and processing
        def process_data(product):
            try:
                data = gran.get_product_data(product=product, dqf_filter=dqf_filter).flatten()
                nan_mask = ~np.isnan(data) # this is true when the data is not nan
                combined_mask = nan_mask & LWC_mask # this is true when the data is not nan and the cloud is liquid
                data_25, _, _, _ = binned_statistic_2d(glon_flat[combined_mask], glat_flat[combined_mask], data[combined_mask], statistic='mean', bins=[lon_bins, lat_bins])
            except Exception as e:
                gran.download(product = product)
                print(f"Error processing {product} data: {e}")
                data_25 = np.full(expected_shape, nan_fill_value)
            
            return data_25
        
        # try:
        #     total = np.histogram2d(glon_flat,glat_flat,bins=[lon_bins, lat_bins]) #  GOES lon needs to be between 0 and 360 because of the way the hsitogram has been constructed
        #     cld = np.histogram2d(glon_flat[LWC_mask],glat_flat[LWC_mask],bins= [lon_bins, lat_bins])
        #     CF_25 = cld[0]/total[0]
        # except Exception as e:
        #     print(f"Error processing CF data: {e}")
        #     CF_25 = np.full(expected_shape, nan_fill_value)



        # Process each type of data
        CSM_25 = process_data('L2-ACM')
        CTP_25 = process_data('L2-ACTP')
        CER_25 = process_data('L2-CPS')
        COD_25 = process_data('L2-COD')
       

        
        try:
            nan_mask = ~np.isnan(LWC_mask) # this is true when the data is not nan
            CF_25, _, _, _ = binned_statistic_2d(glon_flat[nan_mask], glat_flat[nan_mask], LWC_mask.astype(int).flatten()[nan_mask], statistic='mean', bins=[lon_bins, lat_bins])
        except Exception as e:
             print(f"Error processing CF 25 resolution data: {e}")
             CF_25 = np.full(expected_shape, nan_fill_value)



        # Append data for current time
        times.append(current_time)
        CSM_25_list.append(CSM_25.copy())
        CTP_25_list.append(CTP_25.copy())
        CER_25_list.append(CER_25.copy())
        COD_25_list.append(COD_25.copy())
        CF_25_list.append(CF_25.copy())

        # Move to the next timestep
        current_time += timedelta(minutes=t_step)

    # Convert lists to arrays
    times = np.array([(t - datetime(1970, 1, 1)).total_seconds() for t in times])  # Convert datetime to seconds since epoch
    CSM_25_array = np.array(CSM_25_list)
    CTP_25_array = np.array(CTP_25_list)
    CER_25_array = np.array(CER_25_list)
    COD_25_array = np.array(COD_25_list)
    CF_25_array = np.array(CF_25_list)

    # Create NetCDF file for the day
    filename = os.path.join(output_dir, f'GOES_LWC_{year}_25_grid_{date.strftime("%Y%m%d")}v2.nc')

    if os.path.exists(filename):
        try:
            os.remove(filename)
            print(f"Deleted existing file: {filename}")
        except Exception as e:
            print(f"Failed to delete file: {e}")
            logging.error(f"Failed to delete file: {filename} due to {e}")
            return
        
    with nc.Dataset(filename, 'w', format='NETCDF4') as ds:
        # Define dimensions
        ds.createDimension('time', len(times))
        ds.createDimension('lon', len(lon_bins) - 1)
        ds.createDimension('lat', len(lat_bins) - 1)
        
        # Create variables
        time_var = ds.createVariable('time', 'f8', ('time',))
        lon_var = ds.createVariable('lon', 'f8', ('lon',))
        lat_var = ds.createVariable('lat', 'f8', ('lat',))
        CSM_var = ds.createVariable('CSM_25', 'f4', ('time', 'lon', 'lat'), fill_value=nan_fill_value)
        CTP_var = ds.createVariable('CTP_25', 'f4', ('time', 'lon', 'lat'), fill_value=nan_fill_value)
        CER_var = ds.createVariable('CER_25', 'f4', ('time', 'lon', 'lat'), fill_value=nan_fill_value)
        COD_var = ds.createVariable('COD_25', 'f4', ('time', 'lon', 'lat'), fill_value=nan_fill_value)
        CF_var = ds.createVariable('CF_25', 'f4', ('time', 'lon', 'lat'), fill_value=nan_fill_value)
        
        # Assign data to variables
        time_var[:] = times
        lon_var[:] = lons
        lat_var[:] = lats
        CSM_var[:, :, :] = CSM_25_array
        CTP_var[:, :, :] = CTP_25_array
        CER_var[:, :, :] = CER_25_array
        COD_var[:, :, :] = COD_25_array
        CF_var[:, :, :] = CF_25_array

        # Add attributes
        ds.setncattr('description', 'Processed GOES data')
        time_var.setncattr('units', 'seconds since 1970-01-01 00:00:00')
        time_var.setncattr('calendar', 'gregorian')
        lon_var.setncattr('units', 'degrees')
        lat_var.setncattr('units', 'degrees')
        CSM_var.setncattr('units', 'unitless')
        CTP_var.setncattr('units', 'unitless')
        CER_var.setncattr('units', 'unitless')
        COD_var.setncattr('units', 'unitless')
        CF_var.setncattr('units', 'unitless')

# Main loop to process all days in 2020
current_date = start_date


# %%





while current_date <= end_date:
    process_day(current_date)
    print(f"Finished processing {current_date}")
    current_date += timedelta(days=1)

# %%




