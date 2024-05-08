import trackerlib
from csat2 import misc, ECMWF, GOES
from csat2.ECMWF.ECMWF import _calc_eis
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
from cartopy import crs as ccrs
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from tqdm.notebook import tqdm
import time
import xarray as xr
import numpy.ma as ma
from advection_funcs import subset_around_point
import matplotlib.patches as mpatches
import math
import cftime
from scipy.optimize import curve_fit
from netCDF4 import Dataset
import datetime
from matplotlib.colors import LogNorm



### specify the initial time of the initial parcel
year, doy, hour = 2020, 92, 15 # Year, day of year, hour of day, note 17:00 UTC corresponds to 9am local time in california, 15UTC is 7am local time
init_time = misc.time.ydh_to_datetime(year, doy, hour)
time_step = datetime.timedelta(hours=0.5)  # Time step of the integration, this is 30 mins by default

n_trajectories = 6  # number of trajectories to run, since we start consecutive trajectories from the same initial position, this is the number of initial hours to loop over

t_step = 96 # time step in hours between setting off successive trajectories

rho_w = 1000 # density of water kg/m^3

time_between_trajectroies = datetime.timedelta(hours=t_step) # time between each trajectory,

time_advected = 96 # number of hours each parcel is advected for, something on order of days is appropriate, but use a small value for testing

total_time = datetime.timedelta(hours=time_advected) + datetime.timedelta(hours=(n_trajectories -1)*t_step) # total time of the integration
hours, seconds = divmod(total_time.total_seconds(), 3600)
minutes = seconds // 60
#init_position = np.array([initial_lon, initial_lat])  # Initial position of the air parcel
#winddata = ECMWF.ERA5WindData(level="1000hPa",res="0.25grid",linear_interp="both") # wind data on 1 degree grid, linearly interpolated in space and time
n_steps = int(total_time.total_seconds() / time_step.total_seconds()) +1 # Number of steps in the integration, add one on to account for the initial position
channel =7 # GOES channel of interest, note that this is the same resolution as the phase data

times = [init_time + i * time_step for i in range(n_steps)] # List of times for the integration

start_of_year = datetime.datetime(year, 1, 1) 
GOES_doy = [(dt - start_of_year).days + 1 for dt in times] ##
GOES_doy = [f"{doy:03d}" for doy in GOES_doy] ## format so it is 3 digits long i.e. 001 for 1st Jan

GOES_times = np.array([dt.strftime("%H%M") for dt in times]) ## list of times in the format HHMM, note this will have shape length of single trajectory + t_step * successive trajectories/ time_step
