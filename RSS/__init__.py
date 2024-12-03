from csat2 import locator
import numpy as np
from netCDF4 import Dataset
from glob import glob
import csat2.misc
import csat2.misc.dlist
import csat2.misc.stats
from .amsre_daily_v7 import AMSREdaily
from .amsr2_daily import AMSR2daily
from .windsat_daily_v7 import WindSatDaily
from .ssmis_daily_v7 import SSMISdaily
DEBUG = False

#Satellite code for files - not currently used
sat = {'WINDSAT': 'wind', 'AMSR-2':'f34', 'AMSR-E':'f32'}
# Default version
default_version = {'WINDSAT': '7.0.1', 'AMSR-2':'8', 'AMSR-E':'7','F18':'8','F17':'7.0.1','F16':'7.0.1'}
# Readin class
CLASS = {'WINDSAT': WindSatDaily, 'AMSR-E': AMSREdaily, 'AMSR-2': AMSR2daily,'F18':SSMISdaily,'F17':SSMISdaily,'F16':SSMISdaily}

def readin(product, year, doy, *args, **kwargs):
    if product in ['AMSR-E', 'AMSR-2', 'WINDSAT','F18','F17','F16']:
        return readin_griddedRSS(product, year, doy, *args, **kwargs)
    else:
        raise(ValueError, '{} does not exist'.format(product))

maclwp = {'00':  0.006107,
          '10': -0.0001258,
          '01': -0.002365,
          '20': -1.393e-5,
          '11':  9.531e-5,
          '02':  0.000208,
          '30':  3.127e-7,
          '21': -7.838e-7,
          '12': -4.802e-6,
          '03': -8.658e-6}

    
def readin_griddedRSS(product, year, doy, sds, timescale='daily', ver=None, correctLWP=False):
    if not ver:
        ver = default_version[product]
    filename = locator.search('RSS', product,
                              year=year,
                              doy=doy,
                              version=ver,
                              timescale=timescale)
    #print(filename)

    # if not filename:
    #     raise FileNotFoundError(f"No files found for product: {product}, year: {year}, doy: {doy}, version: {ver}, timescale: {timescale}")
    if filename[0].endswith('.gz'):
        data = CLASS[product](filename[0], missing=np.nan)
    else:
        data = Dataset(filename[0])
        #print(f"Variables in the netCDF file: {list(data.variables.keys())}")
        #print(f"Variable '{name}' shape: {var.shape}")
    output = {}
    for name in sds:
        try:
            output[name] = data.variables[name][:, ::-1]
        except IndexError:
            var = data.variables[name]
            #print(f"Variable '{name}' shape: {var.shape}")  # Print the shape of the variable
        
            if var.ndim == 1:
                output[name] = var[::-1]  # Reverse the one-dimensional array
            elif var.ndim == 2:
                output[name] = var[:, ::-1] 

        if product == 'WINDSAT':
            output[name] = output[name][::-1]

        # if product == 'AMSR-2':
        #     output[name] = output[name][:]

        

    if ('cloud' in sds) and correctLWP:
        wvp = data.variables['vapor']
        u = data.variables['windMF']
        clwp_correct = (
            maclwp['00'] +
            maclwp['10']*wvp + maclwp['01']*u +
            maclwp['20']*(wvp**2) + maclwp['11']*u*wvp + maclwp['02']*(u**2) +
            maclwp['30']*(wvp**3) + maclwp['21']*u*(wvp**2) +
            maclwp['12']*wvp*(u**2) + maclwp['03']*(u**3))
        clwp_correct = np.clip(clwp_correct, -0.03, 0.03)
        output['cloud'] -= csat2.misc.stats.zero_nans(clwp_correct)
        output['correct'] = clwp_correct

    
    return output