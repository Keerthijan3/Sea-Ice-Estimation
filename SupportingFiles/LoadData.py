'''
In this script, functions required to load data of various types
are provided.
'''


import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from sklearn.feature_extraction.image import extract_patches



def LoadImage(path):
    '''
    Given Path to SAR image:
    Loads the HH and HV image
    Returns images
    '''
    HH = plt.imread(path+'HH_8x8.tif')[:, :, 0]
    HV = plt.imread(path+'HV_8x8.tif')[:, :, 0]
    return HH, HV


def LoadTruth(path):
    '''
    Load ASI data
    '''
    icdata = Dataset(path+'IC_for_im.nc')
    ic = icdata.variables['iceConcentration'][:]
    ic = ic.transpose()
    nans = np.isnan(ic)
    return ic, nans


def LoadTruthv2(path):
    '''
    Load NT2 data
    '''
    icdata = Dataset(path+'IC_for_im_v2.nc')
    ic = icdata.variables['iceConcentration'][:]
    ic = ic.transpose()
    nans = np.isnan(ic)
    return ic, nans


def LoadTruthClass(path):
    '''
    Converts all to integers between 0-10 - 11 classes
    '''
    icdata = Dataset(path+'IC_for_im.nc')
    ic = icdata.variables['iceConcentration'][:]
    ic = ic.transpose()
    nans = np.isnan(ic)
    ic[~np.isnan(ic)] = ic[~np.isnan(ic)]/10
    ic = ic.astype(int)
    return ic, nans
