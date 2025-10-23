#!/usr/bin/env python
# coding: utf-8

# In[35]:


import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from metpy.plots import USCOUNTIES
import glob
import numpy as np
import xarray as xr
import pyart
from scipy.interpolate import interp1d
import matplotlib
import cv2
from scipy.interpolate import RegularGridInterpolator
import os
import datetime
from scipy import ndimage
from sklearn.cluster import DBSCAN
from scipy.signal import find_peaks
from scipy.ndimage import label, find_objects
import pandas as pd
from scipy import ndimage


# In[36]:


def get_vad(vad):
    vad_heights = vad['height'].values
    speed = vad['speed_12swp'].values 
    direction = vad['direction_12swp'].values

    u = -speed * np.sin(np.deg2rad(direction))
    v = -speed * np.cos(np.deg2rad(direction))

    # Interpolation function for VAD mean wind
    # wind_interp = interp1d(vad_heights, speed, bounds_error=False, fill_value='extrapolate')
    u_interp = interp1d(vad_heights, u, bounds_error=False, fill_value='extrapolate')
    v_interp = interp1d(vad_heights, v, bounds_error=False, fill_value='extrapolate')
    heights = np.arange(np.min(vad_heights), np.max(vad_heights) + 10, 10) 

    # wind_profile = wind_interp(heights)
    u_profile = u_interp(heights)
    v_profile = v_interp(heights)

    return u_profile, v_profile, heights 

def momentum_flux(depths, wavelengths, k=0.4):
    mom_flux = {}

    for i, t in enumerate(depths.keys()):

        time_str = str(t)
        print(f"Time: {time_str}")
        roll_depth = depths[time_str]
        wv = wavelengths[time_str]

        vad_path = vad_files[i]  
        u_profile, v_profile, heights = get_vad(xr.open_dataset(vad_path))

        mom_flux[time_str] = {}

        for j, (cross_section, depth) in enumerate(roll_depth.items()):
            print(f"Cross Section: {cross_section}")

            if cross_section not in wv:
                print(f"  Skipping cross section '{cross_section}' – not in wavelength data.")
                continue

            mom_flux[time_str][cross_section] = {}
            max_rvels = []
            for l in range(len(wv[cross_section].keys())):
                # print(wv[mode][f'Roll{k}'])
                max_residual = wv[cross_section][f'Roll{l+1}'][3]
                max_rvels.append(max_residual)
                print(max_rvels)

            # print(max_residual)
            # print(max_rvels)

            if len(max_rvels) == 0:
                print('POYO! Max residual velocity list is empty!')
                continue

            for u, z_prime in depth.items():

                # Get roll bottom and top height
                z_prime = z_prime * 1000
                roll_top = 200 + z_prime
                roll_bottom = 200

                idx_bottom = np.argmin(np.abs(heights - roll_bottom))
                idx_top = np.argmin(np.abs(heights - roll_top))

                # Extract wind profile values
                # wind_bottom = wind_profile[idx_bottom]
                # wind_top = wind_profile[idx_top]

                u_bottom = u_profile[idx_bottom]
                u_top = u_profile[idx_top]
                v_bottom = v_profile[idx_bottom]
                v_top = v_profile[idx_top]

                du = u_top - u_bottom
                dv = v_top - v_bottom
                du_dz = np.sqrt(du**2 + dv**2) / z_prime 
                print(f'Magnitude of shear: {np.sqrt(du**2 + dv**2)}')

                # Compute vertical velocity perturbation (w')
                w_prime = k * z_prime * du_dz
                u_prime = max_rvels[int(u[4])-1]

                print(f'  {u}:')

                print(f"    u' = {u_prime}, w' = {w_prime}")
                mom_flux_val = u_prime*w_prime
                print(f"    mom_flux = {mom_flux_val} m2/s2")
                # print(w_prime)
                mom_flux[time_str][cross_section][f'{u}'] = mom_flux_val

    return mom_flux

ds = xr.open_dataset('cross_sections_new.nc')
data = np.load('roll_data_0.15p_new.npz', allow_pickle = True)

data = {key: data[key].item() for key in data.files}

rdi = np.load('roll_depths_0.15p_new.npz', allow_pickle=True)
rdidx = np.load('roll_distance_indices_0.15p_new.npz', allow_pickle=True)

depths = {key: rdi[key].item()  for key in rdi.files}
rdidx = {key: rdidx[key].item() for key in rdidx.files}

wavelengths = np.load('roll_wavelengths_0.15p.npz', allow_pickle=True)
wavelengths = {key: wavelengths[key].item() for key in wavelengths.files}


vad_folder = r'\\uahdata\rgroup\tcbl\FrancineProject\KHDC_VADs\Output\10km'
vad_files = sorted(glob.glob(os.path.join(vad_folder, 'KHDC_*_VAD.nc')))

mom_flux = momentum_flux(depths, wavelengths)

mom_flux


# In[37]:


np.savez('mom_flux_0.15p_new.npz', **mom_flux)


# In[ ]:




