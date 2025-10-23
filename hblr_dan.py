"""
@author: ejohnson

"""

# =================================================================================================
# hblr_dan.py (Hurricane Boundary Layer Roll Detection, Analysis, and I wanted to name it Dan)
# Shout out to Dan Cecil!
#
# A comprehensive script to identify, pair, and analyze hurricane boundary layer rolls from
# gridded residual velocity data.
#
# This script performs the following steps:
#    1. Determines optimal roll orientation and generates perpendicular cross-sections.
#    2. Identifies positive and negative velocity blobs using a multi-peak,
#       area-based filtering method.
#    3. Pairs blobs based on proximity to form roll couplets.
#    4. Calculates the wavelength of each roll pair.
#    5. Calculates the depth of each roll pair according to valid wavelengths.
#    6. Calculates momentum flux based on roll characteristics.
#    7. Saves all intermediate and final results to .npz files.
#    8. Generates and saves summary plots for roll depth and wavelength distributions.
#
# Last Updated: October 23, 2025 by Evan Johnson (ejohns-tcbl)
# =================================================================================================

# --- Imports ---
import argparse
import matplotlib
import matplotlib.pyplot as plt
import glob
import numpy as np
import xarray as xr
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.signal import find_peaks
from scipy.ndimage import label, find_objects, center_of_mass
from scipy.fft import fft, fftfreq
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import distance_transform_edt
import os
import datetime
import pandas as pd
import sys
import warnings
from collections import defaultdict
import colormaps as cmaps # pip install colormaps (https://pratiman-91.github.io/colormaps/)
from skimage.measure import regionprops
import logging
from itertools import combinations

# --- Helper Class to Suppress C-level stdout (debugging purposes)---
class suppress_stdout_stderr(object):
    """
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, redirecting stderr and stdout to /dev/null. This is useful for
    suppressing C-level output.
    
    """
    def __init__(self):
        # Open a pair of null file descriptors
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout and stderr file descriptors
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to the descriptors
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close all file descriptors
        for fd in self.null_fds:
            os.close(fd)
        for fd in self.save_fds:
            os.close(fd)

# Font size for plots
matplotlib.rcParams.update({'font.size': 18})

# --- Core analysis and plotting functions (finding the orientation of cross sections) ---

def get_hist(gas, bins=1000):
    """
    Helper function to create a histogram of residual velocity gradient vector directions

    inputs:
        gas: Direction of gradients in roll data
        bins: number of desired bins in the histogram
    outputs:
        hist: the histogram representing all the gradient directions
        bin_edges: the bin edges of the histogram
    """
    hist, bin_edges = np.histogram(gas.flatten(), bins=bins, range=(-np.pi, np.pi))
    return hist, bin_edges


def get_da(desired_angle_radians, roll_data, debug = False, plotting=True):
    """
    inputs:
        desired_angle_radians: Desired orientation of rolls in radians
        roll_data: 2d field of residual velocity representing where the rolls are
    outputs:
        da: Dominant angle according the histogram of gradient directions.
            If there is more than one dominant angle, da is the closest to the desired_angle_radians.
    """
    # Calculating the gradient of the field
    gy, gx = np.gradient(roll_data)


    # Plotting the gradient field
    if plotting:
        plt.quiver(roll_data.x0, roll_data.y0, gx, gy)
        # plt.show()
        plt.close()

    # Calculating the direction of gradients
    gas = np.arctan2(gy, gx)

    # Create a histogram of gradient orientations to find the most common one in the region where are rolls are
    hist, bin_edges = get_hist(gas, bins=len(gas))
    where_maxes = np.where(hist == np.max(hist))[0]

    # Plotting gradient vector field
    if plotting:
        fig, ax = plt.subplots()

        x = np.linspace(-np.pi, np.pi, len(gas))
        ax.plot(x, hist)

        # Set ticks and labels at key multiples of pi
        ticks = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
        tick_labels = [r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$']

        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels)

        plt.xlabel('Gradient Directions (radians)')
        plt.ylabel('Number of Occurrences')
        # plt.show()
        plt.close()

    if len(where_maxes) == 1:
        da = bin_edges[np.argmax(hist)]
        if debug:
            logging.info(f"Number of maximums in histogram is only 1, so we are choosing it ({da} radians)")
    else:
        if debug:
            logging.info("Number of maximums in gradients histogram is greater than 1, continuing")
        hist_values = []
        for i, idx in enumerate(where_maxes):
            if debug:
                logging.info(f"Angle of gradient in histogram number {i} is {bin_edges[idx]} radians")
            hist_values.append(bin_edges[idx])

        # Finding the closest gradient vector orientation to the desired angle (0 radians for automated purposes)
        hist_values_ar = np.array(hist_values)
        closest_max_idx = np.argmin(np.abs(hist_values_ar - desired_angle_radians))
        da = bin_edges[where_maxes[closest_max_idx]]
        if debug:
            logging.info(f"Closest dominant angle to desired angle in radians: {da}")

    return da


def find_distance_to_boundary(start_point, angle_rad, box_bounds):
    """
    Casts a ray from a start_point at a given angle and finds the
    distance to the first box boundary it intersects.

    inputs:
        start_point: The point along the line which is angled at da from get_da
        angle_rad: dominant angle (da) from get_da
        box_bounds: the bounds of the domain box
    outputs:
        The minimum distance on both sides of the point along the line
    """
    # Point along the grid/line
    x0, y0 = start_point

    # Getting the lengths of the adjacent (cos) and opposite side (sin) of the ray unit vector
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    # A ray unit vector is represented by x(t) or y(t)

    # The direction vector is (cos(angle_rad), sin(angle_rad))

    # x(t) = x0 + t_right/t_left*cos(angle_rad) -> t_right/t_left = (x(t) - x0) / cos(angle_rad)
    # y(t) = y0 + t_top/t_bottom*sin(angle_rad) -> t_top/t_bottom = (y(t) - y0) / sin(angle_rad)

    # At the end of the function you see that it is returning the minimum of the ray distances, indicating the first ones
    # that intersect the bounding box

    # Creating a list for distances
    distances = []

    if not np.isclose(cos_a, 0):
        t_right = (box_bounds['x_max'] - x0) / cos_a
        t_left = (box_bounds['x_min'] - x0) / cos_a
        if t_right >= 0: distances.append(t_right)
        if t_left >= 0: distances.append(t_left)

    if not np.isclose(sin_a, 0):
        t_top = (box_bounds['y_max'] - y0) / sin_a
        t_bottom = (box_bounds['y_min'] - y0) / sin_a
        if t_top >= 0: distances.append(t_top)
        if t_bottom >= 0: distances.append(t_bottom)

    return min(distances) if distances else 0


def get_cs_distance(roll_data, da, xe, x_coords, y_coords, resolution=100):

    """
    Creates the line at the angle from get_da, finds the distance from each point along
    the line to the edge of the bounding box using find_distance_to_boundary

    inputs:
        roll_data: 2d field of residual velocity data at a certain height (default z = 0.35 km)
        da: dominant angle (da) from get_da
        xe: the absolute value x-distance from the edge of the domain (basically, N (domain size in km) = xe)
        x_coords: x coordinates of the roll_data
        y_coords: y coordinates of the roll_data
        resolution: the number of cross sections you want across the domain. Default is 100.
    outputs:
        cs_distances_A, cs_distances_B: list of distances from the cross section line (oriented perpendicular to the rolls)
                                        to the edge of the domain on both sides of the line
        xf, yf: list of closest grid indices (corresponding to the x and y coordinates) to the cross section line
    """

    # Middle point of the cross section
    plcx, plcy = 0, 0

    # Dominant orientation of the vector (representing the orientation of the rolls)
    dag = np.degrees(da)

    # Fancy trigonemetry that we worked through to get the angle of the perpendicular line to dag from the x-axis
    pangle = 180 - dag

    fangle = 180 - 90 - pangle
    # The distances of the perpendicular line along the y-axis
    y = xe * np.tan(np.radians(fangle))

    # The perpendicular line
    x_line = np.linspace(plcx - xe, plcx + xe, resolution)
    y_line = np.linspace(plcy - y, plcy + y, resolution)

    # Closest indices to the grid for each perpendicular line point
    ix = np.array([np.argmin(np.abs(x_coords - x)) for x in x_line])
    iy = np.array([np.argmin(np.abs(y_coords - y)) for y in y_line])
    xf = x_coords[ix]
    yf = y_coords[iy]

    # Defining the box (region of interest) bounds
    box_bounds = {
        "x_min": np.min(x_coords), "x_max": np.max(x_coords),
        "y_min": np.min(y_coords), "y_max": np.max(y_coords)
    }

    # Getting the grid points along the line into an array
    line_points = [np.array([x, y]) for x, y in zip(xf, yf)]

    # The angles perpendicular to the perpendicular line (orientation of the cross section, should be the same as dag and -dag)
    pa = np.radians(fangle) + np.pi / 2
    pb = np.radians(fangle) - np.pi / 2

    # Finding the distance from each grid/line point to the edge of the box, refer to comments in "find_distance_to_boundary" function
    all_distances = []
    for point in line_points:
        dist_A = find_distance_to_boundary(point, pa, box_bounds)
        dist_B = find_distance_to_boundary(point, pb, box_bounds)
        all_distances.append({
            "point_coords": point,
            "distance_A": dist_A,
            "distance_B": dist_B
        })

    # Adding the distances (on both sides of the cross section [orienation of the rolls] line)
    cs_distances_A = []
    cs_distances_B = []
    for val in all_distances:
        cs_distances_A.append(val['distance_A'])
        cs_distances_B.append(val['distance_B'])

    return cs_distances_A, cs_distances_B, xf, yf


def get_cross_sections(subset, roll_data, cs_distances_A, cs_distances_B, xf, yf, da, num_points=100, save_images = True, N = 5, folder_name = 'blank'):

    """
    Creates the cross section at each z level in the 3D grid along the cross section line

    inputs:
        subset: larger (than roll_data) 2d field of residual velocity data at a certain height (default z = 0.35 km)
        roll_data: 2d field of residual velocity data at a certain height (default z = 0.35 km)
        cs_distances_A, cs_distances_B: list of distances from the cross section line (oriented perpendicular to the rolls)
                                        to the edge of the domain on both sides of the line
        xf, yf: list of closest grid indices (corresponding to the x and y coordinates) to the cross section line
        num_points: number of points for each side of the line from the center point (default = 100 points,
                    so 200 total for the whole line)
        save_images: boolean to indicate whether to save images of cross sections and CAPPI at z (default = 0.35 km) with
                     cross section line
        N: desired domain_size in km
        folder_name: the path for saving images
    outputs:
        cross_sections: 2D list of cross section data for each z levels
        distances_ar: 1D array of the horizontal distance (in km) along the cross section line
    """

    z_levels = subset['z0'].values
    x_coords = subset['x0'].values
    y_coords = subset['y0'].values

    cross_sections = []
    distances_ar = []
    count = 0

    for csla, cslb, cxb, cyb in zip(cs_distances_A, cs_distances_B, xf, yf):
        # Finding the dx and dy for the cross section based on the dominant gradient orientation
        dx = csla * np.cos(da)
        dy = csla * np.sin(da)

        dxb = cslb * np.cos(da)
        dyb = cslb * np.sin(da)

        # Defining the x and y for cross section interpolation
        x_cross = np.linspace(cxb - dxb, cxb + dx, num_points)
        y_cross = np.linspace(cyb - dyb, cyb + dy, num_points)
        distances_km = np.linspace(0, csla + cslb, num_points)

        distances_ar.append(distances_km)

        # Creating the interpolation function for the cross section
        interp_func = RegularGridInterpolator(
            (z_levels, y_coords, x_coords),
            subset.values,
            bounds_error=False,
            fill_value=np.nan
        )

        # Performing the cross section interpolation for each z level
        cross_section = np.array([
            interp_func((z, y_cross, x_cross)) for z in z_levels
        ])

        cross_sections.append(cross_section)


        if save_images:
            # Plotting the cross section line on the subset_2d map
            fig, ax = plt.subplots()

            # Subset_2d field
            cf = ax.contourf(roll_data.x0, roll_data.y0, roll_data, levels=np.arange(-4, 4.25, 0.25), cmap=cmaps.curl)
            fig.colorbar(cf)

            # Cross section line
            ax.plot(
                [cxb - dxb, cxb + dx],
                [cyb - dyb, cyb + dy],
                color='red',
                linestyle='-',
                linewidth=2.5,
                marker='o',
                markersize=5,
            )

            ax.scatter(cxb + dx, cyb + dy, s=50, c='k', zorder=1000)
            ax.scatter(cxb - dxb, cyb - dyb, s=50, c='orange', zorder=1000)

            ax.set_ylim([np.min(roll_data.y0), np.max(roll_data.y0)])
            ax.set_xlim([np.min(roll_data.x0), np.max(roll_data.x0)])
            
            with suppress_stdout_stderr():
                plt.savefig(f'./Roll_Lines_{N}km/{folder_name}/{folder_name}_{str(count).zfill(2)}.png', bbox_inches='tight')
            plt.close()

            # Plotting the cross section itself
            fig, ax = plt.subplots(figsize=(8, 4), dpi=250)
            cf = ax.contourf(distances_km, z_levels, cross_section, cmap=cmaps.curl, levels=np.arange(-4, 4.25, 0.25))
            ax.scatter(0, 0.2, s=75, c='orange', zorder=1000)
            fig.colorbar(cf, label='Residual Velocity (m/s)')

            # ax.contour(distances_km, z_levels, cross_section, colors = 'k', levels=[0])
            # ax.axhline(0.5, color='k', linestyle='--')  # Example reference line at 0.5 km
            ax.set_xlabel('Distance Along Roll Direction (km)')
            ax.set_ylabel('Height (km)')
            ax.set_ylim([0.2, 1])

            plt.title(f'KNQA at {ds.time.values[0].astype(str)[0:22]}')
            plt.tight_layout()
            with suppress_stdout_stderr():
                plt.savefig(f'./Cross_Sections_{N}km/{folder_name}/{folder_name}_{str(count).zfill(2)}.png', bbox_inches='tight')
            plt.close()

        count += 1

    return cross_sections, distances_ar

# --- Blob Identification and Pairing Functions ---

def get_velocity_data(dxr, mode='positive'):
    """
    Helper function that extracts absolute velocity data for a given mode (typically 'positive' or 'negative').

    inputs:
        dxr: xarray datarray of cross section data
        mode: the sign of residual velocity data you want to return the absolute value of
    outputs:
        2D numpy array of absolute value of residual velocity for a certain sign (|+Vr'| or |-Vr'|)

    """

    masked_data = dxr.where(dxr > 0) if mode == 'positive' else dxr.where(dxr < 0)
    return np.abs(masked_data.to_numpy())

def find_prevalent_blobs(data_section, data_section_extra, max_height, peak_threshold, area_quantile, pos_centroids, mode = 'positive', abs_vel_threshold=1.45, plot_binary = False):
    """
    Finds significant blobs and returns their centroids and a labeled array of all blobs.

    inputs:
        data_section: 2D numpy array of absolute value of residual velocity (Vr')
        data_section_extra: 2D numpy array of absolute value of residual velocity (of the opposite sign)
        max_height: the index of the max analysis height desired (default is 6 corresponding to z = 1.5 km)
        peak_threshold: lowest percent (default = 0.25) of blob Vr' max at which the Vr' for a blob is valid
        area_quantile: the lower percent (across all detected blobs, default = 0.75) at which a blob area is valid
        pos_centroids: the centroids of the detected positive blobs (None for first run, see pseudocode below)
        mode: the sign of the residual velocity to find blobs for (positive at first, see pseudocode below)
        abs_vel_threshold: the max residual velocity for a blob must be greater than this (to filter out ones that
                           were erroneous before this step)
    outputs:
        Dictionary that has the blob centroids
        labeled_array: the final 2D data corresponding to an individual blob

    Pseudocode/Steps:

        For both positive and negative residual velocity:

            1. Grab values up to the max analysis height of the data
            2. Use the scipy's find peaks along the cross section distance axis to get blob maxes
            3. Create a binary mask for the data corresponding to peak_threshold*peaks of blob

        Next:

            - If mode is 'positive' (done first):
                  Get the positive centroids using the imported center_of_mass function of the blobs. Use the
                  positive blobs/centroids as anchor for each roll
            - If mode is 'negative' (done second):
                  Using the positive centroids from the first run of the function as the anchor points for rolls,
                  find the (1) closest and (2) most coherent negative residual blobs
            - Return centroids and data depending on sign/mode
    """
    # Max height in terms of index
    restricted_section = data_section[:max_height + 1, :]
    final_mask = np.zeros_like(restricted_section, dtype=bool)

    # Going through the distance along cross section
    for col in range(restricted_section.shape[1]):

        column_data = restricted_section[:, col]

        # If data is all nans, skip
        if np.all(np.isnan(column_data)):
            continue

        # Data that is not nan
        heights = np.where(~np.isnan(column_data), column_data, -np.inf)
        

        # Finding the blob maxes using find_peaks from scipy
        peaks, _ = find_peaks(heights)
        for peak_idx in peaks:
            peak_val = column_data[peak_idx]
            

            if np.isnan(peak_val):
                continue

            # If blob max is less than the abs_vel_threshold, skip
            if peak_val < abs_vel_threshold:
                continue

            # Final blob mask is equal to peak_threshold*max Vr' in blob
            threshold = peak_threshold * peak_val
            final_mask[:, col] |= (column_data >= threshold)

    # Creating a binary mask and labeling features using scipy.ndimage
    binary_mask = ~np.isnan(np.where(final_mask, restricted_section, np.nan))

    # if plot_binary:
    #     plt.contourf(binary_mask)
    #     plt.show()
        
    labeled_array, num_features = label(binary_mask, structure=np.ones((3, 3)))



    if num_features == 0:
        return {}, labeled_array

    if mode == 'positive':

        # If mode is 'positive' (done first):
        #         Get the positive centroids using the imported center_of_mass function of the blobs. Use the
        #         positive blobs/centroids as anchor for each roll

        blob_areas = np.bincount(labeled_array.ravel())

        if len(blob_areas) <= 1:
            return {}, labeled_array

        blob_areas = blob_areas[1:]

        MIN_BLOB_AREA_PIXELS = 200

        area_threshold = np.quantile(blob_areas, area_quantile)

        # Get slices (bounding boxes) for each feature
        slices = find_objects(labeled_array)
        
        # Compute width for each feature (x-axis range)
        widths = []
        
        for i, sl in enumerate(slices):
            if sl is None:
                widths.append(0)
                continue
            y_slice, x_slice = sl
            width = (x_slice.stop - x_slice.start)*0.10143568 # horizontal extent
            widths.append(width)

        widths = np.array(widths)
        # logging.info("Widths of features:", widths)

        labels_to_keep = [i + 1 for i, area in enumerate(blob_areas) if area > area_threshold and area >= MIN_BLOB_AREA_PIXELS and widths[i] > (1.9716928008*0.10143568)]

        if not labels_to_keep:
            return {}, labeled_array

        centroids = center_of_mass(binary_mask, labeled_array, labels_to_keep)

    elif mode == 'negative':

        # If mode is 'negative' (done second):
        #                 Using the positive centroids from the first run of the function as the anchor points for rolls,
        #                 find the (1) closest and (2) most coherent negative residual blobs

        if pos_centroids:
            blob_areas = np.bincount(labeled_array.ravel())
            if len(blob_areas) <= 1: return {}, labeled_array
            area_threshold = np.quantile(blob_areas, 0)
            labels_to_keep = [i + 1 for i, area in enumerate(blob_areas) if area > area_threshold]
            if not labels_to_keep: return {}, labeled_array

            neg_centroids = center_of_mass(binary_mask, labeled_array, labels_to_keep)

            # Getting the positive centroids from dictionary into list
            pos_centroids_calc = []
            for pos_label, pos_centroid in pos_centroids.items():
                pos_centroids_calc.append(pos_centroid)


            # Filling the positive or negative centroids array with NaN in case it is a smaller shape than the other
            if np.shape(np.array(neg_centroids))[0] > np.shape(np.array(pos_centroids_calc))[0]:
                target_shape = np.array(neg_centroids).shape
                padded_array = np.full(target_shape, np.nan)
                padded_array[:np.array(pos_centroids_calc).shape[0], :np.array(pos_centroids_calc).shape[1]] = np.array(pos_centroids_calc)
                pos_centroids_calc = padded_array
            else:
                target_shape = np.array(pos_centroids_calc).shape
                padded_array = np.full(target_shape, np.nan)
                padded_array[:np.array(neg_centroids).shape[0], :np.array(neg_centroids).shape[1]] = np.array(neg_centroids)
                neg_centroids = padded_array

            # Keep only valid rows (no NaNs)
            valid_pos_mask = ~np.isnan(pos_centroids_calc).any(axis=1)
            neg_centroids = np.array(neg_centroids)
            valid_neg_mask = ~np.isnan(neg_centroids).any(axis=1)

            pos_centroids_calc = np.array(pos_centroids_calc)
            filtered_pos = pos_centroids_calc[valid_pos_mask]
            filtered_neg = neg_centroids[valid_neg_mask]

            # Broadcast subtraction, get index of closest neg_centroid for each pos_centroids_calc
            diff = filtered_pos[:, None, :] - filtered_neg[None, :, :]  # shape (N, M, 2)
            dists = np.linalg.norm(diff, axis=2)  # shape (N, M)

            # Get index of closest neg_centroid for each pos_centroid
            idx = np.argmin(dists, axis=1)

            # Retrieve the closest values
            closest_values = filtered_neg[idx]

            # Insert the results back into full-size array if needed
            full_result = np.full_like(pos_centroids_calc, np.nan)
            full_result[valid_pos_mask] = closest_values

            # Use the indices to get the actual closest values
            centroids = full_result
        else:
            return {}, labeled_array # Return empty data structures if no positive centroids

    if plot_binary:
        plt.contourf(labeled_array)
        plt.show()

    return {lbl: cent for lbl, cent in zip(labels_to_keep, centroids)}, labeled_array

def process_and_pair_blobs(positive_data, negative_data, max_height, peak_threshold, area_quantile, debug=False, plot_binary = False):
    """
    Processes data to find and pair prevalent blobs.

    inputs:
        positive_data: absolute value of positive Vr' obtained from get_velocity_data()
        negative_data: absolute value of negative Vr' obtained from get_velocity_data() 
        max_height: the maximum height index that analysis will be performed at, corresponding with heights in grid
                    (same as analysis_max_height)
        peak_threshold: relative threshold (of max res. vel. in blob) for peak detection in blobs (e.g., 0.15)
        area_quantile: quantile for blob area filtering (e.g., 0.75 for top 25%)
        debug: if on, will save INFO statements to debug.log and print in console
        plot_binary: if on, will plot binary field of rolls
    outputs:
        results: dictionary that has roll pair data, the positive labeled array (scipy.ndimage), and negative labeled array (scipy.ndimage).
                 structure below:
                 
                 {'pairs': pd.DataFrame(pairs), 'positive_labeled_array': pos_labeled_array, 'negative_labeled_array': neg_labeled_array}

        positive_centroids_dict, negative_centroids_dict: Dictionaries that has the blob centroids from find_prevalent_blobs

    Pseudocode/Steps:
    
        1. Finds prevalent and potential blobs for pairing'
        2. Finds solidity (from regionprops) of each blobs
        3. Assign score for each potential blob pair, using positive blob as anchor:
            Goes through each positive centroid/blob, assigning score to each negative. Highest score pair wins.
                score = distance/solidity

                    distance: numpy.linalg.norm between positive and negative centroid
                    solidity: for prop in skimage.measure.regionprops, prop.solidity

        4. Return a dictionary of roll pairs with highest score
    """

    # Finding prevalent and potential blobs, see find_prevalent_blobs() documentation
    pos_centroids_dict, pos_labeled_array = find_prevalent_blobs(
        positive_data, negative_data, max_height, peak_threshold,
        pos_centroids=None, mode='positive', area_quantile=area_quantile, plot_binary = plot_binary
    )
    neg_centroids_dict, neg_labeled_array = find_prevalent_blobs(
        negative_data, positive_data, max_height, peak_threshold,
        pos_centroids=pos_centroids_dict, mode='negative', area_quantile=area_quantile, plot_binary = plot_binary
    )

    # Account for no blobs
    if not pos_centroids_dict or not neg_centroids_dict:
        if debug: logging.info("INFO: Not enough positive or negative blobs to begin pairing.")
        return None, {}, {}

    if debug:
        logging.info(f"\nINFO: Found {len(pos_centroids_dict)} positive and {len(neg_centroids_dict)} negative candidate blobs.")
        logging.info("-" * 40)

    # Find the solidity of each blob
    neg_properties = regionprops(neg_labeled_array)
    neg_solidity_map = {prop.label: prop.solidity for prop in neg_properties}

    available_neg_labels = list(neg_centroids_dict.keys())
    pairs = []
    sorted_pos_labels = sorted(pos_centroids_dict.keys(), key=lambda k: pos_centroids_dict[k][1])

    # Loop through positive labels/centroids/blobs, scoring each negative blob relative to the positive anchor
    for pos_label in sorted_pos_labels:
        pos_centroid = pos_centroids_dict[pos_label]
        if not available_neg_labels:
            if debug: logging.info(f"INFO: No more available negative blobs to pair with Anchor Pos #{pos_label}.")
            break

        if debug:
            logging.info(f"--- Evaluating Anchor Pos Blob #{pos_label} ---")
            logging.info(f"  Available candidates: {available_neg_labels}")

        best_neg_label = None
        max_score = -1

        for neg_label in available_neg_labels:
            neg_centroid = neg_centroids_dict[neg_label]
            distance = np.linalg.norm(np.array(pos_centroid) - np.array(neg_centroid))

            

            if distance < 1e-6: distance = 1e-6

            solidity = neg_solidity_map.get(neg_label, 0)
            score = solidity / distance

            if debug: logging.info(f"       - Candidate Neg #{neg_label}: dist={distance:.2f}, solidity={solidity:.2f}, SCORE={score:.2f}")

            if distance*0.10143568 > 2.5:
                continue

            if score > max_score:
                max_score = score
                best_neg_label = neg_label

        if best_neg_label is not None:
            if debug: logging.info(f"  -> WINNER for Pos #{pos_label} is Neg #{best_neg_label} with score {max_score:.2f}\n")
            pairs.append({'pos_label': pos_label, 'neg_label': best_neg_label})
            available_neg_labels.remove(best_neg_label)

    if debug:
        logging.info("-" * 40)
        logging.info("INFO: Final Pairs Found:")
        if pairs:
            for p in pairs:
                logging.info(f"  Pos #{p['pos_label']} <--> Neg #{p['neg_label']}")
        else:
            logging.info("  None")
        logging.info("-" * 40)

    if not pairs:
        return None, pos_centroids_dict, neg_centroids_dict

    results = {
        'pairs': pd.DataFrame(pairs),
        'positive_labeled_array': pos_labeled_array,
        'negative_labeled_array': neg_labeled_array
    }

    return results, pos_centroids_dict, neg_centroids_dict

def calculate_blob_solidity(labeled_array):
    """
    Calculates the solidity for every blob in a labeled array using regionprops.
    Solidity is the ratio of the blob's area to its convex hull area.

    inputs:
        labeled_array (np.array): The array containing all labeled blobs.

    outputs:
        dict: A dictionary mapping each blob label (int) to its solidity (float).
    """
    # regionprops efficiently calculates many properties for all blobs at once
    properties = regionprops(labeled_array)

    # Create a dictionary of {label: solidity} for easy lookup
    # prop.solidity is a built-in property: prop.area / prop.convex_area
    solidity_map = {prop.label: prop.solidity for prop in properties}
    extent_map = {prop.label: prop.extent for prop in properties}
    return solidity_map, extent_map


def find_and_pair_blobs(dxr, max_height, peak_threshold, area_quantile, solidity_threshold=0.8, extent_threshold=0.6, debug=False, plot_binary = False):
    """
    Orchestrator to run the full blob pairing pipeline and format for the main workflow.
    
    """
    positive_data = get_velocity_data(dxr, mode='positive')
    negative_data = get_velocity_data(dxr, mode='negative')

    results, pos_centroids_dict, neg_centroids_dict = process_and_pair_blobs(
        positive_data, negative_data, max_height, peak_threshold, area_quantile=area_quantile, debug=debug, plot_binary = plot_binary
    )

    if not results or results['pairs'].empty: return []

    paired_pos_labels = results['pairs']['pos_label'].unique()
    paired_neg_labels = results['pairs']['neg_label'].unique()
    pos_labeled_array, neg_labeled_array = results['positive_labeled_array'], results['negative_labeled_array']
    pos_slices, neg_slices = find_objects(pos_labeled_array), find_objects(neg_labeled_array)
    dz = dxr.coords['height'].diff('height').mean().item()

    pos_blob_details = {
        lbl: {
            'slice': pos_slices[lbl - 1],
            'depth': (pos_slices[lbl - 1][0].stop - pos_slices[lbl - 1][0].start) * dz,
            'sh': pos_slices[lbl - 1][0].start * dz,
            'eh': pos_slices[lbl - 1][0].stop * dz,
            'centroid': pos_centroids_dict.get(lbl)
        } for lbl in paired_pos_labels if lbl - 1 < len(pos_slices)
    }

    neg_blob_details = {
        lbl: {
            'slice': neg_slices[lbl - 1],
            'depth': (neg_slices[lbl - 1][0].stop - neg_slices[lbl - 1][0].start) * dz,
            'sh': neg_slices[lbl - 1][0].start * dz,
            'eh': neg_slices[lbl - 1][0].stop * dz,
            'centroid': neg_centroids_dict.get(lbl)
        } for lbl in paired_neg_labels if lbl - 1 < len(neg_slices)
    }

    pos_solidity_map, pos_extent_map = calculate_blob_solidity(pos_labeled_array)
    neg_solidity_map, neg_extent_map = calculate_blob_solidity(neg_labeled_array)

    valid_pos_details = {
        lbl: details for lbl, details in pos_blob_details.items()
        if (pos_solidity_map.get(lbl, 0) >= solidity_threshold) and (pos_extent_map.get(lbl, 0) >= extent_threshold)
    }
    valid_neg_details = {
        lbl: details for lbl, details in neg_blob_details.items()
        if (neg_solidity_map.get(lbl, 0) >= solidity_threshold) and (neg_extent_map.get(lbl, 0) >= extent_threshold)
    }


    pos_blob_details = valid_pos_details
    neg_blob_details = valid_neg_details

    matched_pairs = []
    for _, row in results['pairs'].iterrows():
        pos_label, neg_label = row['pos_label'], row['neg_label']
        if pos_label in pos_blob_details and neg_label in neg_blob_details:
            if pos_blob_details[pos_label]['centroid'] is not None and neg_blob_details[neg_label]['centroid'] is not None:
                matched_pairs.append((pos_blob_details[pos_label], neg_blob_details[neg_label]))    
                pos_label, neg_label = row['pos_label'], row['neg_label']


    return matched_pairs

def is_empty(arr):
    try:
        if (arr.shape[0] == 0) or (arr.shape[1] == 0):
            return True
    except:
         if (arr.shape[0] == 0):
             return True

    return False
    

def mask_and_pair(positive_data, negative_data, pos_centroids_dict, neg_centroids_dict, pos_labeled_array, neg_labeled_array, distance, current_ds, thres = 1000, min_width = 5):

    B = pos_labeled_array
    A = neg_labeled_array

    if is_empty(B) or is_empty(A):
        return np.array([])

    # Distance from negative to positive data
    dist_to_B = distance_transform_edt(1 - B)
    A_to_B_distances = dist_to_B * A

    if is_empty(A_to_B_distances[A_to_B_distances > 0]):
        return np.array([])
        
    min_dist = np.min(A_to_B_distances[A_to_B_distances > 0])
    closest_points = np.argwhere(A_to_B_distances == min_dist)

    # # Optionally plot this distance
    # plt.contourf(distance, current_ds.height, A_to_B_distances, cmap='hot')
    # plt.colorbar(label = 'Distance in Pixels')
    # plt.ylabel('Height (km AGL)')
    # plt.xlabel('Distance (km)')
    # plt.show()

    # Distance from positive to negative data
    dist_to_A = distance_transform_edt(1 - A)
    
    B_to_A_distances = dist_to_A * B

    if is_empty(B_to_A_distances[B_to_A_distances > 0]):
        return np.array([])
        
    min_dist = np.min(B_to_A_distances[B_to_A_distances > 0])
    closest_points_back = np.argwhere(B_to_A_distances == min_dist)
    
    # # Optionally plot this distance
    # plt.contourf(distance, current_ds.height, B_to_A_distances, cmap='hot')
    # plt.colorbar(label = 'Distance in Pixels')
    # plt.ylabel('Height (km AGL)')
    # plt.xlabel('Distance (km)')
    # plt.show()

    # Finding the differences in the positive-negative data distance
    dist_diff = A_to_B_distances - B_to_A_distances
    # dist_diff[np.abs(dist_diff) <= 400] = 0 
    dist_diff[np.abs(dist_diff) >= 1200] = 0
    
    # plt.contourf(distance, current_ds.height, dist_diff, cmap = cmaps.curl, levels = np.arange(-1200, 1400, 200))
    # plt.colorbar(label = 'Distance Difference in Pixels')
    # plt.ylabel('Height (km AGL)')
    # plt.xlabel('Distance (km)')
    # plt.show()

    # Binary Mask to apply to actual data
    diff_mask = dist_diff != 0

    # Getting rid of absurdly large distances (see thres in parameters)
    A_to_B_distances[A_to_B_distances > thres] = 0
    B_to_A_distances[B_to_A_distances > thres] = 0

    # Applying the aforementioned diff_mask 
    pdata, ndata = positive_data, negative_data*-1
    for r in range(B.shape[0]):
        for c in range(A.shape[1]):
            if A[r][c] != 0:
                pdata[r][c] = ndata[r][c]
    
    fmask = pdata

    # Grabbing the diff_mask and applying it to fmask (original data)
    if np.all(diff_mask == 0):
        return np.array([])
        
    data_where = np.where(diff_mask, fmask, np.nan)
    # plt.contourf(distance, current_ds.height, data_where, cmap = cmaps.curl, levels = np.arange(-4, 4.25, 0.25))
    # plt.colorbar(label = 'Residual Velocity (m/s)')
    # plt.title(f"Filtered Blobs")
    # plt.ylabel('Height (km AGL)')
    # plt.xlabel('Distance (km)')
    # plt.show()

    # Mask to pair blobs into rolls
    blob_pair_mask = ~np.isnan(data_where)

    # Label connected blobs
    labeled_blobs, num_blobs = label(blob_pair_mask)
    
    # Get bounding boxes
    slices = find_objects(labeled_blobs)
    
    # Filter by width of slices (see min_width parameter) 
    filtered_mask = np.zeros_like(blob_pair_mask, dtype=bool)
    
    for i, slc in enumerate(slices):
        if slc is None:
            continue
    
        blob_id = i + 1
        region = (labeled_blobs[slc] == blob_id)
    
        width = slc[1].stop - slc[1].start  # width = number of columns
    
        if width >= min_width:
            filtered_mask[slc] |= region  # include this blob (greater than min_width [# of columns])
    
    #  Apply to data_where (original data)
    filtered_data = np.where(filtered_mask, data_where, np.nan)
    
    # # Optionally Plot
    # plt.figure()
    # plt.contourf(distance, current_ds.height, filtered_data, cmap=cmaps.curl, levels = np.arange(-4, 4.25, 0.25))
    # plt.colorbar(label = 'Residual Velocity (m/s)')
    # plt.title(f"Blobs with Width ≥ {min_width} pixels")
    # plt.ylabel('Height (km AGL)')
    # plt.xlabel('Distance (km)')
    # plt.show()

    return filtered_data

# --- Wavelength-related functions ---

def find_all_blob_pairs_by_proximity(blob_centroids):
    blob_centroids = np.array(blob_centroids, dtype=np.float32)
    n_blobs, n_heights = blob_centroids.shape

    # To store results: list of matched blob pairs and their distances per height
    matches_per_height = [[] for _ in range(n_heights)]

    # Work on a copy to mask values progressively
    working_copy = blob_centroids.copy()

    for h in range(n_heights):
        while True:
            values = working_copy[:, h]
            valid_idx = np.where(~np.isnan(values))[0]

            if len(valid_idx) < 2:
                break  # No more pairs to match

            # Find all combinations
            pairs = list(combinations(valid_idx, 2))
            diffs = [abs(values[i] - values[j]) for i, j in pairs]

            # Find the closest pair
            min_idx = np.argmin(diffs)
            i, j = pairs[min_idx]
            distance = diffs[min_idx]

            if distance > 2.5:
                distance = np.nan

            # Record the pair and distance
            matches_per_height[h].append({
                'pair': (i, j),
                'wavelength': distance,
                'positions': (values[i], values[j])
            })

            # Mask these two blobs at this height
            working_copy[i, h] = np.nan
            working_copy[j, h] = np.nan

    return matches_per_height

def filter_blob_matches_with_positions(matches_per_height, min_valid_heights=3):
    """
    Filter blob pairs and store their positions and wavelengths at each height.
    Returns dict:
        {
            (i, j): {
                'heights': [height indices],
                'positions': [(pos_i, pos_j), ...],
                'wavelengths': [wavelength, ...]
            }
        }
    """
    from collections import defaultdict
    import numpy as np

    pair_data = defaultdict(lambda: {'heights': [], 'positions': [], 'wavelengths': []})

    for h, matches in enumerate(matches_per_height):
        for match in matches:
            i, j = match['pair']
            wavelength = match['wavelength']
            pos1, pos2 = match['positions']
            pair = tuple(sorted((i, j)))

            if not np.isnan(wavelength):
                pair_data[pair]['heights'].append(h)
                pair_data[pair]['positions'].append((pos1, pos2))
                pair_data[pair]['wavelengths'].append(wavelength)

            
            
    # Filter pairs with enough valid heights
    filtered = {pair: data for pair, data in pair_data.items()
                if (len(data['heights']) >= min_valid_heights) and (data['heights'][0] == 0) and (data['wavelengths'][0] != 0)}

    return filtered

def test_wavelength(heights, distances, filt_data):

    blob_mask = ~np.isnan(filt_data)
    labeled_blobs, num_blobs = label(blob_mask)
    slices = find_objects(labeled_blobs)

    twidths = []
    for i, slc in enumerate(slices):
        if slc is None:
            continue
        width = slc[1].stop - slc[1].start
        hwidth = width // 2
        twidths.append(hwidth)

    centroids = center_of_mass(blob_mask, labeled_blobs, index=range(1, num_blobs + 1))
    blob_centroids = []

    for k, (chi, di) in enumerate(centroids):
        di = int(di)
        hcentroids = []
        for hi, h in enumerate(heights):
            left = max(0, di - twidths[k])
            right = min(filt_data.shape[1], di + twidths[k] + 1)

            near_distance = np.array(distances[left:right])
            # if debug: logging.info(near_distance)
            data = filt_data[hi, left:right]

            if len(data) == 0 or np.all(np.isnan(data)):
                hcentroids.append(np.nan)
            else:
                peak = near_distance[np.nanargmax(data)]
                hcentroids.append(peak)
        blob_centroids.append(hcentroids)

    if debug: logging.info("Blob centroids", blob_centroids)

    # wavelengths, diffs_per_height = wavelength_per_height(blob_centroids)
    
    # for i, (wl, diffs) in enumerate(zip(wavelengths, diffs_per_height)):
    #     if debug: logging.info(f"Height index {i}:")
    #     if debug: logging.info(f"  All differences: {np.round(diffs, 3)}")
    #     minim = np.min(diffs) if diffs else np.nan
    #     if debug: logging.info(f"  Min difference: {minim}")

    if blob_centroids:
        matches_per_height = find_all_blob_pairs_by_proximity(blob_centroids)
    else:
        filtered_blob_pairs = {}
        return filtered_blob_pairs

    filtered_blob_pairs = filter_blob_matches_with_positions(matches_per_height, min_valid_heights=3)
    
    if debug: 
        logging.info("\nFiltered Blob Pairs with Positions:")
    for pair, data in filtered_blob_pairs.items():
        if debug: 
            logging.info(f"Blob pair {pair}:")
            logging.info(f"  Heights: {data['heights']}")
            logging.info(f"  Positions: {data['positions']}")
            logging.info(f"  Wavelengths: {np.round(data['wavelengths'], 3)}")

    return filtered_blob_pairs

# --- The one depth function (based on wavelength heights instead of the other way around) ---

def get_depths(heights, filtered_blob_pairs):

    depths = {}
    for pair, data in filtered_blob_pairs.items():
        hblob = np.array(data['heights'])
        depth = (heights[hblob[-1]] - heights[hblob[0]]) + 0.2 # Adds 0.2 km to account for bottom grid
        depths[pair] = {'depth_km':depth, 'positions':data['positions']}

    return depths

# --- For boolean arguments in the command line ---

def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', '1'): return True
    elif v.lower() in ('no', 'false', 'f', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')


# --- MAIN EXECUTION WORKFLOW ---


def main():
    """
    Parses command-line arguments for the main script.
    """
    parser = argparse.ArgumentParser(description='Perform analysis on HBL rolls from RadXGrid files.')
    parser.add_argument('--grid_dir', type=str, required=True, help='Full path to RadXGrid files.')
    parser.add_argument('--vad_dir', type=str, required=True, help='Base path to the directory containing VAD domain folders (e.g., 5km, 10km).')
    parser.add_argument('--domain_size', type=int, required=True, help='Desired domain size in kilometers.')
    parser.add_argument('--peak_threshold', type=float, default=0.15, help='Relative threshold (of max res. vel. in blob) for peak detection in blobs (e.g., 0.15).')
    parser.add_argument('--area_quantile', type=float, default=0.75, help='Quantile for blob area filtering (e.g., 0.75 for top 25%).')
    parser.add_argument('--plotting', type=str2bool, default=False, help='Turn on/off saving of intermediate cross-section images.')
    parser.add_argument('--radar', type=str, required=True, help="Your radar site ID (e.g., 'KNQA').")
    parser.add_argument('--mixing_length', type=str2bool, default=False, help='Turn on updated mixing length method for momentum flux.')
    parser.add_argument('--debug', type=str2bool, default=False, help='Turn on/off detailed print statements for debugging.')
    args = parser.parse_args()
    if len(sys.argv) <= 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    return args

if __name__ == '__main__':
    args = main()

    # --- Configure Logging ---
    log_filename = 'debug.log'
    debug_mode = args.debug

    