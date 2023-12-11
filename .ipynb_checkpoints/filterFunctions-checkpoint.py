import numpy as np
from numpy import sin, cos, tan, arctan, deg2rad, rad2deg
from scipy import signal, fft, interpolate
from scipy.ndimage import gaussian_filter
import sys

def getGeodesicDistFromLonLat(center_lon, center_lat, LON, LAT, Radius):
    ### calculates Geodesic distance from center_lon, center_lat
    ### all inputs are given as radians
    ### radius is given in meters

    phi2 = (LAT)
    phi1 = (center_lat)
    dlambda = (LON - center_lon)
    num = np.sqrt((cos(phi2) * sin(dlambda))**2 + (cos(phi1) *
                  sin(phi2) - sin(phi1)*cos(phi2) * cos(dlambda))**2)
    den = (sin(phi1)*sin(phi2) + cos(phi1)*cos(phi2) * cos(dlambda))
    dsigma = np.arctan2(num, den)
    r = Radius*dsigma
    return r


def getKernel(rinKm, ellinKm, UAREA):
    ###UAREA is in m^2
    G = (0.5 - 0.5*np.tanh((rinKm - ellinKm/2)/10))
    # G[rinKm > ellinKm/2] = 0.0

    ############# GAUSSIAN FILTER ###############
    #sigma = 1/4 * (ellinKm/2)
    #G = 1/(sigma*2*np.pi) * np.exp(-1/(2*sigma**2)*rinKm**2)
    ############################
    normalization = np.nansum(G*UAREA)
    kernel = G/normalization
    return kernel

def getKernelAtLat(ell, lat_index, LON, LAT, UAREA, Radius):
    #### ell is given in meters
    #### all units is given at radians and meters##
    #### this is for only lat lon Grid
    #### tolerance is set to 100km

    tolerance = 100e3
    ylen, xlen = np.shape(UAREA)

    x_center = int(xlen//2)

    center_lat = LAT[lat_index, x_center]
    center_lon = LON[lat_index, x_center]

    dlon = LON[0, 1] - LON[0, 0]
    dlat = LAT[1, 0] - LAT[0, 0]

    #print('Radius, center_lat', Radius, center_lat)
    sys.stdout.flush()
    half_lonRange = (ell+tolerance)/(2*Radius*cos(center_lat))
    half_latRange = (ell+tolerance)/(2*Radius)

    half_nlon = int(half_lonRange//dlon)
    half_nlat = int(half_latRange//dlat)

    x_s = x_center-half_nlon
    x_e = x_center+half_nlon + 1

    if x_s < 0 or x_e > xlen:
        print('domain not wide enough !!! increase in x-direction')
        return False

    y_s = lat_index - half_nlat
    y_e = lat_index + half_nlat + 1

    k_center_x = half_nlon
    k_center_y = half_nlat

    if y_s < 0:
        y_s = 0  # this clips the kernel at northern border
        k_center_y = lat_index  # y kernel center according to clipping
    if y_e > ylen:
        y_e = ylen  # this clips the kernel at southern border

    k_UAREA = UAREA[y_s:y_e, x_s:x_e].copy()
    k_LAT = LAT[y_s:y_e, x_s:x_e].copy()
    k_LON = LON[y_s:y_e, x_s:x_e].copy()

    r = getGeodesicDistFromLonLat(center_lon, center_lat, k_LON, k_LAT, Radius)
    kernel = getKernel(r/1e3, ell/1e3, k_UAREA)

    return kernel, k_center_x, k_center_y, k_UAREA


def getKernelAtLatLon(ell, lon_index, lat_index, LON, LAT, UAREA, Radius):
    # ell is given in meters
    #### all units is given at radians and meters##
    # this is for only lat lon Grid
    # tolerance is set to 100km

    tolerance = 100e3
    ylen, xlen = np.shape(UAREA)

    center_lat = LAT[lat_index, lon_index]
    center_lon = LON[lat_index, lon_index]

    dlon = LON[0, 1] - LON[0, 0]
    dlat = LAT[1, 0] - LAT[0, 0]

    # print('Radius, center_lat', Radius, center_lat)
    sys.stdout.flush()
    half_lonRange = (ell+tolerance)/(2*Radius*cos(center_lat))
    half_latRange = (ell+tolerance)/(2*Radius)

    half_nlon = int(half_lonRange//dlon)
    half_nlat = int(half_latRange//dlat)

    x_s = lon_index - half_nlon
    x_e = lon_index + half_nlon + 1

    y_s = lat_index - half_nlat
    y_e = lat_index + half_nlat + 1

    k_center_x = half_nlon
    k_center_y = half_nlat

    if y_s < 0:
        y_s = 0  # this clips the kernel at northern border
        k_center_y = lat_index  # y kernel center according to clipping
    if y_e > ylen:
        y_e = ylen  # this clips the kernel at southern border

    if x_s < 0:
        x_s = 0  # this clips the kernel at west border
        k_center_x = lon_index  # y kernel center according to clipping
    if x_e > xlen:
        x_e = ylen  # this clips the kernel at east border

    k_UAREA = UAREA[y_s:y_e, x_s:x_e].copy()
    k_LAT = LAT[y_s:y_e, x_s:x_e].copy()
    k_LON = LON[y_s:y_e, x_s:x_e].copy()

    r = getGeodesicDistFromLonLat(center_lon, center_lat, k_LON, k_LAT, Radius)
    kernel = getKernel(r/1e3, ell/1e3, k_UAREA)

    return kernel, k_center_x, k_center_y, k_UAREA


def getFilteredAtij(field, k_UAREA, yindx, xindx, kernel, k_center_x, k_center_y):
    k_ylen, k_xlen = np.shape(kernel)
    shifted_field = np.roll(
        field, (k_center_y-yindx, k_center_x-xindx), axis=(0, 1))

    k_field = shifted_field[0:k_ylen, 0:k_xlen].copy()
    field_bar_ji = np.sum(k_field * kernel * k_UAREA)

    return field_bar_ji


def getFilteredFieldParallel(ell, field_list, UAREA, LON, LAT, Radius, 
                             startIndex, endIndx, rank):

    #### endIndx is not included for filtering 

    nfields, ylen, xlen = np.shape(field_list)
    fieldbar_list = np.zeros((nfields, ylen, xlen), dtype=float)

    start_j, start_i = int(startIndex//xlen), startIndex%xlen
    end_j, end_i = int(endIndx//xlen)+1, endIndx%xlen

    if end_i == 0:
        end_j -=1
        end_i = xlen

    #print('j start j end',start_j, end_j, 'rank', rank)
    #print('i start i end',start_i, end_i, 'rank', rank)

    sys.stdout.flush()

    for j in range(start_j, end_j):
        print('completed {0:5.2f}% at rank {1:d}'.format((j-start_j)/(end_j-start_j)*100, rank))
        #print('at lat index', j, 'lat {0:8.5f} deg'.format(rad2deg(LAT[j,0])))
        sys.stdout.flush()
        
        kernel, k_center_x, k_center_y, k_UAREA = getKernelAtLat(
            ell, j, LON, LAT, UAREA, Radius)

        
        xstartIndx = 0
        xendIndx = xlen

        if j == start_j:
            xstartIndx = start_i

        if j == end_j:
            xendIndx = end_i
        
        #print('max xindex', xendIndx)
        sys.stdout.flush()
        
        for i in range(xstartIndx,xendIndx):
            #if (i- k_center_x) < 0 or (i+k_center_x) > (xlen+1):
            #    kernel, k_center_x, k_center_y, k_UAREA = getKernelAtLatLon(
            #        ell, i, j, LON, LAT, UAREA, Radius)

            for n in range(nfields):
                fieldbar_list[n, j, i] = getFilteredAtij(field_list[n, :, :],
                                                         k_UAREA,
                                                         j, i,
                                                         kernel,
                                                         k_center_x, k_center_y)
    
    # print(fieldbar_list[0, start_j, start_i:start_i+3])
    # print(fieldbar_list[1, start_j, start_i:start_i+3])
    # print(fieldbar_list[2, start_j, start_i:start_i+3])
    return fieldbar_list
