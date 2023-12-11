import  numpy as np

infolder = './input_conv_k4'
outfolder = './output_conv_k4'
gridFile = 'ROMSgrid.nc'


#### gridsize
xlen = 1120
ylen = 434
timelen = 248

startTimeIndex = 30
endTimeIndex = 31 #timelen

# dimension info list
dimInfoList = [
    {'name': 'ULAT',
     'len': ylen,
     'val': [],
     'valtype': np.float,
     'units': 'radians'},

    {'name': 'ULONG',
     'len': xlen,
     'val': [],
     'valtype': np.float,
     'units': 'radians'}
]


# number of variables
nvar = 8   
### add information about variables and 
### the file they are in in the list below
#### DONT FORGET THE COMMAS #####
varInfoList = [
    {'name': 'uo',
     'units': 'm/s',
     'long_name': 'zonal velocity',
     'valtype': np.float,
     'file': 'ROMS_withWRF_data_tavg_56.nc'},

    {'name': 'vo',
     'units': 'm/s',
     'long_name': 'meridional velocity',
     'valtype': np.float,
     'file': 'ROMS_withWRF_data_tavg_56.nc'},

    {'name': 'taux',
     'units': 'Pascal',
     'long_name': 'zonal wind stress',
     'valtype': np.float,
     'file': 'ROMS_withWRF_data_tavg_56.nc'},

    {'name': 'tauy',
     'units': 'Pascal',
     'long_name': 'meridional wind stress',
     'valtype': np.float,
     'file': 'ROMS_withWRF_data_tavg_56.nc'},

     {'name': 'tauxUo',
     'units': 'm^2/s^2',
     'long_name': 'product of uo and taux',
     'valtype': np.float,
      'file': 'ROMS_withWRF_data_tavg_56.nc'},

     {'name': 'tauyVo',
      'units': 'm^2/s^2',
      'long_name': 'product of vo and tauy',
      'valtype': np.float,
      'file': 'ROMS_withWRF_data_tavg_56.nc'},

    {'name': 'uoSq',
     'units': 'm^2/s^2',
     'long_name': 'square of uo',
     'valtype': np.float,
     'file': 'ROMS_withWRF_data_tavg_56.nc'},

    {'name': 'voSq',
     'units': 'm^2/s^2',
     'long_name': 'square of vo',
     'valtype': np.float,
     'file': 'ROMS_withWRF_data_tavg_56.nc'},
]


#### Filterlength list   ####

ellinKmList = [100, 200, 300]
