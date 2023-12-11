import  numpy as np

infolder = './input'
outfolder = './output_velSpectra'
gridFile = 'ROMSgrid.nc'


#### gridsize
xlen = 1120
ylen = 434
timelen = 248

startTimeIndex = 0
endTimeIndex = timelen

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
nvar = 4
### add information about variables and 
### the file they are in in the list below
#### DONT FORGET THE COMMAS #####
inputFile = 'ROMS_withWRF_data.nc'
varInfoList = [
    {'name': 'uo',
     'units': 'm/s',
     'long_name': 'zonal velocity',
     'valtype': np.float,
     'file': inputFile},

    {'name': 'vo',
     'units': 'm/s',
     'long_name': 'meridional velocity',
     'valtype': np.float,
     'file': inputFile},

    {'name': 'ua',
     'units': 'm/s',
     'long_name': 'zonal U10 velocity for atm',
     'valtype': np.float,
     'file': inputFile},

    {'name': 'va',
     'units': 'm/s',
     'long_name': 'meridional V10 velocity for atm',
     'valtype': np.float,
     'file': inputFile}
]


#### Filterlength list   ####

ellinKmList = [10, 20, 50, 80, 100, 200, 300, 500, 800]
