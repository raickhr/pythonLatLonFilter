import  numpy as np

infolder = './input'
outfolder = './output_Recipe'
gridFile = 'ROMSgrid.nc'


#### gridsize
xlen = 1120
ylen = 434
timelen = 31

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
nvar = 6
### add information about variables and 
### the file they are in in the list below
#### DONT FORGET THE COMMAS #####
inputFile = 'ROMS_withWRF_data_Recipe_dailyAvg.nc'
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

    {'name': 'taux',
     'units': 'Pascal',
     'long_name': 'zonal wind stress',
     'valtype': np.float,
     'file': inputFile},

    {'name': 'tauy',
     'units': 'Pascal',
     'long_name': 'meridional wind stress',
     'valtype': np.float,
     'file': inputFile},

     {'name': 'tauxUo',
     'units': 'm^2/s^2',
     'long_name': 'product of uo and taux',
     'valtype': np.float,
      'file': inputFile},

     {'name': 'tauyVo',
      'units': 'm^2/s^2',
      'long_name': 'product of vo and tauy',
      'valtype': np.float,
      'file': inputFile}
]


#### Filterlength list   ####

ellinKmList = [10, 20, 50, 80, 100, 200, 300, 500, 800]
