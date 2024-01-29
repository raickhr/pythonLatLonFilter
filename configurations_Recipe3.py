import  numpy as np

infolder = './input'
outfolder = './output_Recipe3_2016'
gridFile = 'ROMSgrid.nc'


#### gridsize
xlen = 1120
ylen = 434
timelen = 100

startTimeIndex = 0
endTimeIndex = timelen

# dimension info list
dimInfoList = [
    {'name': 'ULAT',
     'len': ylen,
     'val': [],
     'valtype': np.double,
     'units': 'radians'},

    {'name': 'ULONG',
     'len': xlen,
     'val': [],
     'valtype': np.double,
     'units': 'radians'}
]


# number of variables
nvar = 10
### add information about variables and 
### the file they are in in the list below
#### DONT FORGET THE COMMAS #####
inputFile = 'ROMS_withWRF_0p25degCrse_data_Recipe3_inst_2p5deg.nc'
varInfoList = [
    {'name': 'uo',
     'units': 'm/s',
     'long_name': 'zonal velocity',
     'valtype': np.double,
     'file': inputFile},

    {'name': 'vo',
     'units': 'm/s',
     'long_name': 'meridional velocity',
     'valtype': np.double,
     'file': inputFile},

    {'name': 'taux',
     'units': 'Pascal',
     'long_name': 'zonal wind stress',
     'valtype': np.double,
     'file': inputFile},

    {'name': 'tauy',
     'units': 'Pascal',
     'long_name': 'meridional wind stress',
     'valtype': np.double,
     'file': inputFile},

     {'name': 'tauxUo',
     'units': 'm^2/s^2',
     'long_name': 'product of uo and taux',
     'valtype': np.double,
      'file': inputFile},

     {'name': 'tauyVo',
      'units': 'm^2/s^2',
      'long_name': 'product of vo and tauy',
      'valtype': np.double,
      'file': inputFile},

     {'name': 'uaSq',
     'units': 'm^2/s^2',
     'long_name': 'square of ua',
     'valtype': np.double,
      'file': inputFile},

     {'name': 'vaSq',
      'units': 'm^2/s^2',
      'long_name': 'square of va',
      'valtype': np.double,
      'file': inputFile},

     {'name': 'uoSq',
     'units': 'm^2/s^2',
     'long_name': 'square of uo',
     'valtype': np.double,
      'file': inputFile},

     {'name': 'voSq',
      'units': 'm^2/s^2',
      'long_name': 'square of vo',
      'valtype': np.double,
      'file': inputFile}

]


#### Filterlength list   ####

ellinKmList = [10, 20, 50, 80, 100, 200, 300, 500, 800]