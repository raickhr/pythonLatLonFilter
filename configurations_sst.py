import  numpy as np

infolder = './input'
outfolder = './output_orig_2016'
gridFile = 'ROMSgrid.nc'


#### gridsize
xlen = 1120
ylen = 434
timelen = 56

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
nvar = 2
### add information about variables and 
### the file they are in in the list below
#### DONT FORGET THE COMMAS #####
inputFile = 'ROMS_withWRF_data_inst.nc'
varInfoList = [
    {'name': 'sst',
     'units': 'deg C',
     'long_name': 'sea surface potential temperature',
     'valtype': np.double,
     'file': inputFile},

     'name': 'sstSq',
     'units': '(deg C)^2',
     'long_name': 'sea surface potential temperature squared',
     'valtype': np.double,
     'file': inputFile}

]


#### Filterlength list   ####

ellinKmList = [10, 20, 50, 80, 100, 200, 300, 500, 800]
