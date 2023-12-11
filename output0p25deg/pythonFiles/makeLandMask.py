from netCDF4 import Dataset
import numpy as np


ds = Dataset('/pscratch/sd/s/srai/ROMSwithWRF/run/input/ROMS_withWRF_data.nc')

uo = np.array(ds.variables['uo'][:,:,:])
timeAvguo = np.mean(uo, axis=0)


ds.close()

landMask = timeAvguo == 0 

wds = Dataset('/pscratch/sd/s/srai/ROMSwithWRF/run/input/landMask.nc','w',format='NETCDF4')

wds.createDimension('x', 1120)
wds.createDimension('y', 434)


mask = wds.createVariable('landMask', int, ('y', 'x'))
mask[:,:] = landMask*1

wds.close()

