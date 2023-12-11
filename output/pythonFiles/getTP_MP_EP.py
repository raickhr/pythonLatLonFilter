from netCDF4 import Dataset
import numpy as np


rootFolder = '/pscratch/sd/s/srai/ROMSwithWRF/run/output/'

ellList = [10, 20, 50, 80, 100]

timeLen, Ylen, Xlen = 200, 434, 1120

timeUnits = 'hours since 0001-01-01 00:00:00'

for ell in ellList:
    print('in ell', ell)

    ellFold = rootFolder + f'{ell}km/1to200/'

    # TAUX = np.zeros((timeLen, Ylen, Xlen), dtype=float)
    # TAUY = np.zeros((timeLen, Ylen, Xlen), dtype=float)
    # UVEL = np.zeros((timeLen, Ylen, Xlen), dtype=float)
    # VVEL = np.zeros((timeLen, Ylen, Xlen), dtype=float)

    # TPPA = np.zeros((timeLen, Ylen, Xlen), dtype=float)
    # MPPA = np.zeros((timeLen, Ylen, Xlen), dtype=float)
    # EPPA = np.zeros((timeLen, Ylen, Xlen), dtype=float)
    # timeVal = np.zeors((timeLen), dtype=float)

    writeFileName = ellFold + f'filtered_{ell:04d}.nc'
    wds = Dataset(writeFileName, 'w', format='NETCDF4')
    wds.createDimension('Y', Ylen)
    wds.createDimension('X', Xlen)
    wds.createDimension('time', None)

    dsTimeVal = wds.createVariable('time', float, ('time'))

    dsTAUX = wds.createVariable('TAUX', float, ('time', 'Y','X'))
    dsTAUY = wds.createVariable('TAUY', float, ('time', 'Y','X'))
    dsUVEL = wds.createVariable('UVEL', float, ('time', 'Y','X'))
    dsVVEL = wds.createVariable('VVEL', float, ('time', 'Y','X'))

    dsTPPA = wds.createVariable('TotalPowerPerArea', float, ('time', 'Y','X'))
    dsMPPA = wds.createVariable('MeanPowerPerArea', float, ('time', 'Y','X'))
    dsEPPA = wds.createVariable('EddyPowerPerArea', float, ('time', 'Y','X'))


    for timeIndex in range(timeLen):
        print('\t\t', timeIndex)
        fileName = ellFold + f'ROMS_withWRF_data_filteredAt_{ell:04d}_timeAt{timeIndex:03d}.nc'
        
        ds = Dataset(fileName)

        txUo = np.array(ds.variables['tauxUo'][0, :, :])
        tyVo = np.array(ds.variables['tauyVo'][0, :, :])
        tx = np.array(ds.variables['taux'][0, :, :])
        ty = np.array(ds.variables['tauy'][0, :, :])
        uo = np.array(ds.variables['uo'][0, :, :])
        vo = np.array(ds.variables['vo'][0, :, :])

        dsTimeVal[timeIndex] = ds.variables['time'][0]

        ds.close()

        dsTAUX[timeIndex, :, :] = tx.copy()
        dsTAUY[timeIndex, :, :] = ty.copy()

        dsUVEL[timeIndex, :, :] = uo.copy()
        dsVVEL[timeIndex, :, :] = vo.copy()

        dsTPPA[timeIndex, :, :] = txUo + tyVo
        dsMPPA[timeIndex, :, :] = tx*uo + ty*vo
        dsEPPA[timeIndex, :, :] = txUo - tx*uo + tyVo - ty *vo

    wds.close()

