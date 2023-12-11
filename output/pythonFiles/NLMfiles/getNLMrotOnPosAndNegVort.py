from netCDF4 import Dataset
import numpy as np


def createVariableAndWrite(wds, varName, varUnits, varLongName, varDimension, varType, varArr):
    cdfVar = wds.createVariable(varName, varType, varDimension)
    cdfVar.units = varUnits
    cdfVar.long_name = varLongName
    cdfVar[:] = varArr[:]


def setMaskZeroToNan(ds, varName):
    var = np.array(ds.variables[varName][0, :, :]).copy()
    var[var == 0] = float('nan')
    return var


if __name__ == '__main__':
    #rootFolder = '/discover/nobackup/projects/mesocean/srai/runGeos7dayAvg/'
    rootFolder = '/pscratch/sd/s/srai/ROMSwithWRF/run/'
    ds_masks = Dataset(rootFolder + 'input/landMask.nc')
    GridFile = rootFolder + 'input/ROMSgrid.nc'

    gridDS = Dataset(GridFile)
    UAREA = np.array(gridDS.variables['UAREA'])
    ylen, xlen = np.shape(UAREA)
    DX = np.array(gridDS.variables['DXU'])
    DY = np.array(gridDS.variables['DYU'])

    ellList = [10, 20, 50, 80, 100]
    nell = len(ellList)
    ndays = 200

    landMask = np.array(ds_masks.variables['landMask'][:,:], dtype=bool)

    for ellIDX in range(nell):
        ell = ellList[ellIDX]
        ellFold = rootFolder + 'output/{0:d}km/1to200/'.format(ell)

        writeFileName = ellFold + \
            'NLMonPosAndNegVort_{0:04d}km.nc'.format(
                ell)

        NLM_pos_vort = np.zeros((ndays, ylen, xlen), dtype=float)
        NLM_neg_vort = np.zeros((ndays, ylen, xlen), dtype=float)

        NLMfileName = 'NLmodelEP_{0:04d}km.nc'.format(ell)
        ds_2 = Dataset(ellFold + NLMfileName)

        fileName = 'okuboWeissAndStrainDirec_{0:04d}km_AllDay.nc'.format(
            ell)

        ds = Dataset(ellFold + fileName)

        TIME = ds.variables['time']

        print('all variables read')

        for dayIDX in range(ndays):
            # fileName = 'okuboWeissAndStrainDirec_{0:04d}km_day{1:03d}.nc'.format(
            #     ell, dayIDX+1)

            print('working in day {0:d} for ell {1:d}km'.format(
                dayIDX+1, ell))

            # okuboWeiss_vel = np.array(ds.variables['okuboWeiss_vel'][dayIDX, :, :])

            omega_vel = np.array(ds.variables['vorticity'][dayIDX, :, :])

            # vortDom_vel = okuboWeiss_vel <= 0 #2e-10

            # posVortMask = np.logical_and(omega_vel >= 0, vortDom_vel)
            # negVortMask = np.logical_and(omega_vel < 0, vortDom_vel)

            posVortMask = omega_vel > 0
            negVortMask = ~posVortMask

            ###############################################################################################################

            NLM_rot = np.array(
                ds_2.variables['NLmodel_EPCg_rot'][dayIDX, :, :]).copy()

            NLM_pos_vort[dayIDX, :, :] = NLM_rot.copy()
            NLM_neg_vort[dayIDX, :, :] = NLM_rot.copy()

            NLM_pos_vort[dayIDX, ~posVortMask] = float('nan')
            NLM_neg_vort[dayIDX, ~negVortMask] = float('nan')

        wds = Dataset(writeFileName, 'w', format='NETCDF4')

        wds.createDimension('Y', ylen)
        wds.createDimension('X', xlen)
        wds.createDimension('time', None)

        cdftime = wds.createVariable('time', float, ('time'))
        #cdftime.units = 'days since 0050-10-01 00:00:00'
        timeArr = np.array(TIME)
        cdftime[:] = timeArr[:]


        createVariableAndWrite(wds, 'posVortNLMrot', 'watts/m^2',
                               'NLM_rot on positive Vorticity',
                               ('time', 'Y', 'X'),
                               float,
                               NLM_pos_vort)

        createVariableAndWrite(wds, 'negVortNLMrot', 'watts/m^2',
                               'NLM_rot on negative Vorticity',
                               ('time', 'Y', 'X'),
                               float,
                               NLM_neg_vort)

        # createVariableAndWrite(wds, 'posVortNLMrot_AreaInt', 'watts',
        #                       'NLM_rot on positive Vorticity',
        #                       ('time', 'Y', 'X'),
        #                       float,
        #                       NLM_pos_vort_AreaInt)

        # createVariableAndWrite(wds, 'negVortNLMrot_AreaInt', 'watts',
        #                       'NLM_rot on negative Vorticity',
        #                       ('time', 'Y', ''),
        #                       float,
        #                       NLM_neg_vort_AreaInt)

        wds.close()


