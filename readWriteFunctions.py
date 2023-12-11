from netCDF4 import Dataset
import numpy as np
import sys


def getGrid(gridFile):
    ## read angles in radians
    ds = Dataset(gridFile)
    ULAT = np.array(ds.variables['ULAT'])
    ULONG = np.array(ds.variables['ULONG'])
    RADIUS = ds.variables['RADIUS'][0]
    ds.close()
    return ULAT, ULONG, RADIUS


def getVariable(varFile, varName, index):
    ## read a variable
    ds = Dataset(varFile)
    var = ds.variables[varName]
    print('\n\n Var Info Start')
    print(var)
    print('\n\n Var Info End')
    sys.stdout.flush()
    varVal = np.array(var[index,:,:])
    ds.close()
    return varVal


def createWriteHandleForNetCDF(writeFileName, dimInfoList):
    wds = Dataset(writeFileName, 'w', format='NETCDF4')

    #### example of dimension info is as below  ######
    # dimensionInfo1 = {'name': 'latitude',
    #              'len': len(lat),
    #              'val': lat,
    #              'valtype': np.float
    #              'units': 'radians'}

    ndims = len(dimInfoList)
    for i in range(ndims):
        name = dimInfoList[i]['name']
        length = dimInfoList[i]['len']
        val = dimInfoList[i]['val']
        valtype = dimInfoList[i]['valtype']
        units = dimInfoList[i]['units']
        wds.createDimension(name, length )

        var = wds.createVariable(name, valtype, (name,))
        var.units = units
        var[:] = val[:]

    return wds

def createVariableAndWrite(wds, varName, varUnits, varLongName, 
                           varDimension, varType, varArr):
    ## wds pointer to write dataset
    ## varName string type
    ## varUnits string type
    ## varLongName string type description of units
    ## varDimension string tuple for name of dimension
    ## vartype data type
    ## varArr value of the variables to be written

    cdfVar = wds.createVariable(varName, varType, varDimension)
    cdfVar.units = varUnits
    cdfVar.long_name = varLongName
    cdfVar[:] = varArr[:]
