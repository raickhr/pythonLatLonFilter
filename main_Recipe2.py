import numpy as np
from numpy import sin, cos, tan, arctan, deg2rad, rad2deg
from netCDF4 import Dataset
from mpi4py import MPI
from readWriteFunctions import *
from filterFunctions import *
from readWriteFunctions import *
from configurations_Recipe2 import *
import sys

### initialize MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

if rank == 0:
    #### angles should be in radians and in 2D array
    #### the grid should be uniform in angles
    #### x-axis is array dim 1
    #### y-axis is array dim 0

    ULAT, ULONG, RADIUS = getGrid(infolder+'/'+ gridFile) 
    radobj = np.array([RADIUS], dtype=float)
    dlon = ULONG[0, 1] - ULONG[0, 0]
    dlat = ULAT[1, 0] - ULAT[0, 0]
    DXU = RADIUS * cos(ULAT)*dlon
    DYU = np.full((ylen, xlen), RADIUS * dlat)
    UAREA = DXU*DYU

else:
    radobj = np.empty((1), dtype=float)
    DXU = np.empty((ylen, xlen), dtype=float)
    DYU = np.empty((ylen, xlen), dtype=float)
    ULAT = np.empty((ylen, xlen), dtype=float)
    ULONG = np.empty((ylen, xlen), dtype=float)
    UAREA = np.empty((ylen, xlen), dtype=float)
    
comm.Bcast(radobj, root=0)
comm.Bcast(DXU, root=0)
comm.Bcast(DYU, root=0)
comm.Bcast(ULAT, root=0)
comm.Bcast(ULONG, root=0)
comm.Bcast(UAREA, root=0)
RADIUS = radobj[0]
comm.Barrier()

sys.stdout.flush()

# divide work
startIndexArr = np.zeros((nprocs), dtype=int)
endIndexArr = np.zeros((nprocs), dtype=int)
npointsArr = np.zeros((nprocs), dtype=int)

totNumPoints = xlen * ylen
avgNpoints = int(totNumPoints//nprocs)
remainder = totNumPoints%nprocs

startIndex = 0
if rank == 0:
    print(totNumPoints)
for procs_id in range(nprocs):
    if rank == 0:
        print(startIndex)
    startIndexArr[procs_id] = startIndex
    npoints = avgNpoints
    if procs_id < remainder:
        npoints += 1
    npointsArr[procs_id] = npoints
    startIndex += npoints
    endIndexArr[procs_id] = startIndex


if startIndex != totNumPoints:
    print('error in division of work!')
    sys.stdout.flush()

comm.Barrier()
# print('div work rank start, end', rank, startIndexArr[rank], endIndexArr[rank] )
sys.stdout.flush()

for time in range(startTimeIndex, endTimeIndex):
    if rank == 0:
        # read varnames and variables
        varValArray = np.zeros((nvar, ylen, xlen), dtype=float)
        filteredVarValArray = np.zeros((nvar, ylen, xlen), dtype=float)
        for varIndx in range(nvar):
            varName = varInfoList[varIndx]['name']
            fileName = infolder + '/' + varInfoList[varIndx]['file']
            varValArray[varIndx, :, :] = getVariable(fileName, varName, time)

    else:
        varValArray = np.empty((nvar, ylen, xlen), dtype=float)
        filteredVarValArray = np.ones((nvar, ylen, xlen), dtype=float)*float('nan')

    comm.Bcast(varValArray, root=0)
    comm.Barrier()
    # print('rank', rank, 'Radius', RADIUS)
    # print('rank', rank, 'DXU', DXU[0:5, 0:5])
    # print('rank', rank, 'DYU', DYU[0:5, 0:5])
    # print('rank', rank, 'ULAT', ULAT[0:5, 0:5])
    # print('rank', rank, 'ULONG', ULONG[0:5, 0:5])
    # print('rank', rank, 'UAREA', UAREA[0:5, 0:5])
    # print('rank', rank, 'varValArray 0', varValArray[0, 0:5, 0:5])
    # print('rank', rank, 'varValArray 1', varValArray[1, 0:5, 0:5])
    # print('rank', rank, 'varValArray 2', varValArray[2, 0:5, 0:5])


    for ellinKm in ellinKmList:
        print('\n\nell = {0:d} km\n\n'.format(ellinKm))
        sys.stdout.flush()

        ell = ellinKm*1e3
        firstfileName = str(varInfoList[0]['file'])
        firstfile = firstfileName[:-3]
        writeFileName = outfolder +'/' + firstfile + '_filteredAt_{0:04d}_timeAt{1:03d}.nc'.format(ellinKm, time)

        filteredVarValArray = getFilteredFieldParallel(ell, varValArray, 
                                                        UAREA, ULONG, ULAT,
                                                        RADIUS, startIndexArr[rank], 
                                                        endIndexArr[rank], rank)

        start_j, start_i = int(startIndexArr[rank]//xlen), startIndexArr[rank] % xlen
        end_j, end_i = int(endIndexArr[rank]//xlen), endIndexArr[rank] % xlen

        # if rank == 2:
        #     print('before send', filteredVarValArray[0, start_j, start_i:start_i+3])
        #     print('before send', filteredVarValArray[1, start_j, start_i:start_i+3])
        #     print('before send', filteredVarValArray[2, start_j, start_i:start_i+3])

        for varIdx in range(nvar):
            dummy = filteredVarValArray[varIdx,:,:].copy()


            bufSendfiltVar = dummy.flatten()[startIndexArr[rank]:endIndexArr[rank]]
            bufRecvfiltVar = np.empty((totNumPoints,), dtype=float)


            #comm.Gather(bufSendfiltVar, bufRecvfiltVar, root=0)
            # comm.Gatherv(sendbuf=bufSendfiltVar, recvbuf=(
            #     bufRecvfiltVar, npointsArr[rank], startIndexArr, MPI.DOUBLE), root=0)

            comm.Gatherv(sendbuf=bufSendfiltVar, recvbuf=(
                bufRecvfiltVar, npointsArr, startIndexArr, MPI.DOUBLE), root=0)

            comm.Barrier()
            if rank == 0:
                filteredVarValArray[varIdx,:,:] = bufRecvfiltVar.reshape(ylen, xlen)
                # print('after recv', filteredVarValArray[0, start_j, start_i:start_i+3])
                # print('after recv', filteredVarValArray[1, start_j, start_i:start_i+3])
                # print('after recv', filteredVarValArray[2, start_j, start_i:start_i+3])

        comm.Barrier()

        if rank == 0:
            dimInfoList[0]['val'] = ULAT[:,0]
            dimInfoList[1]['val'] = ULONG[0, :]
            print('writing file', writeFileName)
            wds = createWriteHandleForNetCDF(writeFileName, dimInfoList)

            for varIndx in range(nvar):

                varName = varInfoList[varIndx]['name']
                varUnits = varInfoList[varIndx]['units']
                varLongName = varInfoList[varIndx]['long_name']
                varDimension = (dimInfoList[0]['name'], dimInfoList[1]['name'])

                createVariableAndWrite(wds, varName, varUnits, varLongName,
                                    varDimension, varInfoList[varIndx]['valtype'], 
                                    filteredVarValArray[varIndx,:,:])

            wds.close()
            print('write ends\n\n')
        
        comm.Barrier()


