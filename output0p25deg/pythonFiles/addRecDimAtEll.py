import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--ell', help="FilterLength", type=int, default=50)

args = parser.parse_args()
ell = args.ell

for i in range(200):
    dimInfo = 'defdim("time",1);time[time]={0:f};time@long_name="Time";time@units="hours since 0001-01-01 00:00:00"'.format(3*i)
    fileName = '{1:d}km/ROMS_withWRF_data_filteredAt_{1:04d}_timeAt{0:03d}.nc'.format(i,ell)
    cmd = "ncap2 -s '{0:s}' -O {1:s} {1:s}".format(dimInfo, fileName)
    cmd2 = 'ncks -O --mk_rec_dmn time {0:s} {0:s}'.format(fileName)
    cmd4 = "ncecat -O -u time {0:s} {0:s}".format(fileName)
    cmd3 = "ncwa -O -a time {0:s} {0:s}".format(fileName)
    
    print(cmd)
    os.system(cmd)

    print(cmd2)
    os.system(cmd2)
    
    print(cmd3)
    os.system(cmd3)

    print(cmd4)
    os.system(cmd4)




