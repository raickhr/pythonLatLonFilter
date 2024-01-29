import os
ellList = [10, 20, 50, 80, 100, 200, 300, 500, 800]
for ell in ellList:
    #cmd = f'mkdir {ell:d}km'
    #cmd = f'mv {ell:d}km/*_filteredAt_{ell:04d}_timeAt0??.nc {ell:d}km'
    #print(cmd)
    #os.system(cmd)
    cmd = f'mv *_filteredAt_{ell:04d}_timeAt*.nc {ell:d}km'
    print(cmd)
    os.system(cmd)
