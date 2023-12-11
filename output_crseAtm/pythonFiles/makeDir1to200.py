import os
ellList = [10, 20, 50, 80, 100, 200, 500, 800]
for ell in ellList:
    #cmd = f'mkdir {ell:d}km/1to200'
    cmd = f'mv {ell:d}km/*_filteredAt_{ell:04d}_timeAt0??.nc {ell:d}km/1to200'
    print(cmd)
    os.system(cmd)
    cmd = f'mv {ell:d}km/*_filteredAt_{ell:04d}_timeAt1??.nc {ell:d}km/1to200'
    print(cmd)
    os.system(cmd)
