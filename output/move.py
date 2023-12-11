import os
ellList = [10, 20, 50, 80, 100, 200, 300, 500, 800]
for ell in ellList:
    #cmd = 'mkdir -p {0:d}km'.format(ell)
    cmd = 'mv *_filteredAt_{0:04}_timeAt???.nc {0:d}km'.format(ell)
    print(cmd)
    os.system(cmd)
