import os
import numpy as np

ellList=np.arange(60,450,20)
ellList=np.append(ellList, np.arange(500,1050,100), axis=0)
ellList = np.append([50], ellList, axis =0)

ellList = [20,50,80,100]
for ell in ellList:
    cmd = 'python pythonFiles/addRecDimAtEll.py --ell={0:d}'.format(ell)
    print(cmd)
    os.system(cmd)

