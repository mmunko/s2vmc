#!/usr/bin/python3

import numpy as np
import os, sys, glob, gdal, shutil

from sklearn.externals import joblib
from tools.tifftools import writeTiff

clasificator = sys.argv[1]

if not os.path.exists('maps'):
    os.mkdir('maps')

model = joblib.load(clasificator)

partials = glob.glob('bin/data*')
partials.sort()

if os.path.exists('maps/parts'):
    shutil.rmtree('maps/parts')
os.mkdir('maps/parts')

for part in partials:
    part_name = part.split('/')[-1][5:14]

    data = np.load(part)
    
    if data.shape[0] > 0:
        predicted = model.predict_proba(data)[:,1]

        r = gdal.Open(glob.glob(os.path.join('test-train-classify','classify',part_name,'*tif'))[0])
        d = r.GetRasterBand(1).ReadAsArray()
        noData = r.GetRasterBand(1).GetNoDataValue()
        d[d != noData] = predicted
        d[d == noData] = 10

        writeTiff(os.path.join('maps/parts','{}.tif'.format(part_name)),d,r,1,10,gdal.GDT_Float32)

os.system('gdal_merge.py -o maps/{}.tif -n 10 maps/parts/*tif'.format(clasificator.split('/')[-1][:-4]))
shutil.rmtree('maps/parts')
