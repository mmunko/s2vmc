#!/usr/bin/python3

import glob
import gdal
import numpy as np
import sys
import os

dr = 'raw_rasters'
tifs = glob.glob(os.path.join(dr,'*tif'))

for tif in tifs:
    r = gdal.Open(tif)
    na = r.GetRasterBand(1).GetNoDataValue()
    d = r.GetRasterBand(1).ReadAsArray()
    print('{}\t rozmery {} x {}, pocet null {}, na value {}'.format(tif,r.RasterXSize,r.RasterYSize,len(d[d == na]),na))
