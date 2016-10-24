import gdal

def writeTiff(name,data, example, nbands,na,dtype):
    """
    Funkcia ktora vytvori raster podla predlohy (referencia, suradnicovy system ....) a priradi donho hodnoty z dat
    """
    driver = gdal.GetDriverByName('GTiff')
    output = driver.Create(name, example.RasterXSize, example.RasterYSize, nbands, dtype)
    output.SetProjection(example.GetProjection())
    output.SetGeoTransform(example.GetGeoTransform())
    if nbands > 1:
        pass
    else:
        output.GetRasterBand(1).SetNoDataValue(na)
        output.GetRasterBand(1).WriteArray(data)
    output.FlushCache()
    output.FlushCache()
    
    return 'Raster {} bol uspesne zapisany.'.format(name)
