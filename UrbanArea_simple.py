from qgis.PyQt.QtCore import QCoreApplication

from qgis.core import (QgsProcessing,
                       QgsFeatureSink,
                       QgsProcessingException,
                       QgsProcessingAlgorithm,
                       QgsMessageLog,
                       QgsProcessingParameterFeatureSource,
                       QgsProcessingParameterFeatureSink,
                       QgsProcessingParameterRasterLayer,
                       QgsProcessingParameterVectorDestination,
                       QgsProcessingParameterFolderDestination,
                       QgsProcessingParameterString,
                       QgsProject,
                       QgsVectorLayer,
                       QgsRasterLayer
                       )
from qgis import processing

import os, sys, shutil
from osgeo import gdal, ogr, osr
import tensorflow
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from pathlib import Path
import pandas
import geopandas
from shapely import speedups
from shapely.geometry import shape, mapping
from shapely.ops import unary_union
import fiona
import itertools
from operator import itemgetter
import subprocess
import re

class ExampleProcessingAlgorithm(QgsProcessingAlgorithm):
    INPUT = 'INPUT'
    OUTPUT = 'OUTPUT'
    SHAPE = 'SHAPE'
    MODEL = 'MODEL'

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return ExampleProcessingAlgorithm()

    def name(self):
        return 'urbanarea'

    def displayName(self):
        return self.tr('UrbanArea')

    def group(self):
        return self.tr('UrbanArea')

    def groupId(self):
        return 'urbanarea'

    def shortHelpString(self):
        return self.tr("The script determines urbanized areas and generates a shapefile")

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT,
                self.tr('Input raster'), None, False
            )
        )

        self.addParameter(
            QgsProcessingParameterFolderDestination(
                self.OUTPUT,
                self.tr('Output Folder')
            )
        )

        self.addParameter(
            QgsProcessingParameterString(
                self.SHAPE,
                self.tr('Output shapefile name'),
                defaultValue = "urbanarea"
            )
        )

        self.addParameter(
            QgsProcessingParameterString(
                self.MODEL,
                self.tr('Model folder'),
                defaultValue = "D:/SCHOOL/Diploma/Urban_Area/3_Production/_Model/"
            )
        )

    def processAlgorithm(self, parameters, context, feedback):

        source = self.parameterAsSource(
            parameters,
            self.INPUT,
            context
        )
        
        output = self.parameterAsString(
            parameters,
            self.OUTPUT,
            context
        )

        sourcelink = self.parameterDefinition('INPUT').valueAsPythonString(parameters['INPUT'], context)
        head, tail = os.path.split(sourcelink)
        headn = re.sub("'", "", head)
        tailn = re.sub("'", "", tail)

        #Data----------------------------------------------
        #Main dynamic parameters
        #GeoTIFF EPSG:4326
        #path = "C:/Users/Stepan/Desktop/"
        #filename = "GEOTIFF2.tif"
        #shapefilepath = "C:/Users/Stepan/Desktop/out/"
        #shapefilename = "shapefile"

        path = headn + "/"
        filename = tailn
        shapefilepath = output+"/"
        shapefilename = parameters['SHAPE']

        #Main static parameters
        #modelpath = "C:/Users/Stepan/Desktop/ModelData/"
        modelpath = parameters['MODEL']
        modeljsonname= "modelunet1.json"
        modelwightsname = "modelunet1.h5"

        #Another static parameters
        num_classes = 2
        epsg = 4326
        filenamereproject = filename.split(".")[0] + "_reproject" + "." + filename.split(".")[1]

        #Technical folders
        techdirmain = path + "technicalfolderforwork"
        techdirin1 = techdirmain + "/" + "split" + "/"
        techdirin2 = techdirmain + "/" + "predict" + "/"
        techdirin3 = techdirmain + "/" + "shape" + "/"

        #Split folder parameters
        filenamesplit = "split_"
        img_width = 384
        img_height = 368

        #File search options
        filestartswith = "split"
        fileendswith = '.tif'
        #Data----------------------------------------------
        
        speedups.enable()
        
        #Functions-----------------------------------------
        #Сustom metrics function
        def dice_coef(y_true, y_pred):
            return (2. * K.sum(y_true * y_pred) + 1.) / (K.sum(y_true) + K.sum(y_pred) + 1.)

        #Сolor converting function 
        def index2color(index2):
            index = np.argmax(index2)
            color=[]
            if index == 0:
                color = [0, 0, 0] #background
            elif index == 1:
                color = [255, 0, 0] #target pixels
            return color

        #Сopying folder function
        def copytree(src, dst, symlinks=False, ignore=None):
            if not os.path.exists(dst):
                os.makedirs(dst)
            for item in os.listdir(src):
                s = os.path.join(src, item)
                d = os.path.join(dst, item)
                if os.path.isdir(s):
                    copytree(s, d, symlinks, ignore)
                else:
                    if not os.path.exists(d) or os.stat(s).st_mtime - os.stat(d).st_mtime > 1:
                        shutil.copy2(s, d)

        #Processing tile function
        def processimage(model, filenamein, filenameout, n_classes):
            img = image.load_img(os.path.join(techdirin2,filenamein), target_size=(img_height, img_width))
            xTest = image.img_to_array(img)
            predict = np.array(model.predict(xTest.reshape(1, img_height, img_width, 3)))
            pr = predict[0]
            pr = pr.reshape(-1, n_classes)
            pr1 = []

            for k in range(len(pr)):
                pr1.append(index2color(pr[k]))

            pr1 = np.array(pr1)
            pr1 = pr1.reshape(img_height, img_width, 3)  
            readyimg = Image.fromarray(pr1.astype('uint8'))
            readyimg.save(techdirin2+filenameout)

        #Сonverting to shapefile function    
        def converttoshapefile(image):
            name = str(image).split(".")[0]
            raster_path = techdirin2 + image
            shapefile_path = techdirin3 + name + ".shp"    
            
            sourceRaster = gdal.Open(raster_path)
            band = sourceRaster.GetRasterBand(1)
            outShapefile = "polygonized"
            driver = ogr.GetDriverByName("ESRI Shapefile")
            dest_srs = osr.SpatialReference()
            dest_srs.ImportFromEPSG(epsg)
            
            if os.path.exists(shapefile_path):
                driver.DeleteDataSource(shapefile_path)
            outDatasource = driver.CreateDataSource(shapefile_path)
            outLayer = outDatasource.CreateLayer("polygonized", srs=dest_srs)
            gdal.Polygonize( band, band, outLayer, 0, [], callback=None )
            outDatasource.Destroy()
            sourceRaster = None

        #Dissolving function
        #dissolve(shp1, shp2, ["FIELD1", "FIELD2", "FIELDN"])    
        def dissolve(input, output, fields):
            with fiona.open(input) as input:
                with fiona.open(output, 'w', **input.meta) as output:
                    grouper = itemgetter(*fields)
                    key = lambda k: grouper(k['properties'])
                    for k, group in itertools.groupby(sorted(input, key=key), key):
                        properties, geom = zip(*[(feature['properties'], shape(feature['geometry'])) for feature in group])
                        output.write({'geometry': mapping(unary_union(geom)), 'properties': properties[0]})    
        #Functions-----------------------------------------

        #1. Preparing
        if os.path.exists(techdirmain):
            shutil.rmtree(techdirmain)

        #2. Creating structure
        if not os.path.exists(techdirmain):
            os.mkdir(techdirmain)
            os.mkdir(techdirin1)
            #os.mkdir(techdirin2)
            os.mkdir(techdirin3)
        subprocess.check_call('gdalwarp ' + path + filename + ' ' + path + filenamereproject + ' -t_srs "+proj=longlat +ellps=WGS84"')

        #3. Splitting image into tiles
        ds = gdal.Open(path + filenamereproject)
        #ds = gdal.Open(path + filename)
        band = ds.GetRasterBand(1)
        xsize = band.XSize
        ysize = band.YSize
        for i in range(0, xsize, img_width):
            for j in range(0, ysize, img_height):
                com_string = "gdal_translate -of GTIFF -co TFW=yes -srcwin " + str(i)+ ", " + str(j) + ", " + str(img_width) + ", " + str(img_height) + " " + str(path) + str(filenamereproject) + " " + str(techdirin1) + str(filenamesplit) + str(i) + "_" + str(j) + ".tif"
                subprocess.check_call(com_string)

        #4. Copying split folder
        copytree(techdirin1, techdirin2, symlinks=False, ignore=None)
        
        #5. Model loading
        model_json_file = open(modelpath + modeljsonname,"r") 
        loaded_model_json = model_json_file.read()
        model_json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(modelpath + modelwightsname)
        loaded_model.compile(optimizer=Adam(lr=1e-3), loss='categorical_crossentropy', metrics=[dice_coef])
        
        #6. Tiles processing     
        filelist = os.listdir(techdirin2)
        for file in filelist:
            if file.startswith(filestartswith) and file.endswith(fileendswith):
                processimage(loaded_model, str(file), str(file), num_classes)
        
        #7. Converting to shapefiles
        newfilelist = os.listdir(techdirin2)
        for newfile in newfilelist:
            if newfile.startswith(filestartswith) and newfile.endswith(fileendswith):
                converttoshapefile(str(newfile))
        
        #8. Merging shapefiles
        speedups.disable()
        shapefolder = Path(techdirin3)
        shapefiles = shapefolder.glob("*.shp")
        gdf = pandas.concat([
            geopandas.read_file(shp)
            for shp in shapefiles
        ]).pipe(geopandas.GeoDataFrame)
        gdf.to_file(techdirmain +"/"+ shapefilename +".shp")

        outshp = geopandas.read_file(techdirmain +"/" + shapefilename +".shp")
        outshp["value"] = 1
        outshp.to_file(techdirmain +"/"+ shapefilename + ".shp")
            
        #9. Dissolving shapefile geometry
        dissolve(techdirmain +"/"+ shapefilename + ".shp", shapefilepath + shapefilename + ".shp", ["value"])

        #10. Removing technical folders
        shutil.rmtree(techdirmain)
        
        #11. Load layers
        vlayer = QgsVectorLayer(shapefilepath + shapefilename + ".shp", shapefilename, "ogr")
        root = QgsProject.instance().layerTreeRoot()
        QgsProject.instance().addMapLayer(vlayer, False)
        root.addLayer(vlayer)

        rlayer = QgsRasterLayer(path + filenamereproject, filenamereproject)
        QgsProject.instance().addMapLayer(rlayer, False)
        root.addLayer(rlayer)
       
        return {self.OUTPUT: output}
        
def Cancel(self):
    self.running = False