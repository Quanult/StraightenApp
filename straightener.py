from ast import Str
import imagej
from scyjava import jimport
from jpype import JArray, JInt
import cv2
import numpy as np


def Straighten_img(ij, img_path, xpoints, ypoints, line):

    imp = ij.IJ.openImage(img_path)
    #################################
    # xpoints = [124,126,131,137,131,128,121,114]
    xpoints_int = JArray(JInt)(xpoints)
    # ypoints = [44,63,105,128,148,172,194,206]
    ypoints_int = JArray(JInt)(ypoints)

    ###################################
    straightener = jimport('ij.plugin.Straightener')
    polyRoi = jimport('ij.gui.PolygonRoi')
    roi = jimport('ij.gui.Roi')

    new_polyRoi = polyRoi(xpoints_int,ypoints_int,len(xpoints), int(roi.POLYLINE))
    imp.setRoi(new_polyRoi)

    straightened_img = straightener().straightenLine(imp,line)
    straightened_img = ij.py.from_java(straightened_img.getFloatArray())
    straightened_img = np.asarray(straightened_img)
    straightened_img.astype(int)

    blank_img = np.zeros_like(cv2.imread(img_path))
    x_start = round((blank_img.shape[0] - straightened_img.shape[0])/2)
    y_start = round((blank_img.shape[1] - straightened_img.shape[1])/2)
    for channel in range(blank_img.shape[2]):
        blank_img[x_start:x_start+straightened_img.shape[0],y_start:y_start+straightened_img.shape[1],channel]= straightened_img
    
    return blank_img

if __name__ == "__main__":
    img_path = "D:/AI lab/joint_detection/data/data_classify/data_train/1/9700TEST.6.tiff133328134_1.jpg"
    xpoints = [124,126,131,137,131,128,121,114]
    ypoints = [44,63,105,128,148,172,194,206]
    line = 30
    str_img = Straighten_img(img_path, xpoints, ypoints, line)

    cv2.imshow('vvv', str_img)
    cv2.waitKey(0)
