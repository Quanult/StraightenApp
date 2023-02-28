import imagej
from scyjava import jimport
from jpype import JArray, JInt
import cv2
import numpy as np
import os

def left_right_flip(image):
    
    flipped_image = np.flip(image, axis=1)  
    return flipped_image

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
    
    blank_img = left_right_flip(blank_img)
    # print(blank_img.shape)
    return blank_img

def Straighten_img_white(ij, img, xpoints, ypoints, line):
    path = './buffer.jpg'
    cv2.imwrite(path, img)
    imp = ij.IJ.openImage(path)
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

    blank_img = np.ones_like(img) * 255
    x_start = round((blank_img.shape[0] - straightened_img.shape[0])/2)
    y_start = round((blank_img.shape[1] - straightened_img.shape[1])/2)
    blank_img[x_start:x_start+straightened_img.shape[0],y_start:y_start+straightened_img.shape[1]]= straightened_img
    
    os.remove(path)
    blank_img = left_right_flip(blank_img)
    # print(blank_img.shape)
    return blank_img

def Straighten_img_black(ij, img, xpoints, ypoints, line):
    path = './buffer.jpg'
    cv2.imwrite(path, img)
    imp = ij.IJ.openImage(path)
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

    blank_img = np.zeros_like(img)
    x_start = round((blank_img.shape[0] - straightened_img.shape[0])/2)
    y_start = round((blank_img.shape[1] - straightened_img.shape[1])/2)
    blank_img[x_start:x_start+straightened_img.shape[0],y_start:y_start+straightened_img.shape[1]]= straightened_img
    
    os.remove(path)
    blank_img = left_right_flip(blank_img)
    # print(blank_img.shape)
    return blank_img


if __name__ == '__main__':
    image = cv2.imread('test_straigh.jpg')
    thresh = cv2.threshold(image, 255, 255, cv2.THRESH_BINARY)[1]
    cv2.imwrite('test.jpg', thresh)
    

    
