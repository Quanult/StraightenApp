# import imutils
import numpy as np
import cv2
import os
# import shutil
import argparse
from Straighten_karyotype.transparent_img_white import rewrite_image


def Blackground(img, size):  # add single chromosome onto black background
    h, w = img.shape[:2]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    roi = 255 - cv2.inRange(img, 1, 250)  # seperate single chromosome
    roi = img - roi
    h, w = roi.shape

    result = np.zeros(shape=(size, size), dtype=np.uint8)  # make background with shape =(size,size)
    i = int((size - h) / 2)
    j = int((size - w) / 2)
    result[i:i + h, j:j + w] = roi  # add single chromosome in the center of the above background
    return result

def Whiteground(img, size):  # add single chromosome onto black background
    h, w = img.shape[:2]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    result = np.ones(shape=(size, size), dtype=np.uint8) * 255  # make background with shape =(size,size)
    i = int((size - h) / 2)
    j = int((size - w) / 2)
    result[i:i + h, j:j + w] = img  # add single chromosome in the center of the above background
    return result

def img_denoising(img):
    im0 = cv2.blur(img, (1,1))
    ret,thresh1 = cv2.threshold(im0, 2,255,cv2.THRESH_BINARY)
    kernel = np.ones((5,5), np.uint8)
    bg= cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)
    mask = bg.clip(0,1)
    img_denoising = (img * mask)
    # img_color = cv2.cvtColor(img_denoising, cv2.COLOR_GRAY2BGR)
    return img_denoising
    
def extract_chromosome(img_path): #img: numpy
    img = cv2.imread(img_path)
    background = img.copy()
    black_chro = []
    white_chro = []

    h_img, w_img = img.shape[:2]

    # Threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to gray
    _, pre_thresh = cv2.threshold(gray, 252, 255, cv2.THRESH_BINARY)
    thresh = cv2.morphologyEx(pre_thresh, cv2.MORPH_CLOSE, np.ones(shape=(3, 3), dtype=np.uint8))
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST,
                                    cv2.CHAIN_APPROX_SIMPLE)  # use contour to extract single chromosome
   
    # Bounding box
    boxes = []  # boxes include box for each chromosome, box = (x,y,w,h)
    for i, contour in enumerate(contours):  # each contour is a single chromosome
        x, y, w, h = cv2.boundingRect(contour)

        if 10 < w and  h > 20:  # remove outliers,may need to edit 10
            boxes.append((x, y, w, h))      
            



    boxes = [box for box in boxes if box[0] != 0]
    row1 = [box for box in boxes if box[1] < h_img / 4]
    row2 = [box for box in boxes if box[1] > h_img / 4 and box[1] < h_img * 2 / 4]
    row3 = [box for box in boxes if box[1] > 2 * h_img / 4 and box[1] < h_img * 3 / 4]
    row4 = [box for box in boxes if box[1] > 3 * h_img / 4 and box[1] < h_img * 4 / 4]

    if len(row1) != 10:
        print(len(row1), len(row2), len(row3), len(row4))

    if len(row2) != 14:
        print(len(row1), len(row2), len(row3), len(row4))
        
    if len(row3) != 12:
        print(len(row1), len(row2), len(row3), len(row4))
        
    if len(row4) != 10 and len(row4) !=9:
        print(len(row1), len(row2), len(row3), len(row4))
        

    ## Step 4:
    rows = [row1, row2, row3, row4]
    start_class_each_row = [1, 6, 13, 19]
    end_class_each_row = [5, 12, 18, 23]
    Chromosome_ID = 0

    # class_label = []
    class_box = []
    for index_row,row_boxes in enumerate(rows): 

        boxes = sorted(row_boxes, key=lambda box: box[0])  # sort horizontally
        class_box.extend(boxes)
        start_class = start_class_each_row[index_row]
        class_chromosome = start_class
        label = [start_class]
        ID = []
        previous_box = boxes[0]

        for index_box,box in enumerate(boxes):
            Chromosome_ID += 1
            ID.append(Chromosome_ID)
            if index_box == 0:
                pass
            else:
                x_max_previous = previous_box[0] + previous_box[2]
                x_min_current = box[0]
                if abs(x_min_current - x_max_previous) < 15:
                    label.append(class_chromosome)
                    previous_box = box
                else:
                    class_chromosome = class_chromosome + 1
                    label.append(class_chromosome)
                    previous_box = box

        # class_label.extend(label)
        # Step 5: Save file
        for i, box in enumerate(boxes):  # process for each box
            x, y, w, h = box
            Chromosome = img[y:y + h, x:x + w]  # get single chromosome
            background[y:y + h, x:x + w] = 255
            black_image = Blackground(Chromosome, size=256)  # add black background for single chromosome
            black_image = img_denoising(black_image)
            black_chro.append(black_image)
            white_image = Whiteground(Chromosome, size= 256)
            # white_image = cv2.cvtColor(white_image, cv2.COLOR_GRAY2BGR)
            white_chro.append(white_image)
    return np.asarray(black_chro), np.asarray(white_chro), class_box, background # class_label

if __name__ == '__main1__':

    img = cv2.imread('./Karyotype_image/M6.A.nst.png')
    black_chro, white_chro, class_box, background = extract_chromosome(img)
    
    new_img = rewrite_image(white_chro, background, class_box)
    cv2.imwrite('new_img.jpg', new_img)

if __name__ == '__main__':
    # 23/97002149.3.tiff5774822883_45.jpg
    # 23/97002149.5.tiff5814832483_45.jpg
    pass
