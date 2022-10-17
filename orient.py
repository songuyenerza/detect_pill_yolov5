from lib2to3.pgen2 import grammar
from xmlrpc.client import getparser
import cv2
import pytesseract
import urllib
import numpy as np
import re
import os
import imutils
from PIL import Image

def mer_box(boxs):
    box0 = []
    box1 = []
    box2=[]
    box3=[]
    for box in boxs:
        box0.append(box[0])
        box1.append(box[1])
        box2.append(box[2])
        box3.append(box[3])
    x0 = min(box0)
    y0 = min(box1)
    x1 = max(box2)
    y1 = max(box3)
    bounding_box = (int(x0) , int (y0) , int(x1), int(y1))
    return bounding_box
def Average(lst):
    return sum(lst) / len(lst)
def orient(image):
    image = np.asarray(resp, dtype="uint8")
    #   image = cv2.imdecode(image, cv2.IMREAD_COLOR) # Initially decode as color
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.equalizeHist(gray)
    kernel = np.ones((7,7),np.float32)/25
    gray = cv2.filter2D(gray,-1,kernel)
    # cv2.imwrite("/media/anlab/0e731fe3-5959-4d40-8958-e9f6296b38cb/home/anlab/songuyen/rotate_test/croped5.png", gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] 
    boxes = pytesseract.image_to_boxes(gray)
    h, w, _ = resp.shape
    for b in boxes.splitlines():
        b = b.split(' ')
        if b != ['~', '0', '0', '0', '0', '0'] :
            if int(b[1]) < int(b[3]):
                img = resp[int(b[2]): int(b[4]), int(b[1]):int(b[3]) ]
        else:
            img = resp
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dilation = cv2.dilate(thresh, rect_kernel, iterations = 1)
    _, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    im2 = resp.copy()
    box = []
    agg_list = []
    w_list = []
    h_list = []
    for cnt in contours:
            area  = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            # print(x,y,w,h)
            if area > 5000 and area < 40000:
                if 160<x<850 and 160<y<850:
                    rect = cv2.minAreaRect(cnt)
                    (_,_),(w,h), agg = rect
                    w_list.append(w)
                    h_list.append(h)
                    agg_list.append(agg)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    # im2 = cv2.drawContours(im2,[box],0,(100,100,100),2)
    # box_mer = mer_box(box)
    # im3 = im2[box_mer[1]:box_mer[3], box_mer[0]: box_mer[2]]
    # print("__________________", box)
    if len(agg_list) != 0:
        best_angle = Average(agg_list)
        if Average(h_list) < Average(w_list):
            best_angle = best_angle +90
        (h, w) = im2.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
        im2 = cv2.warpAffine(im2, M, (w, h), flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE)
        # im2 = Image.fromarray(im2)
        # white = (128,128,128)
        # im2 = im2.rotate(best_angle, 0, expand = 1, fillcolor = white)
        # im2 = np.array(im2)

    return im2

if __name__ == '__main__':
     
    data_foler = "/home/anlab/Desktop/Songuyen/PIl_detection/box_output/"
    for path in os.listdir(data_foler):
            img_path = data_foler + path
            resp = cv2.imread(img_path)
            im2 = orient(resp)
            cv2.imwrite("/media/anlab/0e731fe3-5959-4d40-8958-e9f6296b38cb/home/anlab/songuyen/rotate_test/images/" + path, im2)

