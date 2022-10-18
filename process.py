import cv2
import os
import numpy as np
import cv2
import numpy as np
from scipy.ndimage import interpolation as inter

def correct_skew(img, image, delta=10, limit=180):
    def determine_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1, dtype=float)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
        return histogram, score

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] 
    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        histogram, score = determine_score(thresh, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    corrected = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, \
            borderMode=cv2.BORDER_REPLICATE)

    return best_angle, corrected
data_folder = "/home/anlab/Desktop/Songuyen/PIl_detection/box_output/"
for path in os.listdir(data_folder):
    
    img_path = data_folder + path
    img = cv2.imread(img_path)
    img_cp = img.copy()
    center = img.shape
    w =  center[1] * 0.7
    h =  center[0] *0.7
    x = center[1]/2 - w/2
    y = center[0]/2 - h/2
    img = img[int(y):int(y+h), int(x):int(x+w)]
    angle, corrected = correct_skew(img_cp,img)
    cv2.imwrite("/media/anlab/0e731fe3-5959-4d40-8958-e9f6296b38cb/home/anlab/songuyen/rotate_test/out_test/" +  path, corrected)
