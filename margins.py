import glob, os
import numpy as np
import cv2

#Crops out the margins.

def getAvg(img, height):
    h, w = img.shape[:2]
    print('height: ' + str(height))
    avg = 0
    for i in range(0, w):
        minPixel = img[height, i]
        #for j in range(0, 5):
            #minPixel = min(minPixel, img[height - j, i])
        avg += minPixel
        #avg += img[height, i]
    return avg/w

def getAvgV(img, width):
    h, w = img.shape[:2]
    avg = 0
    for i in range(0, h):
        avg += img[i, width]
        # minPixel = img[i, width]
        # for j in range(0, 5):
        #     minPixel = min(minPixel, min(img[i, width - j], img[i, width + j]))
        # avg += minPixel
    return avg/h

def cropTop(image):
    h, w = image.shape[:2]
    y = 0
    while w - cv2.countNonZero(image[y,:]) < 25:
        y += 1
    return y

def cropBottom(image):
    h, w = image.shape[:2]
    y = h - 1
    while w - cv2.countNonZero(image[y,:]) < 25:
        y -= 1
    return y

def cropLeft(image):
    h, w = image.shape[:2]
    x = 50
    while h - cv2.countNonZero(image[:,x]) < 25:
        x += 1
    return x

def cropRight(image):
    h, w = image.shape[:2]
    x = w - 50
    while h - cv2.countNonZero(image[:,x]) < 25:
        x -= 1
    return x


def cropMargins(file):
    image = cv2.imread(file, 0)
    top = cropTop(image)
    bottom = cropBottom(image)
    left = cropLeft(image)
    right = cropRight(image)
    #print top, bottom, left, right
    return image[top-5 : bottom+5, left-5 : right+5]


def cleanImage(image):
    inv = cv2.bitwise_not(image)
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4,2))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,5))
    closing = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    return cv2.bitwise_not(opening)
    #return opening


def marginCrop(folder):
    scans = folder
    nDirectory = 'margins_fixed'
    os.chdir(scans)
    if not os.path.exists(nDirectory):
        os.mkdir(nDirectory)
    for file in sorted(glob.glob("*.png")):
        print file + '-margins cropped'
        cropped = cropMargins(file)
        #cropped = cleanImage(original)
        #cropped = original
        cv2.imwrite(os.path.join(nDirectory, file), cropped)
