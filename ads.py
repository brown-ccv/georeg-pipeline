import glob, os
import cv2
import numpy as np

#Removes the ads

def cleanImage(image):
    inv = cv2.bitwise_not(image)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    closed = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, kernel)
    closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    return closed

def removeAds(im_bw):
    pad = 0
    #cv2.imwrite(os.path.join('no_ads', 'bw_test2.png'), im_bw)
    # inverted = cv2.bitwise_not(inverted)
    # gray = cv2.cvtColor(inverted, cv2.COLOR_BGR2GRAY)
    #original = cv2.imread(original, 0)
    #if len(original.shape) == 3:
        #gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    #else:
        #gray = original
    im_bw_copy = im_bw.copy()
    #cv2.imwrite(os.path.join('no_ads', 'bw_test.png'), im_bw_copy)
    height,width = im_bw_copy.shape[:2]
    blank_image = np.zeros((height,width,3), np.uint8)
    sf = float(height + width)/float(13524 + 9475)
    minContour = 3000 * sf
    im2, contours, hierarchy = cv2.findContours(im_bw_copy,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        perimeter = cv2.arcLength(cnt, True)
        if perimeter > minContour:
        #x,y,w,h = cv2.boundingRect(cnt)
        #contourA = w*h
        #if contourA > minContour:
            #print(perimeter)
            cv2.drawContours(blank_image, [cnt], -1, (0,255,0), 3)
            x,y,w,h = cv2.boundingRect(cnt)
            bottom = max([vertex[0][1] for vertex in cnt])
            top = min([vertex[0][1] for vertex in cnt])
            left = min([vertex[0][0] for vertex in cnt])
            right = max([vertex[0][0] for vertex in cnt])
            if (w > (width / 2)) and (bottom < (height * 0.75)):
                cv2.rectangle(im_bw,(0,0),(width,bottom), (255,255,255), -1)
            if (w * h) < 0.9 * width * height:
                if h > (0.1 * float(height)) and right < (0.25 * float(width)):
                    cv2.rectangle(im_bw,(0,0),(right,height), (255,255,255), -1)
                elif h > (0.1 * float(height)) and left > (0.75 * float(width)):
                    cv2.rectangle(im_bw,(left,0),(width,height), (255,255,255), -1)
                elif ((height - bottom) < 11) or (top < 11):
                    o_vertices = cv2.approxPolyDP(cnt, 0.005*perimeter, True)
                    approx = cv2.convexHull(o_vertices, clockwise=True)
                    cv2.drawContours(im_bw, [approx], -1, (255, 255, 255), -1)
                    cv2.drawContours(im_bw, [approx], -1, (255, 255, 255), int(5 * sf))
                    cv2.drawContours(blank_image, [approx], -1, (255, 255, 255), -1)
                else:
                    cv2.rectangle(im_bw,(x-pad,y-pad),(x+w+pad,y+h+pad), (255,255,255), -1)
                    cv2.rectangle(blank_image,(x-pad,y-pad),(x+w+pad,y+h+pad), (255,255,255), -1)
    #cv2.imwrite(os.path.join(nDirectory, file), original)
    #cv2.imwrite(os.path.join('no_ads', 'bw_test3.png'), im_bw)
    #cv2.imwrite(os.path.join('no_ads', 'contours.png'), blank_image)
    return im_bw

def noAds(image, area):
    #cleaned = cleanImage(image)
    noAds = removeAds(image, area)
    return noAds

def rmAds(folder):
    scans = folder
    nDirectory = 'no_ads'
    os.chdir(scans)
    if not os.path.exists(nDirectory):
        os.mkdir(nDirectory)
    for file in sorted(glob.glob("*.png")):
        original = cv2.imread(file, 0)
        h, w = original.shape[:2]
        threshold = 127
        original_padded = cv2.copyMakeBorder(original[:,0:w-50],10,10,10,10, cv2.BORDER_CONSTANT, value=255)
        im_bw = cv2.threshold(original_padded, threshold, 255, cv2.THRESH_BINARY)[1]
        #cv2.imwrite(os.path.join('no_ads', 'bw_test.png'), im_bw)
        #im = cleanImage(original)
        removeAds(im_bw)
        cv2.imwrite(os.path.join(nDirectory, file), im_bw)
        print file + '-no ads'



# if not os.path.exists(nDirectory):
#   os.mkdir(nDirectory)
# for file in sorted(glob.glob("*.png")):
#   print(file)
#   im = cv2.imread(file)
#   gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
#   im2, contours, hierarchy = cv2.findContours(gray,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
#   for cnt in contours:
#       #if cv2.arcLength(cnt, True) > minContour:
#       if cv2.contourArea(cnt) > minContour:
#           x,y,w,h = cv2.boundingRect(cnt)
#           roi=im[y:y+h,x:x+w]
#           cv2.imwrite(os.path.join(nDirectory, file), roi)
#           cv2.rectangle(im,(x,y),(x+w,y+h), (255,255,255),-1)
    