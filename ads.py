import glob, os, re
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import time
from multiprocessing import Pool

#Removes the ads

def naturalSort(String_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', String_)]

def cleanImage(image):
    inv = cv2.bitwise_not(image)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    closed = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, kernel)
    closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    return closed

def removeAds(im_bw, file):
    #pad = 0
    #cv2.imwrite(os.path.join('no_ads', 'bw_test2.png'), im_bw)
    # inverted = cv2.bitwise_not(inverted)
    # gray = cv2.cvtColor(inverted, cv2.COLOR_BGR2GRAY)
    #original = cv2.imread(original, 0)
    #if len(original.shape) == 3:
        #gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    #else:
        #gray = original
    do_diagnostics = False
    height,width = im_bw.shape[:2]
    cv2.rectangle(im_bw,(0,0),(width,height), (255,255,255), 100)
    im_bw_copy = im_bw.copy()
    if do_diagnostics:
        cv2.imwrite(os.path.join('no_ads', file.partition('.png')[0] + '.bw_test.jpg'), im_bw)
    blank_image = np.zeros((height,width,3), np.uint8)
    white_image = 255.0 * np.ones((height,width), np.uint8)
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
            #if (w > (width / 2)) and (top > (height * 0.90)):
                #cv2.rectangle(im_bw,(0,0),(width,bottom), (255,255,255), -1)
            if (w * h) < 0.9 * width * height:
                cv2.drawContours(white_image, [cnt], -1, (0,0,0), -1)
                if h > (0.1 * float(height)) and right < (0.25 * float(width)):
                    cv2.rectangle(im_bw,(0,0),(right,height), (255,255,255), -1)
                elif h > (0.1 * float(height)) and left > (0.75 * float(width)):
                    cv2.rectangle(im_bw,(left,0),(width,height), (255,255,255), -1)
                #elif ((height - bottom) < 11) or (top < 11):
                    #o_vertices = cv2.approxPolyDP(cnt, 0.005*perimeter, True)
                    #approx = cv2.convexHull(o_vertices, clockwise=True)
                    #cv2.drawContours(im_bw, [approx], -1, (255, 255, 255), -1)
                    #cv2.drawContours(im_bw, [approx], -1, (255, 255, 255), int(5 * sf))
                    #cv2.drawContours(blank_image, [approx], -1, (255, 255, 255), -1)
                else:
                    cv2.drawContours(im_bw, [cnt], -1, (255,255,255), -1)
                    #cv2.rectangle(im_bw,(x-pad,y-pad),(x+w+pad,y+h+pad), (255,255,255), -1)
                    #cv2.rectangle(blank_image,(x-pad,y-pad),(x+w+pad,y+h+pad), (255,255,255), -1)
    #cv2.imwrite(os.path.join(nDirectory, file), original)
    if do_diagnostics:
        cv2.imwrite(os.path.join('no_ads', file.partition('.png')[0] + '.white.jpg'), white_image)
    yi = height-50
    while cv2.countNonZero(white_image[yi,:]) < (0.7*float(width)) and yi > (0.95*float(height)):
        yi -= 1
    #yi -= int(height/300)
    if do_diagnostics:
        print('Bottom cutoff: ' + str(100.0*float(yi)/float(height)))
    cv2.rectangle(im_bw,(0,yi),(width,height), (255,255,255), -1)
    if do_diagnostics:
        cv2.imwrite(os.path.join('no_ads', file.partition('.png')[0] + '.contours.jpg'), blank_image)
    return im_bw

def noAds(image, area):
    #cleaned = cleanImage(image)
    noAds = removeAds(image, area)
    return noAds

def get_binary(file):
    do_plots = False
    t1 = time.time()
    original = cv2.imread(file, 0)
    t2 = time.time()
    #print('Image read time: ' + str(round(t2-t1, 2)) + ' s')
    h, w = original.shape[:2]
    hist_raw,bins = np.histogram(original.ravel(),256,[0,256])
    hist = pd.Series(hist_raw[:200]).rolling(10).mean()
    h_total = hist.sum()
    h_cumulative = hist.cumsum()/h_total
    h_grad = pd.Series(np.gradient(hist, 5)).rolling(5).mean()
    if do_plots:
        fig = plt.figure()
        plt.plot(hist)
        fig.savefig(file.partition('.png')[0] + '.grayscale_histogram.pdf', bbox_inches='tight')
        plt.close(fig)
        cfig = plt.figure()
        ax = h_cumulative.plot()
        cfig.savefig(file.partition('.png')[0] + '.grayscale_cumulative.pdf', bbox_inches='tight')
        plt.close(cfig)
    threshold = h_grad[(h_grad.index > 60) & (h_grad.index < 150)].idxmin()
    g_cutoff = h_grad.min()*0.15
    while threshold<175 and h_grad.iloc[threshold] < g_cutoff:
        threshold += 1
    #print(threshold)
    if do_plots:
        gfig = plt.figure()
        plt.plot(h_grad)
        plt.plot([0,200],[g_cutoff,g_cutoff])
        gfig.savefig(file.partition('.png')[0] + '.grayscale_grad.pdf', bbox_inches='tight')
        plt.close(gfig)
    original_padded = cv2.copyMakeBorder(original[:,0:w-50],10,10,10,10, cv2.BORDER_CONSTANT, value=255)
    im_bw = cv2.threshold(original_padded, threshold, 255, cv2.THRESH_BINARY)[1]
    return im_bw

def process_image(file):
    nDirectory = 'no_ads'
    t1 = time.time()
    im_bw = get_binary(file)
    t2 = time.time()
    #print('Binary conversion time: ' + str(round(t2-t1, 2)) + ' s')
    #cv2.imwrite(os.path.join('no_ads', 'bw_test.png'), im_bw)
    #im = cleanImage(original)
    t1 = time.time()
    removeAds(im_bw, file)
    t2 = time.time()
    #print('Ad removal time: ' + str(round(t2-t1, 2)) + ' s')
    t1 = time.time()
    cv2.imwrite(os.path.join(nDirectory, file), im_bw)
    t2 = time.time()
    #print('Image write time: ' + str(round(t2-t1, 2)) + ' s')
    print file + '-no ads'
    return

def rmAds(folder):
    scans = folder
    nDirectory = 'no_ads'
    os.chdir(scans)
    if not os.path.exists(nDirectory):
        os.mkdir(nDirectory)
    pool = Pool(4)
    pool.map(process_image, sorted(glob.glob("*.png"), key=naturalSort))




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
    