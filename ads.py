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
            if (w > (width / 2)) and (top > (height * 0.85)):
                cv2.rectangle(im_bw,(0,top),(width,height), (255,255,255), -1)
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
    if do_diagnostics:
        cv2.imwrite(os.path.join('no_ads', file.partition('.png')[0] + '.contours.jpg'), blank_image)
    return im_bw


def removeAds_old(im_bw, file):
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
            if (w > (width / 2)) and (top > (height * 0.85)):
                cv2.rectangle(im_bw,(0,top),(width,height), (255,255,255), -1)
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
    if do_diagnostics:
        cv2.imwrite(os.path.join('no_ads', file.partition('.png')[0] + '.contours.jpg'), blank_image)
    return im_bw

def noAds(image, area):
    #cleaned = cleanImage(image)
    noAds = removeAds(image, area)
    return noAds

def get_binary(file):
    nDirectory = 'no_ads'
    do_plots = True
    t1 = time.time()
    original = cv2.imread(file, 0)
    #uneq = cv2.imread(file, 0)
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    #original = clahe.apply(uneq)
    #original = cv2.equalizeHist(uneq)
    cv2.imwrite(os.path.join(nDirectory, file.partition('jp2')[0] + '_eq.jp2'), original)
    t2 = time.time()
    #print('Image read time: ' + str(round(t2-t1, 2)) + ' s')
    h, w = original.shape[:2]
    hist_raw,bins = np.histogram(original.ravel(),256,[0,256])
    if do_plots:
        rfig = plt.figure()
        plt.plot(hist_raw[:175])
        plt.xlabel('Grayscale Value')
        plt.ylabel('Pixel Count')
        rfig.savefig(file.partition('.jp2')[0] + '.grayscale_raw.pdf', bbox_inches='tight')
        plt.close(rfig)
    hist = pd.Series(hist_raw[:200]).rolling(20).mean().rolling(20).mean()
    h_total = hist.sum()
    h_cumulative = hist.cumsum()/h_total
    h_grad = pd.Series(np.gradient(hist, 30)).rolling(30).mean()
    if do_plots:
        fig = plt.figure()
        plt.plot(hist[:175])
        plt.xlabel('Grayscale Value')
        plt.ylabel('Pixel Count')
        fig.savefig(file.partition('.jp2')[0] + '.grayscale_histogram.pdf', bbox_inches='tight')
        plt.close(fig)
        #cfig = plt.figure()
        #ax = h_cumulative.plot()
        #cfig.savefig(file.partition('.jp2')[0] + '.grayscale_cumulative.pdf', bbox_inches='tight')
        #plt.close(cfig)
    threshold = h_grad[(h_grad.index > 60) & (h_grad.index < 150)].idxmin()
    g_cutoff = h_grad.min()*0.09 
    while threshold<175 and h_grad.iloc[threshold] < g_cutoff:
        threshold += 1
    threshold += 5
    print(threshold)
    threshold = 30
    if do_plots:
        gfig = plt.figure()
        plt.plot(h_grad[:175])
        plt.plot([0,175],[g_cutoff,g_cutoff])
        plt.xlabel('Grayscale Value')
        plt.ylabel('Gradient of Pixel Count')
        gfig.savefig(file.partition('.jp2')[0] + '.grayscale_grad.pdf', bbox_inches='tight')
        plt.close(gfig)
    original_padded = cv2.copyMakeBorder(original[:,:],10,10,10,10, cv2.BORDER_CONSTANT, value=255)
    im_bw = cv2.threshold(original_padded, threshold, 255, cv2.THRESH_BINARY)[1]
    return im_bw

def process_image(file):
    nDirectory = 'no_ads'
    t1 = time.time()
    im_bw = get_binary(file)
    t2 = time.time()
    #print('Binary conversion time: ' + str(round(t2-t1, 2)) + ' s')
    #cv2.imwrite(os.path.join('no_ads', file + 'bw_test.jpg'), im_bw)
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
    do_multiprocessing = False
    if do_multiprocessing:
        pool = Pool(4)
        pool.map(process_image, sorted(glob.glob("*.jp2"), key=naturalSort))
    else:
        for file in sorted(glob.glob("*.jp2"), key=naturalSort):
            process_image(file)


