import glob, os, re
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import time
from multiprocessing import Pool
import shutil
import pickle as pkl

#Removes the ads

def naturalSort(String_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', String_)]

def cleanImage(image):
    inv = cv2.bitwise_not(image)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    closed = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, kernel)
    closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    return closed

def removeAds(im_bw, file, do_diagnostics, perimeter_cutoff):
    chop_file = file.rpartition("/")[2].partition('.jp2')[0]
    height,width = im_bw.shape[:2]
    sf = float(height + width)/float(13524 + 9475)
    cv2.rectangle(im_bw,(0,0),(width,height), (255,255,255), int(150*sf))
    im_bw_copy = im_bw.copy()
    if do_diagnostics:
        cv2.imwrite(os.path.join('no_ads', chop_file + '_bw_test.jpg'), im_bw)
    blank_image = np.zeros((height,width,3), np.uint8)
    white_image = 255.0 * np.ones((height,width), np.uint8)
    minContour = perimeter_cutoff * sf
    im2, contours, hierarchy = cv2.findContours(im_bw_copy,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        perimeter = cv2.arcLength(cnt, True)
        #if perimeter > minContour:
        x,y,w,h = cv2.boundingRect(cnt)
        contourA = (2*(w+h))**2 / max(perimeter,1)
        if contourA > minContour:
            #print(perimeter)
            cv2.drawContours(blank_image, [cnt], -1, (0,255,0), 3)
            #x,y,w,h = cv2.boundingRect(cnt)
            bottom = max([vertex[0][1] for vertex in cnt])
            top = min([vertex[0][1] for vertex in cnt])
            left = min([vertex[0][0] for vertex in cnt])
            right = max([vertex[0][0] for vertex in cnt])
            if (w > (width / 2)) and (bottom < (height * 0.75)):
                drawn_cnt = cv2.drawContours(255.0 * np.ones((height,width), np.uint8), [cnt], -1, (0,0,0), -1)
                y = bottom
                while (cv2.countNonZero(drawn_cnt[y,:]) > 0.75*width) and (y > 0):
                    y -= 1
                if y < top:
                    y = bottom
                cv2.rectangle(im_bw,(0,0),(width,y), (255,255,255), -1)
                cv2.drawContours(im_bw, [cnt], -1, (255,255,255), -1)
            if (w > (width / 2)) and (top > (height * 0.85)):
                drawn_cnt = cv2.drawContours(255.0 * np.ones((height,width), np.uint8), [cnt], -1, (0,0,0), -1)
                y = top
                while (cv2.countNonZero(drawn_cnt[y,:]) > 0.75*width) and (y < (height - 1)):
                    y += 1
                if y > bottom:
                    y = top
                cv2.rectangle(im_bw,(0,top),(width,height), (255,255,255), -1)
                cv2.drawContours(im_bw, [cnt], -1, (255,255,255), -1)
            #if (w > (width / 2)) and (top > (height * 0.90)):
                #cv2.rectangle(im_bw,(0,0),(width,bottom), (255,255,255), -1)
            if (w * h) < 0.9 * width * height:
                cv2.drawContours(white_image, [cnt], -1, (0,0,0), -1)
                if h > (0.15 * float(height)) and right < (0.25 * float(width)):
                    drawn_cnt = cv2.drawContours(255.0 * np.ones((height,width), np.uint8), [cnt], -1, (0,0,0), -1)
                    x = right
                    while (cv2.countNonZero(drawn_cnt[:,x]) > 0.85*height) and (x > 0):
                        x -= 1
                    #x = max(0,x-int(sf*15))
                    if x < left:
                        x = right
                    cv2.rectangle(im_bw,(0,0),(x,height), (255,255,255), -1)
                    cv2.drawContours(im_bw, [cnt], -1, (255,255,255), -1)
                elif h > (0.15 * float(height)) and left > (0.75 * float(width)):
                    drawn_cnt = cv2.drawContours(255.0 * np.ones((height,width), np.uint8), [cnt], -1, (0,0,0), -1)
                    x = left
                    while (cv2.countNonZero(drawn_cnt[:,x]) > 0.85*height) and (x < (width - 1)):
                        x += 1
                    #x = max(0,x+int(sf*15))
                    if x > right:
                        x = left
                    cv2.rectangle(im_bw,(x,0),(width,height), (255,255,255), -1)
                    cv2.drawContours(im_bw, [cnt], -1, (255,255,255), -1)
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
        cv2.imwrite(os.path.join('no_ads', chop_file + '_white.jpg'), white_image)
    if do_diagnostics:
        cv2.imwrite(os.path.join('no_ads', chop_file + '_contours.jpg'), blank_image)
    return im_bw

def noAds(image, area):
    #cleaned = cleanImage(image)
    noAds = removeAds(image, area)
    return noAds

def get_binary(file, threshold_dict, do_diagnostics, do_plots):
    nDirectory = 'no_ads'
    t1 = time.time()
    
    original = cv2.imread(file, 0)

    chop_file = file.partition('.jp2')[0]

    #uneq = cv2.imread(file, 0)
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    #original = clahe.apply(uneq)
    #original = cv2.equalizeHist(uneq)
    if do_diagnostics:
        cv2.imwrite(os.path.join(nDirectory, file.rpartition("/")[2].partition('.jp2')[0] + '_gray.jpg'), original)
    t2 = time.time()
    #print('Image read time: ' + str(round(t2-t1, 2)) + ' s')
    h, w = original.shape[:2]
    if (file + '.jp2') in threshold_dict.keys():
        threshold = threshold_dict[file + '.jp2']
    else:
        hist_raw,bins = np.histogram(original.ravel(),256,[0,256])
        if do_plots:
            rfig = plt.figure()
            plt.plot(hist_raw[:200])
            plt.xlabel('Grayscale Value')
            plt.ylabel('Pixel Count')
            rfig.savefig(chop_file + '.grayscale_raw.pdf', bbox_inches='tight')
            plt.close(rfig)
        hist = pd.Series(hist_raw).rolling(20, center=True).mean()
        #h_total = hist.sum()
        #h_cumulative = hist.cumsum()/h_total
        h_grad = pd.Series(np.gradient(hist, 20)).rolling(20, center=True).mean()
        if do_plots:
            fig = plt.figure()
            plt.plot(hist[:200])
            plt.xlabel('Grayscale Value')
            plt.ylabel('Pixel Count')
            fig.savefig(chop_file + '.grayscale_histogram.pdf', bbox_inches='tight')
            plt.close(fig)
            #cfig = plt.figure()
            #ax = h_cumulative.plot()
            #cfig.savefig(file.partition('.jp2')[0] + '.grayscale_cumulative.pdf', bbox_inches='tight')
            #plt.close(cfig)
        threshold = h_grad[(h_grad.index > 60) & (h_grad.index < 150)].idxmin()
        #print(threshold)
        g_cutoff = h_grad.max()/10
        while threshold<175 and h_grad.iloc[threshold] < g_cutoff:
            threshold += 1
        threshold -= 15
        #threshold = 145
        # print(threshold)
        os.system('echo "' + file.partition('.jp2')[0] + ',' + str(threshold) + '" >> threshold_used.csv')
        if do_plots:
            gfig = plt.figure()
            plt.plot(h_grad[:200])
            plt.plot([0,200],[g_cutoff,g_cutoff])
            plt.xlabel('Grayscale Value')
            plt.ylabel('Gradient of Pixel Count')
            gfig.savefig(chop_file + '.grayscale_grad.pdf', bbox_inches='tight')
            plt.close(gfig)
    original_padded = cv2.copyMakeBorder(original[:,:],10,10,10,10, cv2.BORDER_CONSTANT, value=255)
    im_bw = cv2.threshold(original_padded, threshold, 255, cv2.THRESH_BINARY)[1]
    return im_bw

def process_image(input_tuple):
    #separate tuple
    file, params = input_tuple

    t1 = time.time()
    im_bw = get_binary(file, params['threshold'], params['do_diagnostics'], params['do_plots'])
    t2 = time.time()
    #print('Binary conversion time: ' + str(round(t2-t1, 2)) + ' s')
    #cv2.imwrite(os.path.join('no_ads', file + 'bw_test.jpg'), im_bw)
    #im = cleanImage(original)
    t1 = time.time()
    removeAds(im_bw, file, params['do_diagnostics'], params['perimeter_cutoff'])
    t2 = time.time()
    #print('Ad removal time: ' + str(round(t2-t1, 2)) + ' s')

    # write output images
    t1 = time.time()
    chop_file = file.rpartition("/")[2]
    cv2.imwrite(os.path.join('no_ads', chop_file.partition('jp2')[0].partition('png')[0] + 'png'), im_bw)
    t2 = time.time()
    #print('Image write time: ' + str(round(t2-t1, 2)) + ' s')
    print file + '-no ads'

    return


def create_images_dir():
    if not os.path.exists('imgs'):
        os.mkdir('imgs')
    for filename in glob.glob('*.jp2'):
        shutil.copy(filename, 'imgs')
        os.remove(filename)
    for filename in glob.glob('*.pdf'):
        shutil.copy(filename, 'imgs')
        os.remove(filename)
    for filename in glob.glob('*.jpg'):
        shutil.copy(filename, 'imgs')
        os.remove(filename)

def rmAds(params):

    create_images_dir()

    # use hardcoded thresholds if they exist. 
    if os.path.isfile('hardcoded_thresholds.csv'):
        threshold_dict = pd.read_csv('hardcoded_thresholds.csv' , index_col=0).to_dict()['threshold']
        print('Using hardcoded thresholds:')
        print(threshold_dict)
    else:
        threshold_dict = {}
    os.system('echo ",threshold" > threshold_used.csv')

    # create no_ads dir.
    nDirectory = 'no_ads'
    if not os.path.exists(nDirectory):
        os.mkdir(nDirectory)

    # add threshold dict to params
    params['threshold'] = threshold_dict

    # parses single image
    if 'img_name' in params:
        process_image((params['img_name'] + ".jp2", params))
        return

    # create list of file/params tuples
    if params['only_hardcoded']:
        input_list = [(file_base + '.jp2', params) for file_base in threshold_dict.keys()]
    else:
        input_list = [(file, params) for file in sorted(glob.glob("./imgs/*.jp2") + glob.glob("*.png"), key=naturalSort)]
    
    # map input list to process_img
    if params['do_multiprocessing']:
        pool = Pool(params['pool_num'])
        pool.map(process_image, input_list)
    else:
        for input_tuple in input_list:
            process_image(input_tuple)


