import glob, os
import numpy as np
import cv2
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import math
import sklearn
from sklearn.cluster import MeanShift
import pickle as pkl
import time

#Chops columns into entries
#Script contains unused functions and needs heavy editing.

def lineIndent(img, h):
    avg = 0
    height, width = img.shape[:2]
    for x in range(0, width):
        if img[h, x] == 0:
            return x
    return -1

def findFirstIndent(image, h):
    height, width = image.shape[:2]
    for i in range(h, height):
        for x in range(0, width):
            if image[i, x] == 0:
                return i, x
    return -1, -1

def cropEntries(image, file):
    #t1 = time.time()
    croppedImages = []
    crop_points = []
    img = image.copy()
    height, width = img.shape[:2]
    sf = float(width)/float(2611)
    pad = int(4.0/float(height)*float(11675))
    histogram  = pd.Series([width - cv2.countNonZero(img[i,:]) for i in list(range(height))])
    #fig = plt.figure()
    #ax = histogram.plot()
    #ax.set_ylim([0,150])
    #ax.set_xlim([10500,11500])
    #plt.savefig('histogram' + file + '.pdf', bbox_inches='tight')
    #plt.close(fig)
    dip_df = histogram[histogram < sf*25].to_frame().rename(columns = {0:'count'})
    indices = np.array(dip_df.index.tolist()).reshape(-1,1)
    #pkl.dump(indices, open('indices.pkl', 'wb'))
    #t2 = time.time()
    #print('Prep time: ' + str(round(t2-t1, 2)) + ' s')
    #tf1 = time.time()
    ms = MeanShift(bandwidth = sf*50, bin_seeding=True)
    ms.fit(indices)
    dip_group = ms.predict(indices)
    #tf2 = time.time()
    #print('Fit time: ' + str(round(tf2-tf1, 2)) + ' s')
    #t1 = time.time()
    dip_df = dip_df.assign(group = dip_group)
    #cut_points = [0] + sorted(dip_df.groupby('group').apply(lambda x: int(np.mean(x.index))).tolist())[1:-1] + [height]
    cut_points = [0] + sorted(dip_df.groupby('group').idxmin()['count'].tolist())[1:-1] + [height]
    median_height = np.median([cut_points[i+1] - cut_points[i] for i in list(range(len(cut_points) - 1))])
    #t2 = time.time()
    #print('Sort time: ' + str(round(t2-t1, 2)) + ' s')
    for i in list(range(len(cut_points)-1)):
        start,end = cut_points[i],cut_points[i+1]
        if end-start > 1.5*median_height:
            #print(i+1)
            entry_hist = pd.DataFrame(data={'count':[float(width - cv2.countNonZero(img[j,:])) for j in list(range(start,end))]}, index=list(range(start,end)))
            entry_dip_df = entry_hist[entry_hist['count'] < sf*100]
            entry_indices = np.array(entry_dip_df.index.tolist()).reshape(-1,1)
            entry_ms = MeanShift(bandwidth = sf*50, bin_seeding=True)
            entry_ms.fit(entry_indices)
            entry_dip_group = entry_ms.predict(entry_indices)
            entry_dip_df = entry_dip_df.assign(entry_group = entry_dip_group)
            entry_cut_points = [start] + sorted(entry_dip_df.groupby('entry_group').idxmin()['count'].tolist())[1:-1] + [end]
            if len(entry_cut_points) > 2 :
                #print(entry_cut_points)
                #fig2 = plt.figure()
                #ax = entry_hist['count'].plot()
                #for xval in entry_cut_points:
                    #ax2 = plt.axvline(x = xval, linestyle = ':', color = 'r')
                #ax.set_ylim([0,300])
                #plt.savefig('entry_hist' + file + str(i+1) + '.pdf', bbox_inches='tight')
                #plt.close(fig2)
                for entry_i in list(range(len(entry_cut_points)-1)):
                    if histogram.iloc[entry_cut_points[entry_i]:entry_cut_points[entry_i+1]].sum() > sf*20:
                        adjusted_start = entry_cut_points[entry_i]
                        adjusted_end = entry_cut_points[entry_i+1]
                        while (histogram.iloc[adjusted_start] == 0) and (adjusted_start < (adjusted_end-1)):
                            adjusted_start += 1
                        while (histogram.iloc[adjusted_end-1] == 0) and ((adjusted_end-1) > adjusted_start):
                            adjusted_end -= 1
                        adjusted_start = max(adjusted_start - pad, 0)
                        adjusted_end = min(adjusted_end + pad, height)
                        croppedImages.append(img[adjusted_start:adjusted_end, 0:width])
                        crop_points.append([adjusted_start,adjusted_end])
            else:
                if entry_hist['count'].sum() > sf*20:
                    adjusted_start = start + 0
                    adjusted_end = end - 0
                    while (histogram.iloc[adjusted_start] == 0) and (adjusted_start < (adjusted_end-1)):
                        adjusted_start += 1
                    while (histogram.iloc[adjusted_end-1] == 0) and ((adjusted_end-1) > adjusted_start):
                        adjusted_end -= 1
                    adjusted_start = max(adjusted_start - pad, 0)
                    adjusted_end = min(adjusted_end + pad, height)
                    croppedImages.append(img[adjusted_start:adjusted_end, 0:width])
                    crop_points.append([adjusted_start,adjusted_end])
        else:
            if histogram.iloc[start:end].sum() > sf*20:
                adjusted_start = start + 0
                adjusted_end = end - 0
                while (histogram.iloc[adjusted_start] == 0) and (adjusted_start < (adjusted_end-1)):
                    adjusted_start += 1
                while (histogram.iloc[adjusted_end-1] == 0) and ((adjusted_end-1) > adjusted_start):
                    adjusted_end -= 1
                adjusted_start = max(adjusted_start - pad, 0)
                adjusted_end = min(adjusted_end + pad, height)
                croppedImages.append(img[adjusted_start:adjusted_end, 0:width])
                crop_points.append([adjusted_start,adjusted_end])
    #pkl.dump(crop_points, open('crop_points.' + file + '.pkl', 'wb'))
    return croppedImages, crop_points

def adjustMargins(top, bottom, image):
    """ Adjust top/bottom margins because image cleaning sometimes clips 
    top/bottom of letters 
    """
    t = 0
    b = 0
    while t <= 10 and blackPixel(image, top - t):
        t += 1
    while b <= 10 and blackPixel(image, bottom + b):
        b += 1
    return top - t, bottom + b

def blackPixel(image, y):
    height, width = image.shape[:2]
    for i in range(0, width):
        if image[y, i] == 0:
            return True
    return False


def inRange(x1, x2):
	if (x1 - x2) <= 15:
		insideRange = True
	elif x2 > x1:
		insideRange = False
	else:
		insideRange = False
	return insideRange
    

def cleanImage(image):
    inv = cv2.bitwise_not(image)
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4,2))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
    closing = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    return cv2.bitwise_not(opening)
    #return opening

def fineCrop(image):
    height, width = image.shape[:2]
    clean = cleanImage(image)
    sWidth = width/5
    cropLeft = sWidth
    cropRight = sWidth
    histogram  = pd.Series([height - cv2.countNonZero(clean[:,i]) for i in list(range(width))])
    def hasBlackPixel(i):
        if histogram[i] == 0:
            return False
        else:
            return True
    while cropLeft > 0 and hasBlackPixel(cropLeft):
        cropLeft -= 1
    while cropRight < width - 1 and hasBlackPixel(cropRight):
        cropRight += 1
    return image[0 : height, cropLeft  : cropRight]

def entryChop(folder):
    scans = folder
    nDirectory = 'entry'
    os.chdir(scans)
    if not os.path.exists(nDirectory):
        os.mkdir(nDirectory)
    crop_points_dict = {}
    for file in sorted(glob.glob("*.png")):
    	print 'Chopping: ' + file
        fileN = file[:-4]
        ext = file[-4:]
        #t1 = time.time()
        original = cv2.imread(file, 0)
        #t2 = time.time()
        #print('Read time: ' + str(round(t2-t1, 2)) + ' s')
        #t1 = time.time()
        original = fineCrop(original)
        #t2 = time.time()
        #print('fineCrop time: ' + str(round(t2-t1, 2)) + ' s')
        #t1 = time.time()
        original = cv2.copyMakeBorder(original,4,4,10,10, borderType= cv2.BORDER_CONSTANT, value=[255,255,255])
        #t2 = time.time()
        #print('copyMakeBorder time: ' + str(round(t2-t1, 2)) + ' s')
        #t1 = time.time()
        #clean = cleanImage(original)
        entries, points = cropEntries(original, file)
        #t2 = time.time()
        #print('Entry crop time: ' + str(round(t2-t1, 2)) + ' s')
        #print('Saving...')
        #t1 = time.time()
        i = 1
        for image in entries:
            w_image = cv2.copyMakeBorder(image,15,15,15,15, borderType= cv2.BORDER_CONSTANT, value=[255,255,255])
            cv2.imwrite(os.path.join(nDirectory, fileN + "_" + str(i) + ext), w_image)
            crop_points_dict[fileN + "_" + str(i) + ext] = points[i-1]
            i += 1
        #t2 = time.time()
        #print('Done in: ' + str(round(t2-t1, 3)) + ' s')
        #cv2.imwrite(os.path.join(nDirectory, file), clean)
        #cv2.imwrite(os.path.join(nDirectory, "og_" + file), original)
    pkl.dump(crop_points_dict, open('crop_points_dict.pkl', 'wb'))



