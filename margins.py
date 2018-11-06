import glob, os, re
import numpy as np
import cv2
from multiprocessing import Pool

def naturalSort(String_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', String_)]

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

def cropTop(image, p_cutoff):
    h, w = image.shape[:2]
    sf = float(h)/13524.0
    sfw = float(w)/9475.0
    y = 0
    while w - cv2.countNonZero(image[y,:]) < sfw*p_cutoff/2.0:
        y += 1
    print(y)
    if cv2.countNonZero(image[y+int(sf*50):y+int(sf*100),:]) < float(int(sf*50.0))*sfw*p_cutoff/2.0:
        y += int(sf*50)
        while w - cv2.countNonZero(image[y,:]) < sfw*p_cutoff/2.0:
            y += 1
    return y - int(sf*25)

def cropBottom(image, p_cutoff):
    h, w = image.shape[:2]
    sf = float(h)/13524.0
    sfw = float(w)/9475.0
    y = h - 1
    while w - cv2.countNonZero(image[y,:]) < sfw*p_cutoff/2.0:
        y -= 1
    if cv2.countNonZero(image[y-int(sf*100):y-int(sf*50),:]) < float(int(sf*50.0))*sfw*p_cutoff/2.0:
        y -= int(sf*50)
        while w - cv2.countNonZero(image[y,:]) < sfw*p_cutoff/2.0:
            y -= 1
    return y + int(sf*35)

def cropLeft(image, p_cutoff):
    h, w = image.shape[:2]
    #print(h,w)
    sf = float(h)/13524.0
    sfw = float(w)/9475.0
    #print(sf,sfw)
    x = 0
    while h - cv2.countNonZero(image[:,x]) < sf*p_cutoff:
        x += 1
    if cv2.countNonZero(image[:,x+int(sfw*50):x+int(sfw*100)]) < float(int(sfw*50.0))*sf*p_cutoff:
        x += int(sfw*50)
        print('Entered 50 plus loop.')
        print(h - cv2.countNonZero(image[:,x+int(sfw*50):x+int(sfw*100)]))
        print(float(int(sfw*50.0))*sf*p_cutoff)
        while h - cv2.countNonZero(image[:,x]) < sf*p_cutoff:
            x += 1
    return x - int(50*sfw)

def cropRight(image, p_cutoff):
    h, w = image.shape[:2]
    sf = float(h)/13524.0
    sfw = float(w)/9475.0
    x = w - 1
    #while x > 7000 and (h - cv2.countNonZero(image[:,x]) < sf*50.0 or cv2.countNonZero(image[:,x-int(sfw*150):x-int(sfw*50)]) < sfw*100.0*sf*50.0):
        #x -= 1
    while h - cv2.countNonZero(image[:,x]) < sf*p_cutoff:
        x -= 1
    if cv2.countNonZero(image[:,x-int(sfw*200):x-int(sfw*100)]) < float(int(sfw*100.0))*sf*p_cutoff:
        x -= int(100*sfw)
        while h - cv2.countNonZero(image[:,x]) < sf*p_cutoff:
            x -= 1
    return x + int(100*sfw)


def cropMargins(filename_param_tuple):
    file, params = filename_param_tuple
    p_cutoff = params['p_cutoff']
    print file + '-margins cropped'
    image = cv2.imread(file, 0)
    top = cropTop(image, p_cutoff)
    bottom = cropBottom(image, p_cutoff)
    left = cropLeft(image, p_cutoff)
    right = cropRight(image, p_cutoff)
    h, w = image.shape[:2]
    sf = float(h)/13524.0
    sfw = float(w)/9475.0
    cropped = image[top-int(sf*5.0) : bottom+int(sf*5.0), left-int(sfw*5.0) : right+int(sfw*10.0)]
    nDirectory = 'margins'
    filename = file.split("/")[-1]
    print("Written to " + os.path.join(nDirectory, filename))
    cv2.imwrite(os.path.join(nDirectory, filename), cropped)
    #print top, bottom, left, right
    return

def cleanImage(image):
    inv = cv2.bitwise_not(image)
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4,2))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,5))
    closing = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    return cv2.bitwise_not(opening)
    #return opening


def marginCrop(params):

    #make margins dir.
    nDirectory = 'margins'
    if not os.path.exists(nDirectory):
        os.mkdir(nDirectory)

    #create list of image/param tuples.
    x = sorted(glob.glob(os.getcwd() + "/no_ads/*.png"), key=naturalSort)
    params_and_files = [(i, params) for i in x]

    # map params_and_files to cropMargins.
    if params['do_multiprocessing']:
        pool = Pool(params['pool_num'])
        pool.map(cropMargins, params_and_files)
    else:
        for filename_param_tuple in params_and_files:
            cropMargins(filename_param_tuple)
    

