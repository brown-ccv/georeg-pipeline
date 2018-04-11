from PIL import Image
from tesserocr import PyTessBaseAPI, RIL
import os, glob
import time

#Reads an image and performs the OCR.

def ocr(file):
    print 'Reading: ' + file
    t1 = time.time()
    with PyTessBaseAPI() as api:
        t2 = time.time()
        print('API load time: ' + str(t2-t1) + ' s')
        ta = time.time()
        image = Image.open(file)
        tb = time.time()
        print('Open time: ' + str(tb-ta) + ' s')
        ta = time.time()
        api.SetImage(image)
        tb = time.time()
        print('SetImage time: ' + str(tb-ta) + ' s')
        ta = time.time()
        api.SetVariable("tessedit_char_whitelist", "()*,'&.;-0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
        tb = time.time()
        print('SetVariable time: ' + str(tb-ta) + ' s')
        ta = time.time()
        boxes = api.GetComponentImages(RIL.TEXTLINE, True)
        tb = time.time()
        print('GetComponentImages time: ' + str(tb-ta) + ' s')
        ta = time.time()
        outStr = api.GetUTF8Text()
        tb = time.time()
        print('OCR time: ' + str(tb-ta) + ' s')
        ta = time.time()
        outStr = outStr.encode('ascii', 'ignore')
        tb = time.time()
        print('ASCII encoding time: ' + str(tb-ta) + ' s')
    t2 = time.time()
    print('OCR runtime: ' + str(t2-t1) + ' s')
    return outStr

#Doesn't get used in the pipeline.  Just for playing around.
def read(folder):
    scans = folder
    os.chdir(scans)
    for file in sorted(glob.glob("*.png")):
        txt = ocr(file)
        print file + '- ocr'
        with open(file + 'ocr.txt', 'w') as output:
            output.write(txt)

#read('test/margins_fixed/no_ads/cropped/entry')
