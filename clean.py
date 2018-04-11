import glob, os
import numpy as np
import cv2





# listoflists = []

# for file in glob.glob("*.png"):
#   print file
#   l = []
#   img = cv2.imread(file, 0)
#   height, width = img.shape[:2]
#   # M = cv2.getRotationMatrix2D((height/2,width/2),0,1)
#   # img = cv2.warpAffine(im, M, (width, height))
#   left = 0
#   right = 250
#   sHeight = 400
#   l.append(left)
#   while right < width:
#       if(getAvg(img, right, sHeight) > 254):
#           im2 = img[0 : height, left : right + 10]
#           left = right
#           right += 250
#           width2 = im2.shape[1]
#           if width2 > 550 and width2 < 720:
#               l.append(float(right - 250)/width)
#       right += 2
#   if len(l) == 5:
#       listoflists.append(l)
#       print l

# cropPoints = [0, 0, 0, 0, 0]
# for l in listoflists:
#   for i in range(0, 5):
#       cropPoints[i] = l[i] + cropPoints[i]
# for i in range(0, 5):
#   cropPoints[i] = cropPoints[i] / len(listoflists)
# print cropPoints

# finalCrop = [0, 0.253, 0.483, 0.714, 0.945]
# for file in glob.glob("*.png"):


def cleanImage(image):
    inv = cv2.bitwise_not(image)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20,20))
    closed = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, kernel)
    closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    return closed

def deskewImage(original, image):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    non_zero_pixels = cv2.findNonZero(image)
    if non_zero_pixels is not None:
        minrect = cv2.minAreaRect(non_zero_pixels)
        angle = minrect[-1]
        box = cv2.boxPoints(minrect)
        box = np.int0(box)
        
        # color = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        # cv2.drawContours(color, [box], 0, (0, 0, 255), 4)
        # return color
        
        if angle < -45: 
            angle = -(90 + angle)
        else:
            angle = angle
        angle = angle/2.4
        print angle
        root_mat = cv2.getRotationMatrix2D(center, angle, 1)
        rotated = cv2.warpAffine(original, root_mat, (w, h), flags=cv2.INTER_CUBIC)
        return rotated
    return None


def straightenImage(folder):
    scans = folder
    nDirectory = 'deskew'
    os.chdir(scans)
    if not os.path.exists(nDirectory):
        os.mkdir(nDirectory)
    for file in sorted(glob.glob("*.png")):
        original = cv2.imread(file, 0)
        img = cleanImage(original)
        new = deskewImage(original, img)
        #new = img
        if new is not None:
            cv2.imwrite(os.path.join(nDirectory, file), new)


        print file