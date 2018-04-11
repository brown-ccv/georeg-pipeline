import numpy as np
import cv2
import os
import glob
#from sklearn import svm
#from PIL import Image

#Determines if an entry is a header.

def templateFind(image):
	#Takes the Heading Arrow as template and returns its most likely location and score
	img = cv2.imread(image)
	template = cv2.imread("template.png")
	w,h,c = template.shape
	res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF)
	min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
	top_left = max_loc
	bottom_right = (top_left[0] + w, top_left[1] + h)
	return max_val, top_left, bottom_right


def process(image):
	#Load Images
	isHeader = False
	#If image is smaller than template, template cannot be present
	img = cv2.imread(image)
	w,h,c = img.shape
	if w < 32 or h < 32:
		isHeader = False
		#print "Too Small"
	elif img.size > 500000:
		isHeader = False
		#print "Too Big"
	else:
		val, topLeft, bottomRight = templateFind(image)
		#If score is high enough for a match
		if val > 30000000:
			# cv2.rectangle(img, topLeft, bottomRight, (0.255,255), 2)
			# cv2.imshow('Detected', img)
			isHeader = True
			#print 'Header score:'
			#print val
		else:
			isHeader = False
			#print 'Header score:'
			#print val

	return isHeader


# def findHead():
# 	groupA = []
# 	for filename in glob.glob("Headings/*.png"):
# 		img = Image.open(filename)
# 		w,h = img.size
# 		imc = img.crop((0,0,60,39))
# 		im2 = np.array(imc, dtype = float).flatten()
# 		groupA.append(im2)
# 	groupB = []
# 	for filename in glob.glob("NotHeadings/*.png"):
# 		img = Image.open(filename)
# 		imc = img.crop((0,0,60,39))
# 		im2 = np.array(imc, dtype = float).flatten()
# 		groupB.append(im2)


# 	A = len(groupA)
# 	B = len(groupB)
# 	print A, B
# 	targetA = np.zeros(A)
# 	targetB = np.ones(B)
# 	targs = np.append(targetA, targetB)
# 	data = np.append(groupA, groupB, axis =0)
# 	print data
# 	clf = svm.SVC()
# 	clf.fit(data, targs)
# 	trial = Image.open("1984_Page_786 (1)_8.png")
# 	trialc = trial.crop((0,0,60,39))
# 	trial2 = np.array(trialc, dtype = float).flatten()
# 	print clf.predict(trial2)


