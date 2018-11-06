import time
mt1 = time.time()
import ads, margins, columnCrop, entryChop
import os
import sys
import json

if not sys.argv[1]:
	raise Exception('You need to state the input parameters you are running.')
inputParams = str(sys.argv[1])

#This is the driver script for all the image processing.

if __name__ == '__main__':
	with open(inputParams) as json_data:
		d = json.load(json_data)
	os.chdir(d['year_folder'])
	print('Removing ads...')
	t1 = time.time()
	ads.rmAds(d['no_ads'])
	#os.chdir(dir_dir)
	t2 = time.time()
	print('Done in: ' + str(round(t2-t1, 2)) + ' s')
	# print('Cropping margins...')
	# t1 = time.time()
	# margins.marginCrop('no_ads')
	# #os.chdir('no_ads')
	# t2 = time.time()
	# print('Done in: ' + str(round(t2-t1, 2)) + ' s')
	# print('Cropping columns...')
	# t1 = time.time()
	# columnCrop.doCrop('margins_fixed')
	# #os.chdir('margins_fixed')
	# t2 = time.time()
	# print('Done in: ' + str(round(t2-t1, 2)) + ' s')
	# print('Chopping entries...')
	# t1 = time.time()
	# entryChop.entryChop('cropped')
	# t2 = time.time()
	# print('Done in: ' + str(round(t2-t1, 2)) + ' s')

mt2 = time.time()
print('Full runtime: ' + str(round(mt2-mt1, 2)) + ' s')