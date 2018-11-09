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

	#checks if you only wish to parse a single image. 
	all_params = [d['no_ads'], d['margins'],d['columns'],d['entries']]
	if d['single_image']:
		for p in all_params:
			p.update({'img_name': d['img_name']})
		ads.rmAds(all_params[0])
		margins.marginCrop(all_params[1])
		columnCrop.doCrop(all_params[2])
		entryChop.entryChop(all_params[3])


	print('Removing ads...')
	t1 = time.time()
	ads.rmAds(d['no_ads'])
	t2 = time.time()
	print('Done in: ' + str(round(t2-t1, 2)) + ' s')
	print('Cropping margins...')
	t1 = time.time()
	margins.marginCrop(d['margins'])
	t2 = time.time()
	print('Done in: ' + str(round(t2-t1, 2)) + ' s')
	print('Cropping columns...')
	t1 = time.time()
	columnCrop.doCrop(d['columns'])
	t2 = time.time()
	print('Done in: ' + str(round(t2-t1, 2)) + ' s')
	print('Chopping entries...')
	t1 = time.time()
	entryChop.entryChop(d['entries'])
	t2 = time.time()
	print('Done in: ' + str(round(t2-t1, 2)) + ' s')

mt2 = time.time()
print('Full runtime: ' + str(round(mt2-mt1, 2)) + ' s')