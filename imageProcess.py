import time
mt1 = time.time()
import ads, margins, columnCrop, entryChop, parse
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
	img_p = d['image_process']
	all_params = [d['no_ads'], d['margins'],d['columns'],d['entries']]
	
	#checks if you only wish to parse a single image. 
	if img_p['single_image']:
		for p in all_params:
			p.update({'img_name': img_p['img_name']})
	
	print('Removing ads...')
	t1 = time.time()
	if img_p['ads']:
		ads.rmAds(all_params[0])
	t2 = time.time()
	print('Done in: ' + str(round(t2-t1, 2)) + ' s')
	print('Cropping margins...')
	t1 = time.time()
	if img_p['margins']:
		margins.marginCrop(all_params[1])
	t2 = time.time()
	print('Done in: ' + str(round(t2-t1, 2)) + ' s')
	print('Cropping columns...')
	t1 = time.time()
	if img_p['columns']:
		columnCrop.doCrop(all_params[2])
	t2 = time.time()
	print('Done in: ' + str(round(t2-t1, 2)) + ' s')
	print('Chopping entries...')
	t1 = time.time()
	if img_p['entries']:
		entryChop.entryChop(all_params[3])
	t2 = time.time()
	print('Done in: ' + str(round(t2-t1, 2)) + ' s')
	if img_p['parse']:
		os.chdir("..")
		parse.main(d)

mt2 = time.time()
print('Full runtime: ' + str(round(mt2-mt1, 2)) + ' s')