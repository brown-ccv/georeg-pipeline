import ads, margins, columnCrop, entryChop, parse
import os
import sys
import json
import time
import glob, re


folders = ["cd1936","cd1937","cd1938","cd1940","cd1941","cd1943","cd1944","cd1945","cd1946","cd1948","cd1950","cd1952","cd1954","cd1956","cd1959","cd1960","cd1962","cd1964","cd1966","cd1968","cd1970","cd1972","cd1976","cd1978","cd1980","cd1985","cd1990"]

if not sys.argv[1]:
	raise Exception('You need to pass in the parameter files.')
inputParams = str(sys.argv[1])
#foldersList = str(sys.argv[2])

# takes in list of repeats, returns list of elements that didn't repeat.
def occured_once(lst):
	tracker = dict()
	for page in lst:
		if page in tracker:
			tracker[page] += 1
		else:
			tracker[page] = 1

	uncut = []
	for key in tracker.keys():
		if tracker[key] == 1:
			uncut.append(key)
	return uncut

# sorts by natural number system, ie. 23 comes after 4. 
def naturalSort(String_): 
	return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', String_)]

# prints the name of each file that wasn't column cropped properly
# by figuring out which files only appear once in the columns folder, 
# implying the page wasn't cut into three columns properly.
def column_error_tracking(year_dir, d):
	os.chdir("..")
	os.chdir(year_dir)
	x = sorted(glob.glob(os.getcwd() + "/columns/*.png"), key=naturalSort)
	for i in range(len(x)):
	 	x[i] = x[i].split("/")[-1].split()[0]
	singles = occured_once(x)
	for pg in singles:
		print(pg)

# same as image_process code.
def run_year(year_dir, d):
	#print(os.getcwd())
	#os.chdir("..")
	print(os.getcwd())
	os.chdir(year_dir)
	print(os.getcwd())
	d['year_folder'] = year_dir
	img_p = d['image_process']
	all_params = [d['no_ads'], d['margins'],d['columns'],d['entries']]
	
	#checks if you only wish to parse a single image. 
	if img_p['single_image']:
		for p in all_params:
			p.update({'img_name': img_p['img_name']})
	
	if img_p['ads']: ads.rmAds(all_params[0])
	if img_p['margins']: margins.marginCrop(all_params[1])
	if img_p['columns']: columnCrop.doCrop(all_params[2])
	if img_p['entries']: entryChop.entryChop(all_params[3])
	os.chdir("..")
	if img_p['parse']: parse.main(d)


if __name__ == '__main__':

	# opens the inputparams. 
	# TODO: somehow make it s.t. you can input different input params for different year folders?
	with open(inputParams) as json_data:
		d = json.load(json_data)

	# runs the code on each year, starting at the start folder.
	start = "cd1936"
	#os.chdir(start)

	# runs the actual image-process/parse code. 
	for i in range(folders.index(start), len(folders)):
		run_year(folders[i], d)

	# prints all pages with column errors. 
	for i in folders:
		column_error_tracking(i, d)

	os.chdir("..")

	# how many total images are there?
	all_files = sorted(glob.glob("*/imgs/*.jp2"), key=naturalSort)
	print(len(all_files))

	# how many column images do we have?
	all_files = sorted(glob.glob("*/columns/*.png"), key=naturalSort)
	print(len(all_files) / 3)








