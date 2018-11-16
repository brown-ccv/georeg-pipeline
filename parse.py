import time
mt1 = time.time()
import stringParse, arcgeocoder, address
import zipcode
import sqlalchemy
import streetMatch1
import sys, glob, os, re, datetime
import pandas as pd
import numpy as np
import cv2
import pickle as pkl
from PIL import Image
from tesserocr import PyTessBaseAPI, RIL
import multiprocessing

#This is the driver script for pulling the data out of the images, parsing them, matching them, and geocoding them.

if not sys.argv[1]:
	raise Exception('You need to input the name of the directory you are running.')
dir_dir = str(sys.argv[1])

def naturalSort(String_):
	return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', String_)]

def streetTable():
    """Create DataFrame with streets, Zip Codes, and Cities."""
    print 'Creating Zipcode Table'
    #.csv with streets and corresponding zip codes
    street_df = pd.read_csv('streets_by_zip_code.csv', dtype = str)
    street_df.columns = ['Street', 'Zip_Code']

    street_df['Street'] = street_df['Street'].apply(lambda x: ' '.join(address.substitute_directions(x.split())))

    #Make zip a zipcode object.
    street_df['Zip_Code'] = street_df['Zip_Code'].apply(zipcode.isequal)

    #Find city corresponding to each zip code.
    street_df['City'] = street_df['Zip_Code'].apply(lambda x: x.city.encode('ascii', 'ignore'))

    street_df.to_pickle('stZipCty')
    return street_df


def makeCSV(dataFrame):
	today = datetime.date.today()
	dataFrame.set_index('Query')
	dataFrame['Address - From Geocoder'] = dataFrame['Address - From Geocoder'].astype('str').str.rstrip(',').str.strip('[[]]').str.lstrip('u\'').str.rstrip('\'').str.strip('[\\n ]')
	dataFrame['Company_Name'] = dataFrame['Company_Name'].astype('str').str.strip('[[]]').str.lstrip('u\'').str.rstrip('\'').str.strip('[\\n ]')
	dataFrame['File_List'] = dataFrame['File_List'] #.apply(lambda paths: [path.rpartition('/')[2] for path in paths[0]]).astype('str')
	dataFrame['Header'] = dataFrame['Header'].astype('str').str.strip('[[]]').str.lstrip('u\'').str.rstrip('\'').str.strip('[\\n ]').str.lstrip('>')
	dataFrame['Text'] = dataFrame['Text'].astype('str').str.strip('[[]]').str.lstrip('u\'').str.rstrip('\'').str.strip('[\\n ]')
	dataFrame['Query'] = dataFrame['Query'].astype('str').str.strip('[[]]').str.lstrip('u\'').str.rstrip('\'').str.strip('[\\n ]')
	dataFrame['Latitude'] = dataFrame['Latitude'].astype('str').str.strip('[[]]').str.lstrip('u\'').str.rstrip('\'').str.strip('[\\n ]')
	dataFrame['Longitude'] = dataFrame['Longitude'].astype('str').str.strip('[[]]').str.lstrip('u\'').str.rstrip('\'').str.strip('[\\n ]')
	dataFrame.to_csv(dir_dir + '/FOutput.csv', sep = ',')
	

def dfProcess(dataFrame):
	print('Matching city and street...')
	t1 = time.time()
	frame = streetMatch1.streetMatcher(dataFrame, dir_dir)
	t2 = time.time()
	print('Done in: ' + str(round(t2-t1, 3)) + ' s')
	print('Geocoding...')
	t1 = time.time()
	#frame.to_pickle('frame.pkl')
	#frame = pd.read_pickle('frame.pkl')
	fDF = arcgeocoder.geocode(frame, dir_dir)
	#print(str(len(fDF)) + ' addresses')
	t2 = time.time()
	print('Done in: ' + str(round(t2-t1, 3)) + ' s')
	return fDF

def getHorzHist(image):
	height, width = image.shape[:2]

	i=0
	histogram = [0]*width
	#count white pixels in each row
	while i<width:
		histogram[i] = height - cv2.countNonZero(image[:, i])
		# print(cv2.countNonZero(image[:, i]))
		i=i+1
	return histogram

def getFBP(image_file, sf):
	fbp_thresh = 3
	im = cv2.imread(image_file, 0)
	hhist = getHorzHist(im[5:-5,:])
	# print(hhist)
	#get location of first black pixel
	histstr = ','.join([str(li) for li in hhist])
	strpart = histstr.partition('0,')
	listStringPart = strpart[2].split(',')
	listIntPart = map(int, listStringPart)
	blackindices = [i for i, x in enumerate(listIntPart) if x > fbp_thresh]
	if len(blackindices) > 0:
		blackindx = blackindices[0]
	else:
		blackindx = 999999999
	# print(listIntPart, blackindx)
	cut = len(strpart[0].split(',')) + len(strpart[1].split(','))
	firstBlackPix = cut + blackindx - fbp_thresh
	return sf*float(firstBlackPix)

# def is_header_deprecated(fbp, text, file, entry_num):
# 	year = int(file.partition('/')[0].lstrip('cd'))
# 	if year > 1955:
# 		if len([l for l in text if l.isalpha()]) == 0:
# 			return False
# 		elif (fbp > 29) and ((float(len([l for l in text if l.isupper()])))/float(len([l for l in text if l.isalpha()])) > 0.9):
# 			return True
# 		elif (entry_num < 3) and ((float(len([l for l in text if l.isupper()])))/float(len([l for l in text if l.isalpha()])) > 0.95):
# 			return True
# 		else:
# 			return False
# 	else:
# 		if len([l for l in text if l.isalpha()]) == 0:
# 			return False
# 		elif (fbp > 40):
# 			return True
# 		elif (text.lstrip()[0] == '*') and (fbp > 30):
# 			return True
# 		else:
# 			return False

def is_header(rel_fbp, text, file, entry_num):
	year = int(file.partition('/')[0].lstrip('cd'))
	if year <= 1954:
		## Tweak threshold
		if len([l for l in text if l.isalpha()]) == 0:
			return False
		elif (rel_fbp > 40):
			return True
		elif (text.lstrip()[0] == '*') and (rel_fbp > 30):
			return True
		else:
			return False
	elif year <= 1962:
		## Big problems here
		if len([l for l in text if l.isalpha()]) == 0:
			return False
		elif (rel_fbp > 35) and ((float(len([l for l in text if l.isupper()])))/float(len([l for l in text if l.isalpha()])) > 0.9):
			return True
		elif (rel_fbp > 25) and ((float(len([l for l in text if l.isupper()])))/float(len([l for l in text if l.isalpha()])) > 0.97):
			return True
		elif (entry_num < 3) and ((float(len([l for l in text if l.isupper()])))/float(len([l for l in text if l.isalpha()])) > 0.95):
			return True
		elif (text.lstrip()[0] == '*') and (rel_fbp > 30):
			return True
		else:
			return False
	elif year <= 1968:
		## Tweak threshold
		if len([l for l in text if l.isalpha()]) == 0:
				return False
		elif (rel_fbp > 29) and ((float(len([l for l in text if l.isupper()])))/float(len([l for l in text if l.isalpha()])) > 0.9):
			return True
		elif (entry_num < 3) and ((float(len([l for l in text if l.isupper()])))/float(len([l for l in text if l.isalpha()])) > 0.95):
			return True
		else:
			return False
	elif year == 1970:
		## Tweak threshold
		if len([l for l in text if l.isalpha()]) == 0:
				return False
		elif (rel_fbp > 29) and ((float(len([l for l in text if l.isupper()])))/float(len([l for l in text if l.isalpha()])) > 0.9):
			return True
		elif (entry_num < 3) and ((float(len([l for l in text if l.isupper()])))/float(len([l for l in text if l.isalpha()])) > 0.95):
			return True
		else:
			return False
	elif year <= 1990:
		## Tweak threshold
		if len([l for l in text if l.isalpha()]) == 0:
				return False
		elif (rel_fbp > 29) and ((float(len([l for l in text if l.isupper()])))/float(len([l for l in text if l.isalpha()])) > 0.9):
			return True
		elif (entry_num < 3) and ((float(len([l for l in text if l.isupper()])))/float(len([l for l in text if l.isalpha()])) > 0.95):
			return True
		else:
			return False
	else:
		## Tweak threshold
		print("WARNING: Year not in range: ", year)
		if len([l for l in text if l.isalpha()]) == 0:
				return False
		elif (rel_fbp > 29) and ((float(len([l for l in text if l.isupper()])))/float(len([l for l in text if l.isalpha()])) > 0.9):
			return True
		elif (entry_num < 3) and ((float(len([l for l in text if l.isupper()])))/float(len([l for l in text if l.isalpha()])) > 0.95):
			return True
		else:
			return False


def ocr_file(file, api):
	image = Image.open(file)
	api.SetImage(image)
	api.SetVariable("tessedit_char_whitelist", "()*,'&.;-0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
	boxes = api.GetComponentImages(RIL.TEXTLINE, True)
	outStr = api.GetUTF8Text()
	text = outStr.encode('ascii', 'ignore')
	im = cv2.imread(file, 0)
	width = im.shape[1]
	sf = float(width)/float(2611)
	fbp = getFBP(file, sf)
	entry_num = int(file.rpartition('_')[2].rpartition('.png')[0])
	word_attrs = api.GetIterator().WordFontAttributes()
	if word_attrs is None:
		print("WARNING: Could not find word_attrs, sending none")
		word_attrs = {"error": "No word_attrs found"}
	else:
		word_attrs["error"] = "None"
	return file,text,fbp,sf,entry_num,word_attrs

def chunk_process_ocr(chunk_files):
	'''We process the OCR in chunks to avoid having to reload the API each time.'''
	rlist = []
	with PyTessBaseAPI() as api:
		for file in chunk_files:
			rlist.append(ocr_file(file, api))
	return rlist

def process(folder, do_OCR=True, make_table=False):
	#Make the zip code to city lookup table
	if make_table:
		streetTable()

	if do_OCR:
		files = []
		texts = []
		first_black_pixels = []
		sfs = []
		entry_nums = []
		print('Doing OCR')
		t1 = time.time()
		file_list = sorted(glob.glob(folder.rstrip('/') + '/*.png'), key = naturalSort)
		do_multiprocessing = True
		if do_multiprocessing:
			pool = multiprocessing.Pool(4)
			chunk_size = min(max(int(len(file_list)/50.0), 1), 20)
			chunk_list = [file_list[i:i + chunk_size] for i in list(range(0, len(file_list), chunk_size))]
			ocr_results = pool.map(chunk_process_ocr, chunk_list)
			flat_ocr_results = [item for sublist in ocr_results for item in sublist]
		else:
			flat_ocr_results = []
			with PyTessBaseAPI() as api:
				for file in file_list:
					flat_ocr_results.append(ocr_file(file, api))
		raw_data = pd.DataFrame(flat_ocr_results, columns = ['file','text','first_black_pixel','sf','entry_num', 'word_attrs'])
		t2 = time.time()
		print('Done in: ' + str(round(t2-t1, 3)) + ' s')

		print('Saving...')
		t1 = time.time()
		raw_data.to_pickle(dir_dir + '/raw_data.pkl')
		t2 = time.time()
		print('Done in: ' + str(round(t2-t1, 3)) + ' s')
	else:
		print('Reading raw data from raw_data.pkl...')
		t1 = time.time()
		raw_data = pd.read_pickle(dir_dir + '/raw_data.pkl')
		t2 = time.time()
		print('Done in: ' + str(round(t2-t1, 3)) + ' s')

	print('Concatenating entries...')
	t1 = time.time()
	page_breaks = raw_data[raw_data['entry_num'] == 1].index.tolist()
	ilist = list(range(1,raw_data.shape[0]))
	tb = time.time()
	print('Time so far: ' + str(round(tb-t1, 3)) + ' s')
	page_break = {i:max([num for num in page_breaks if i>=num]) for i in ilist}
	tb = time.time()
	print('Time so far: ' + str(round(tb-t1, 3)) + ' s')
	fbp_dict = {index:value for index,value in raw_data['first_black_pixel'].iteritems()}
	tb = time.time()
	print('Time so far: ' + str(round(tb-t1, 3)) + ' s')
	def get_relative_fbp(i):
		pbi = page_break[i]
		if i == pbi:
			rval = 0
		else:
			rval = fbp_dict[i] - min([fbp_dict[j] for j in list(range(max(pbi,i-8),i))])
		return rval
	raw_data = raw_data.assign(relative_fbp = [0.0] + [get_relative_fbp(i) for i in ilist])
	tb = time.time()
	print('Time so far: ' + str(round(tb-t1, 3)) + ' s')

	print(raw_data.head(10))

	raw_data = raw_data.assign(is_header = raw_data.apply(lambda row: is_header(row['relative_fbp'], row['text'], row['file'], row['entry_num']), axis=1))
	print(raw_data["is_header"])
	is_header_dict = {index:value for index,value in raw_data['is_header'].iteritems()}
	print(is_header_dict)
	tb = time.time()
	print('Time so far: ' + str(round(tb-t1, 3)) + ' s')
	raw_data_length = raw_data.shape[0]
	def concatenateQ(i):
		if i==raw_data_length - 1:
			return False
		elif i==0 and is_header_dict[i]:
			return False
		elif is_header_dict[i] and (not is_header_dict[i-1]):
			return False
		elif is_header_dict[i] and is_header_dict[i-1]:
			return True
		elif (not is_header_dict[i]) and is_header_dict[i+1]:
			return False
		elif raw_data.iloc[i+1]['relative_fbp'] > 9.0:
			return True
		else:
			return False

	raw_data = raw_data.assign(cq = raw_data.index.map(concatenateQ))
	tb = time.time()
	print('Time so far: ' + str(round(tb-t1, 3)) + ' s')

	raw_data.to_csv(dir_dir + '/raw_data.csv')

	file_lists = []
	file_list = []
	texts = []
	text = ''
	headers = []
	header = ''
	cq_dict = {index:value for index,value in raw_data['cq'].iteritems()}
	text_dict = {index:value for index,value in raw_data['text'].iteritems()}
	file_dict = {index:value for index,value in raw_data['file'].iteritems()}
	entry_num_dict = {index:value for index,value in raw_data['entry_num'].iteritems()}
	tb = time.time()
	print('Time so far: ' + str(round(tb-t1, 3)) + ' s')
	for index in raw_data.index:
		#raw_row = raw_data.iloc[i]
		row_text = text_dict[index]
		cq = cq_dict[index]
		file = file_dict[index]
		if is_header_dict[index]:
			if cq:
				header += ' ' + row_text.strip()
				#print(header)
			else:
				header = row_text.strip()
		elif entry_num_dict[index] == 1 and row_text == file.rpartition('_Page_')[2].rpartition(' ')[0]:
			pass
		elif cq:
			file_list.append(file)
			text += ' ' + row_text.strip()
		else:
			file_list.append(file)
			text += ' ' + row_text.strip()
			file_lists.append(file_list)
			headers.append(header)
			texts.append(text.strip())
			file_list = []
			text = ''

	data = pd.DataFrame(data={'Header':headers, 'Text':texts, 'File_List':file_lists})

	t2 = time.time()
	print('Done in: ' + str(round(t2-t1, 3)) + ' s')

	print('Writing data to data.csv...')
	t1 = time.time()
	data.to_csv(dir_dir + '/data.csv')
	t2 = time.time()
	print('Done in: ' + str(round(t2-t1, 3)) + ' s')

	print('Parsing text...')
	t1 = time.time()
	do_multiprocessing = False
	if do_multiprocessing:
		pool = multiprocessing.Pool(4)
		output_tuples = pool.map(stringParse.search, data['Text'].tolist())
	else:
		output_tuples = [stringParse.search(search_text) for search_text in data['Text'].tolist()]
	#streets,company_names = zip(*output_tuples)
	streets = [output_tuple[0] for output_tuple in output_tuples]
	company_names = [output_tuple[1] for output_tuple in output_tuples]
	data = data.assign(Street=streets, Company_Name=company_names)
	t2 = time.time()
	print('Done in: ' + str(round(t2-t1, 3)) + ' s')

	print('Expanding...')
	t1 = time.time()
	#data_list = [row for row in data.iterrows()]
	expanded_data_list = []
	for index,row in data.iterrows():
		if type(row['Street']) == list:
			row_streets = row['Street']
			new_row = row.copy()
			for street in row_streets:
				new_row['Street'] = street
				expanded_data_list.append(new_row.copy())
		else:
			expanded_data_list.append(row)
	data = pd.DataFrame(expanded_data_list)
	t2 = time.time()
	print('Done in: ' + str(round(t2-t1, 3)) + ' s')

	print('Matching city and street and geocoding...')
	t1 = time.time()
	result = dfProcess(data)
	t2 = time.time()
	print('Collective runtime: ' + str(round(t2-t1, 3)) + ' s')
	if not result.empty:
		print('Saving to FOutput.csv...')
		t1 = time.time()
		makeCSV(result)
		t2 = time.time()
		print('Done in: ' + str(round(t2-t1, 3)) + ' s')

process(dir_dir + '/no_ads/margins_fixed/cropped/entry', do_OCR=True, make_table=False)

mt2 = time.time()
print('Full runtime: ' + str(round(mt2-mt1, 3)) + ' s')
