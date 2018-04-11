import time
mt1 = time.time()
import stringParse, arcgeocoder, findHeader, regX2
import zipcode
import streetMatch1
import sys, glob, os, re, datetime
import pandas as pd
import cv2
import pickle as pkl
from PIL import Image
from tesserocr import PyTessBaseAPI, RIL

#This is the driver script for pulling the data out of the images, parsing them, matching them, and geocoding them.

crop_points_dict = pkl.load(open('test/no_ads/margins_fixed/cropped/crop_points_dict.pkl'))

def naturalSort(String_):
	return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', String_)]

def streetTable():
    """Create DataFrame with streets, Zip Codes, and Cities."""
    print 'Creating Zipcode Table'
    #.csv with streets and corresponding zip codes
    street_df = pd.read_csv('streets_by_zip_code.csv', dtype = str)
    street_df.columns = ['Street', 'Zip_Code']

    #Make zip a zipcode object.
    street_df['Zip_Code'] = street_df['Zip_Code'].apply(zipcode.isequal)

    #Find city corresponding to each zip code.
    street_df['City'] = street_df['Zip_Code'].apply(lambda x: x.city.encode('ascii', 'ignore'))

    street_df.to_pickle('stZipCty')
    return street_df

def makeCSV(dataFrame):
	today = datetime.date.today()
	dataFrame.set_index('Query')
	dataFrame['Address - From Geocoder'] = dataFrame['Address - From Geocoder'].astype('str').str.rstrip(',')
	dataFrame['Company_Name'] = dataFrame['Company_Name'].astype('str').str.strip('[[]]').str.lstrip('u\'').str.rstrip('\'').str.strip('[\\n ]')
	dataFrame['File_List'] = dataFrame['File_List'].astype('str')
	dataFrame['Header'] = dataFrame['Header'].astype('str').str.strip('[[]]').str.lstrip('u\'').str.rstrip('\'').str.strip('[\\n ]').str.lstrip('>')
	dataFrame['Text'] = dataFrame['Text'].astype('str').str.strip('[[]]').str.lstrip('u\'').str.rstrip('\'').str.strip('[\\n ]')
	dataFrame['Query'] = dataFrame['Query'].astype('str')
	dataFrame.to_csv('FOutput.csv', sep = ',')

def dfProcess(dataFrame):
	frame = streetMatch1.streetMatcher(dataFrame)
	fDF = arcgeocoder.geocode(frame)
	print(str(len(fDF)) + ' addresses')
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
	im = cv2.imread(image_file, 0)
	hhist = getHorzHist(im[5:-5,:])
	# print(hhist)
	#get location of first black pixel
	histstr = ','.join([str(li) for li in hhist])
	strpart = histstr.partition('0,')
	listStringPart = strpart[2].split(',')
	listIntPart = map(int, listStringPart)
	blackindx = next((i for i, x in enumerate(listIntPart) if x), None)
	# print(listIntPart, blackindx)
	cut = len(strpart[0].split(',')) + len(strpart[1].split(','))
	firstBlackPix = cut + blackindx - 2
	return sf*float(firstBlackPix)

def is_header(fbp, text, file, entry_num):
	if len([l for l in text if l.isalpha()]) == 0:
		return False
	elif (fbp > 200) and ((float(len([l for l in text if l.isupper()])))/float(len([l for l in text if l.isalpha()])) > 0.9):
		return True
	elif (entry_num < 3) and ((float(len([l for l in text if l.isupper()])))/float(len([l for l in text if l.isalpha()])) > 0.95):
		return True
	else:
		return False

def process(folder, do_OCR=True, make_table=False):
	#Make the zip code to city lookup table
	if make_table:
		streetTable()

	if do_OCR:
		files = []
		texts = []
		first_black_pixels = []
		sfs = []
		crop_points_list = []
		entry_nums = []
		print('Doing OCR')
		t1 = time.time()
		with PyTessBaseAPI() as api:
			for file in sorted(glob.glob("test/no_ads/margins_fixed/cropped/entry/*.png"), key = naturalSort):
				#print('Reading...')
				#ta = time.time()
				image = Image.open(file)
				api.SetImage(image)
				api.SetVariable("tessedit_char_whitelist", "()*,'&.;-0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
				boxes = api.GetComponentImages(RIL.TEXTLINE, True)
				outStr = api.GetUTF8Text()
				text = outStr.encode('ascii', 'ignore')
				#tb = time.time()
				#print('Read in: ' + str(round(tb-ta, 5)) + ' s')

				#print('Finding scale factor...')
				#ta = time.time()
				im = cv2.imread(file, 0)
				width = im.shape[1]
				sf = float(width)/float(2611)
				#tb = time.time()
				#print('Scale factor found in: ' + str(round(tb-ta, 5)) + ' s')

				#print('Finding first black pixel...')
				#ta = time.time()
				fbp = getFBP(file, sf)
				#print('FBP: ' + str(fbp))
				#tb = time.time()
				#print('First black pixel found in: ' + str(round(tb-ta, 5)) + ' s')

				texts.append(text)
				files.append(file)
				entry_nums.append(int(file.rpartition('_')[2].rpartition('.png')[0]))
				crop_points_list.append(crop_points_dict[file.rpartition('/')[2]])
				sfs.append(sf)
				first_black_pixels.append(fbp)

		raw_data = pd.DataFrame(data={'file':files, 'text':texts, 'first_black_pixel':first_black_pixels, 'sf':sfs, 'crop_points':crop_points_list, 'entry_num':entry_nums})
		t2 = time.time()
		print('Done in: ' + str(round(t2-t1, 3)) + ' s')
		print('Saving...')
		t1 = time.time()
		raw_data.to_pickle('raw_data.pkl')
		t2 = time.time()
		print('Done in: ' + str(round(t2-t1, 3)) + ' s')
	else:
		print('Reading raw data from raw_data.pkl...')
		t1 = time.time()
		raw_data = pd.read_pickle('raw_data.pkl')
		t2 = time.time()
		print('Done in: ' + str(round(t2-t1, 3)) + ' s')

	print('Concatenating entries...')
	t1 = time.time()

	raw_data = raw_data.assign(is_header = raw_data.apply(lambda row: is_header(row['first_black_pixel'], row['text'], row['file'], row['entry_num']), axis=1))
	page_breaks = raw_data[raw_data['entry_num'] == 1].index.tolist()
	def page_break(i):
		return max([num for num in page_breaks if i>=num])
	raw_data = raw_data.assign(relative_fbp = [0.0] + [raw_data.iloc[i]['first_black_pixel'] - raw_data.iloc[max(page_break(i),i-8):i]['first_black_pixel'].min() for i in list(range(1,raw_data.shape[0]))])

	def concatenateQ(i):
		if i==raw_data.shape[0] - 1:
			return False
		elif i==0 and raw_data.iloc[i]['is_header']:
			return False
		elif raw_data.iloc[i]['is_header'] and (not raw_data.iloc[i-1]['is_header']):
			return False
		elif raw_data.iloc[i]['is_header'] and raw_data.iloc[i-1]['is_header']:
			return True
		elif (not raw_data.iloc[i]['is_header']) and raw_data.iloc[i+1]['is_header']:
			return False
		elif raw_data.iloc[i+1]['relative_fbp'] > 50.0 * raw_data.iloc[i+1]['sf']:
			return True
		else:
			return False

	raw_data = raw_data.assign(cq = raw_data.index.map(concatenateQ))

	raw_data.to_csv('raw_data.csv')

	file_lists = []
	file_list = []
	texts = []
	text = ''
	headers = []
	header = ''
	for i in list(range(raw_data.shape[0])):
		raw_row = raw_data.iloc[i]
		row_text = raw_row['text']
		cq = raw_row['cq']
		file = raw_row['file']
		if raw_row['is_header']:
			if cq:
				header += ' ' + row_text.strip()
				print(header)
			else:
				header = row_text.strip()
		elif raw_row['entry_num'] == 1 and row_text == file.rpartition('_Page_')[2].rpartition(' ')[0]:
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
	data.to_csv('data.csv')
	t2 = time.time()
	print('Done in: ' + str(round(t2-t1, 3)) + ' s')

	print('Parsing text...')
	streets = []
	company_names = []
	for txt in data['Text']:
		st, coName = stringParse.search(txt)
		streets.append(st)
		company_names.append(coName)
	data = data.assign(Street=streets, Company_Name=company_names)

	print('Matching city and street and geocoding...')
	t1 = time.time()
	result = dfProcess(data)
	t2 = time.time()
	print('Done in: ' + str(round(t2-t1, 3)) + ' s')
	if not result.empty:
		print('Saving to FOutput.csv...')
		t1 = time.time()
		makeCSV(result)
		t2 = time.time()
		print('Done in: ' + str(round(t2-t1, 3)) + ' s')

process('test/margins_fixed/no_ads/cropped/entry', do_OCR=False, make_table=False)

mt2 = time.time()
print('Full runtime: ' + str(round(mt2-mt1, 3)) + ' s')
