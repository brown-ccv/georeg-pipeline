import time
mt1 = time.time()
import stringParse, cityMatch, streetMatch1, arcgeocoder, findHeader, TesserocrEx, regX2
import sys, pickle, glob, os, re, datetime
import pandas as pd
import pickle
import sqlalchemy
from sqlalchemy import *
import multiprocessing

#This is the driver script for pulling the data out of the images, parsing them, matching them, and geocoding them.

def naturalSort(String_):
	return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', String_)]

def makeCSV(dataFrame):
	today = datetime.date.today()
	dataFrame.set_index('Query')
	dataFrame['Address - From Geocoder'] = dataFrame['Address - From Geocoder'].astype('str').str.rstrip(',')
	dataFrame['Company_Name'] = dataFrame['Company_Name'].astype('str').str.strip('[[]]').str.lstrip('u\'').str.rstrip('\'').str.strip('[\\n ]')
	dataFrame['Filename'] = dataFrame['Filename'].astype('str').str.strip('[[]\']').str.lstrip('test/margins_fixed/no_ads/cropped/entry/')
	dataFrame['Header'] = dataFrame['Header'].astype('str').str.strip('[[]]').str.lstrip('u\'').str.rstrip('\'').str.strip('[\\n ]').str.lstrip('>')
	dataFrame['Text'] = dataFrame['Text'].astype('str').str.strip('[[]]').str.lstrip('u\'').str.rstrip('\'').str.strip('[\\n ]')
	dataFrame['Query'] = dataFrame['Query'].astype('str')
	dataFrame['Latitude'] = dataFrame['Latitude']
	dataFrame['Longitude'] = dataFrame['Longitude']
	dataFrame['Geocode Score'] = dataFrame['Geocode Score']
	dataFrame['Match Score'] = dataFrame['Match Score']
	dataFrame.to_csv('FOutput.csv', sep = ',')


def dfProcess(dataFrame):
	cty = cityMatch.cityMatcher(dataFrame)
	frame = streetMatch1.streetMatcher(cty)
	fDF = arcgeocoder.geocode(frame)
	print(str(len(fDF)) + ' addresses')
	return fDF

def createCache(dataFrame):
	engine = create_engine('sqlite:///ACache.sqlite', echo=False)
	dataFrame = dataFrame.drop_duplicates(['Query'], keep = 'last')
	# dataFrame['Address - From Geocoder'] = dataFrame['Address - From Geocoder'].astype('str') 
	# dataFrame['Query'] = dataFrame['Query'].astype('str')
	# dataFrame['Latitude'] = dataFrame['Latitude'].astype('int')
	# dataFrame['Longitude'] = dataFrame['Longitude'].astype('int')
	# dataFrame['Geocode Score'] = dataFrame['Geocode Score'].astype('int')
	# dataFrame['Match Score'] = dataFrame['Match Score'].astype('int')
	
	dataFrame = dataFrame.drop('Filename', 1)
	dataFrame = dataFrame.drop('Text', 1)
	dataFrame = dataFrame.drop('Company_Name', 1)
	dataFrame = dataFrame.drop('Header', 1)
	dataFrame = dataFrame.set_index('Query', 1)
	dataFrame.to_sql('Table1', engine, if_exists = 'append')

def process(folder):
	global grp
	grp = 'Empty'
	data = pd.DataFrame()
	#df1 = pd.DataFrame()
	print('OCRing and parsing images...')
	t1 = time.time()
	for file in sorted(glob.glob("test/margins_fixed/no_ads/cropped/entry/*.png"), key = naturalSort):
		isHead = findHeader.process(file)
		if isHead == True:
			grp = TesserocrEx.ocr(file)
		elif isHead == False:
			st = []
			cty = []
			
			ta = time.time()
			txt = TesserocrEx.ocr(file)
			#st, cty, coName = regX2.patternRec(txt)
			st, cty, coName, phn = stringParse.search(txt)

			og = pd.DataFrame({
				'Text': [txt],
				'Street': [st],
				'City': [cty],
				'Phone': [phn],
				'Header': [grp],
				'Filename': [file],
				'Company_Name': [coName]
				})
			#print(og)
			tb = time.time()
			print('Read in: ' + str(round(tb-ta, 5)) + ' s')
			data = data.append(og, ignore_index = True)
	t2 = time.time()
	print('Done in: ' + str(round(t2-t1, 3)) + ' s')
	print('Saving...')
	t1 = time.time()
	data.index = range(1,len(data)+1)
	data.to_csv('TempO.csv', sep=',')
	t2 = time.time()
	print('Done in: ' + str(round(t2-t1, 3)) + ' s')

	#print data
	#data.to_pickle('tbl')
	print('Matching city and street and geocoding...')
	t1 = time.time()
	result = dfProcess(data)
	t2 = time.time()
	print('Done in: ' + str(round(t2-t1, 3)) + ' s')
	# result['Text'] = result['Text'].astype(str)
	# result = result.drop_duplicates(subset = ['Address - From Geocoder', 'Text'], keep = 'last')
	if not result.empty:
		print('Saving to FOutput.csv...')
		t1 = time.time()
		makeCSV(result)
		#cache = createCache(result)
		t2 = time.time()
		print('Done in: ' + str(round(t2-t1, 3)) + ' s')
    # parallelize lemstem_df() over text df chunks

	# create as many processes as there are CPUs on your machine
	# num_processes = multiprocessing.cpu_count()
	# # calculate the chunk size as an integer
	# chunk_size = int(data.shape[0]/num_processes)
	# # works even if the df length is not evenly divisible by num_processes
	# chunks = [data.ix[data.index[i:i + chunk_size]]
	#   for i in range(0, data.shape[0], chunk_size)]

	# # create our pool with `num_processes` processes
	# pool = multiprocessing.Pool(processes=num_processes)

	# # partial with fixed second argument because pool.map can't take functions with more than one argument
	# # lemstem_df_par = partial(lemstem_df, method='stem')

	# # apply our function to each chunk in the list
	# result1 = pool.map(dfProcess, chunks)

	# combine the results from our pool to a dataframe
	#df1 = pd.DataFrame().reindex_like(data)
	# textlem['CLEAN_TEXT'] = np.NaN
	# for i in range(len(result1)):
	# 	df1 = df1.append(result1[i])
	# print 'df1:'
	# print df1
	# df1.to_pickle('data')


process('test/margins_fixed/no_ads/cropped/entry')


# def makeCSV():
# 	df1 = pd.read_pickle('df_2_geocode')
# 	df1['Full_Address'] = df1['Address'] + df1['City']
# 	df1 = df1.drop('Address', 1)
# 	df1 = df1.drop('City', 1)
# 	df1 = df1.drop('Conf._Score', 1)
# 	print df1
# 	print len(df1)
# 	#print df1
# 	df2 = getDB()
# 	df2 = df2.drop('index', 1)
# 	print df2
# 	print len(df2)
# 	output = pd.merge(df1, df2, left_on = 'Full_Address', right_on = 'Query', how = 'outer')
# 	#print output
# 	output = output.set_index('Full_Address')
# 	output = output.drop('Query', 1)
# 	#output = output.drop_duplicates(['Company_Name'], keep = 'last')
# 	output.to_csv('FOutput.csv', sep = ',')

mt2 = time.time()
print('Run time: ' + str(round(mt2-mt1, 3)) + ' s')
