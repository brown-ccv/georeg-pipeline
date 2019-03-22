import os
import re, datetime
from brownarcgis import BrownArcGIS
import pandas as pd
import time
#import line_profiler
from multiprocessing import Pool
import multiprocessing
import threading
import gc
import pickle as pkl

geolocator = BrownArcGIS(username = os.environ.get("BROWNGIS_USERNAME"), password = os.environ.get("BROWNGIS_PASSWORD"), referer = os.environ.get("BROWNGIS_REFERER"))


def geolocate(row_tuple):
	address,city=row_tuple
	#print('Geocoding ' + str(row['Address']))
	#print(row_tuple)
	#print(address)
	#print(city)
	try: 
		rval = geolocator.geocode(street=str(address), city =str(city), state='RI', n_matches = 1, timeout = 60)
	except:
		rval = 'timeout'
	return address,city,rval

def geocode(dataFrame, dir_dir):
	t1 = time.time()
	try:
		location_dict = pkl.load(open('location_dict.pkl', 'rb'))
		location_dict = { k:v for k, v in location_dict.items() if v!='timeout' }
	except:
		location_dict = {}
	
	hardcodes = pd.read_csv('geocoder_hardcodes.csv').dropna()
	hardcode_dict = {(row.Address,row.City):(row.Lat,row.Lon) for row in hardcodes.itertuples()}

	timeout_set = set()

	outside = ['SEEKONK', 'ATTLEBORO', 'NORTH ATTLEBORO', 'SOUTH ATTLEBORO']
	dataFrame = dataFrame[~dataFrame['City'].str.contains('|'.join(outside))]
	t2 = time.time()
	print('Prep time: ' + str(round(t2-t1,2)) + ' s')
	t1 = time.time()
	input_set = set((row.Address,row.City) for row in dataFrame.itertuples())
	print(len(input_set))
	input_list = list(input_set - set(location_dict))
	print(len(input_list))
	if input_list:
		n_processes = min(max(int(float(len(input_list))/20.0), 1), 50)
		pool = Pool(n_processes)
		if True:
			locations = [geolocate(input) for input in input_list]
		else:
			locations = pool.map(geolocate, input_list)
		counter = 0
		for location in locations:
			if location[2]=='timeout':
				counter += 1
		if counter != 0:
			print('WARNING!  There were ' + str(counter) + ' addresses that timed out during geocoding.')
		del pool
		gc.collect()
		for location_tuple in locations:
			if location_tuple[2] != 'timeout':
				location_dict[(location_tuple[0], location_tuple[1])] = location_tuple[2]
			else:
				timeout_set.add((location_tuple[0], location_tuple[1]))
	t2 = time.time()
	print('Geocoding search time: ' + str(round(t2-t1,2)) + ' s')

	master_list = []
	errors_list = []
	today = datetime.date.today()
	for row in dataFrame.itertuples():
		#lt1=time.time()
		#Pull data from previous dataframe
		address = str(row.Address)
		city = str(row.City)
		score = row.Conf_Score
		group = row.Header
		flist = row.File_List
		text = row.Text
		coName = row.Company_Name

		#Define Variables
		faddress = str(address) + ' ' + str(city)
		#print 'Geocoding: ' + faddress
		state = "RI"
		timeout = 60

		#Clean Queires
		city = re.sub(r"\'",'',city)
		faddress = re.sub(r"\'",'',faddress)

		#Look up the Location
		#gt1=time.time()
		if (address,city) in timeout_set:
			location = 'timeout'
		else:
			location = location_dict[(address,city)]
		#gt2=time.time()

		if location:
			try:
				match = location['candidates'][0]['attributes']
				conf_score = float(match["score"])
				result = match['match_addr']
				lat = match["location"]["y"]
				lon = match["location"]["x"]


				address_from_geocoder = str(result).rpartition('RI,')[0] + 'RI'

				rowFrame = {
					'Query': [faddress],
					'Address - From Geocoder': address_from_geocoder,
					'Geocode Score': conf_score,
					'Match Score': score,
					'Latitude': lat,
					'Longitude': lon,
					'Date_Added': today,
					'File_List': flist,
					'Text': [text],
					'Company_Name': coName,
					'Header': group
					}
				if conf_score > 85:
					master_list.append(rowFrame)
				else:
					errors_list.append(row)
			except:
				#print('Error for location: ' + location)
				errors_list.append(row)
		elif (address,city) in hardcode_dict.keys():
			lat,lon = hardcode_dict[(address,city)]
			rowFrame = {
					'Query': [faddress],
					'Address - From Geocoder': 'HARDCODE',
					'Geocode Score': 100.0,
					'Match Score': score,
					'Latitude': lat,
					'Longitude': lon,
					'Date_Added': today,
					'File_List': flist,
					'Text': [text],
					'Company_Name': coName,
					'Header': group
					}
			master_list.append(rowFrame)
		else:
			errors_list.append(row)
			continue
	
	t3 = time.time()
	print('Search time: ' + str(round(t3-t2,2)) + ' s')

	master = pd.DataFrame(master_list)
	errors = pd.DataFrame(errors_list)

	t4 = time.time()
	print('Concat time: ' + str(round(t4-t3,2)) + ' s')

	errors.to_csv(dir_dir + '/geocoder_errors.csv')

	pkl.dump(location_dict, open('location_dict.pkl', 'wb'))
	t5 = time.time()
	print('Save time: ' + str(round(t5-t4,2)) + ' s')

	return master
