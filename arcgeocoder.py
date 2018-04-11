import os
import re, datetime
from brownarcgis import BrownArcGIS
import pandas as pd




def geocode(dataFrame):
	geolocator = BrownArcGIS(username = os.environ.get("BROWNGIS_USERNAME"), password = os.environ.get("BROWNGIS_PASSWORD"), referer = os.environ.get("BROWNGIS_REFERER"))

	master = pd.DataFrame()
	#dataFrame = pd.read_pickle('df_2_geocode')
	outside = ['SEEKONK', 'ATTLEBORO', 'NORTH ATTLEBORO', 'SOUTH ATTLEBORO']
	dataFrame = dataFrame[~dataFrame['City'].str.contains('|'.join(outside))]
	for index, row in dataFrame.iterrows():
		#Pull data from previous dataframe
		address = str(row['Address'])
		city = str(row['City'])
		score = row['Conf._Score']
		group = row['Header']
		flist = row['File_List']
		text = row['Text']
		coName = row['Company_Name']

		#Define Variables
		faddress = str(address) + ' ' + str(city)
		print 'Geocoding: ' + faddress
		state = "RI"
		timeout = 60
		today = datetime.date.today()

		#Clean Queires
		city = re.sub(r"\'",'',city)
		faddress = re.sub(r"\'",'',faddress)

		#Look up the Location
		location = geolocator.geocode(street=address, city =city, state=state, n_matches = 1, timeout = 60)

		if location:
			match = location['candidates'][0]['attributes']
			conf_score = float(match["score"])
			result = match['match_addr']
			lat = match["location"]["y"]
			lon = match["location"]["x"]

			patt = r"(.+RI,)(.+)"
			if re.search(patt, str(result)):
				sep = re.search(patt, str(result))
				pt1 = sep.group(1)
				pt2 = sep.group(2)

			rowFrame = pd.DataFrame({
				'Query': [faddress],
				'Address - From Geocoder': [pt1],
				'Geocode Score': [conf_score],
				'Match Score': [score],
				'Latitude': [lat],
				'Longitude': [lon],
				'Date_Added': [today],
				'File_List': [flist],
				'Text': [text],
				'Company_Name': [coName],
				'Header': [group]
				})
			master = master.append(rowFrame)
		else:
			continue

	if master.empty:
		print 'Nothing new to add'

	return master
