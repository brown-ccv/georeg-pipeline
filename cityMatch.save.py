from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import numpy as np
import re
import pandas as pd

#Matches string to database of cities

city_dict = {
	'Prv': 'PROVIDENCE',
	'Prov': 'PROVIDENCE',
	'Providence': 'PROVIDENCE',
	'Paw': 'PAWTUCKET',
	'Pawt': 'PAWTUCKET',
	'Pawtucket': 'PAWTUCKET',
	'N Prv': 'NORTH PROVIDENCE',
	'N Prov': 'NORTH PROVIDENCE',
	'N Providence': 'NORTH PROVIDENCE',
	'E Prv': 'EAST PROVIDENCE',
	'E Prov': 'EAST PROVIDENCE',
	'E Providence': 'EAST PROVIDENCE',
	'Wrwk': 'WARWICK',
	'Warwick': 'WARWICK',
	'W Wrwk': 'WEST WARWICK',
	'West Wrwk': 'WEST WARWICK',
	'W Warwick': 'WEST WARWICK',
	'West Warwick': 'WEST WARWICK',
	'Smithfield': 'SMITHFIELD',
	'Sfld': 'SMITHFIELD',
	'N Smithfield': 'NORTH SMITHFIELD',
	'N Sfld': 'NORTH SMITHFIELD',
	'Cumberland': 'CUMBERLAND',
	'Cmd': 'CUMBERLAND',
	'Seekonk': 'SEEKONK',
	'Seek': 'SEEKONK',
	'Cranston': 'CRANSTON',
	'Crns': 'CRANSTON',
	'Cf': 'CENTRAL FALLS',
	'C F': 'CENTRAL FALLS',
	'Central Falls': 'CENTRAL FALLS',
	'Attleboro': 'ATTLEBORO',
	'Attl': 'ATTLEBORO',
	'N Attleboro': 'NORTH ATTLEBORO',
	'N Attl': 'NORTH ATTLEBORO',
	'S Attleboro': 'SOUTH ATTLEBORO',
	'S Attl': 'SOUTH ATTLEBORO',
	'Woon': 'WOONSOCKET',
	'Woonsocket': 'WOONSOCKET',
	'Lincoln': 'LINCOLN',
	'Lcln': 'LINCOLN',
	'Kingston': 'SOUTH KINGSTOWN',
	'S Kingstown': 'SOUTH KINGSTOWN',
	'South Kingstown': 'SOUTH KINGSTOWN',
	'SKgtwn': 'SOUTH KINGSTOWN',
	'S Kgtwn': 'SOUTH KINGSTOWN',
	'N Kingstown': 'NORTH KINGSTOWN',
	'North Kingstown': 'NORTH KINGSTOWN',
	'NKgtwn': 'NORTH KINGSTOWN',
	'N Kgtwn': 'NORTH KINGSTOWN',
	'Johnston': 'JOHNSTON',
	'Narr': 'NARRAGANSETT',
	'Narragansett': 'NARRAGANSETT',
	'Newport': 'NEWPORT',
	'Bris': 'BRISTOL',
	'Bristol': 'BRISTOL',
	'Tiverton': 'TIVERTON',
	'Little Compton': 'LITTLE COMPTON',
	'Portsmouth': 'PORTSMOUTH',
	'Middletown': 'MIDDLETOWN',
	'Warren': 'WARREN',
	'Barrington': 'BARRINGTON',
	'Burrillville': 'BURRILLVILLE',
	'Foster': 'FOSTER',
	'Coventry': 'COVENTRY',
	'East Greenwich': 'EAST GREENWICH',
	'E Greenwich': 'EAST GREENWICH',
	'West Greenwich': 'WEST GREENWICH',
	'W Greenwich': 'WEST GREENWICH',
	'Richmond': 'RICHMOND',
	'Exeter': 'EXETER',
	'Hopkinton': 'HOPKINTON',
	'Charlestown': 'CHARLESTOWN',
	'New Shoreham': 'NEW SHOREHAM',
	'Block Island': 'NEW SHOREHAM',
	'Jamestown': 'JAMESTOWN'
}

Providence = ['Prv', 'Prov', 'Providence']
Pawtucket = ['Paw', 'Pawt', 'Pawtucket']
nProvidence = ['N Prov']
eProvidence = ['E Prov']
Warwick = ['Wrwk', 'Warwick']
Smithfield = ['Sfld', 'Smithfield']
Cumberland = ['Cmb', 'Cumberland']
Seekonk = ['Seek', 'Seekonk']
Cranston = ['Crns', 'Cranston']
Central_Falls = ['Cf', 'C F', 'Central Falls']
Attleboro = ['Attl', 'Attleboro']
nAttleboro = ['N Attleboro', 'N Attl']
sAttleboro = ['S Attleboro', 'S Attl']
Woonsocket = ['Woon', 'Woonsocket']
Lincoln = ['Lcln', 'Lincoln']
Kingston = ['Kingston']
nKingston = ['N Kingston', 'North Kingston', 'NKgtwn']
Johnston = ['Johnston']
Narragansett = ['Narr', 'Narragansett']
Newport = ['Newport']
Bristol = ['Bristol','Bris']

choices = ['Prv', 'Prov', 'Providence', 'Paw', 'Pawt', 'Pawtucket', 'N Prov', 'E Prov', 'Wrwk', 'Warwick', 
'Sfld', 'Smithfield', 'Cmb', 'Cumberland', 'Seek', 'Seekonk', 'Crns', 'Cranston', 'Cf', 'C F', 'Central Falls', 'Attl', 'Attleboro', 
'N Attleboro', 'N Attl', 'S Attleboro', 'S Attl', 'Woon', 'Woonsocket', 'Lcln', 'Lincoln', 'Kingston', 'North Kingston', 'Johnston', 
'Narr', 'Narragansett', 'Newport', 'N/A', 'N Kgtwn', 'N Kingston', 'Bris', 'Bristol']


def expandCity(city):
	if city in Providence:
		return 'PROVIDENCE'
	if city in Pawtucket:
		return 'PAWTUCKET'
	if city in nProvidence:
		return 'NORTH PROVIDENCE'
	if city in eProvidence:
		return 'EAST PROVIDENCE'
	if city in Warwick:
		return 'WARWICK'
	if city in Smithfield:
		return 'SMITHFIELD'
	if city in Cumberland:
		return 'CUMBERLAND'
	if city in Seekonk:
		return 'SEEKONK'
	if city in Cranston:
		return 'CRANSTON'
	if city in Central_Falls:
		return 'CENTRAL FALLS'
	if city in Attleboro:
		return 'ATTLEBORO'
	if city in nAttleboro:
		return 'NORTH ATTLEBORO'
	if city in sAttleboro:
		return 'SOUTH ATTLEBORO'
	if city in Woonsocket:
		return 'WOONSOCKET'
	if city in Lincoln:
		return 'LINCOLN'
	if city in Kingston:
		return 'KINGSTON'
	if city in nKingston:
		return 'NORTH KINGSTON'
	if city in Johnston:
		return 'JOHNSTON'
	if city in Narragansett:
		return 'NARRAGANSETT'
	if city in Newport:
		return 'NEWPORT'
	if city in Bristol:
		return 'BRISTOL'
	if city == 'N/A':
		return 'N/A'

def cityMatcher(dataFrame):
#This Function will take the DataFrame produced by the regex and return the same DataFrame with cities matched
	query = []
	#dataFrame = pd.read_pickle('tbl')
	correctCnt = 0
	naCnt = 0
	noMatchCnt = 0
	result = pd.DataFrame()
	for index, row in dataFrame.iterrows():
		query = str(row['City'])
		Text = row['Text']
		Street = row['Street']
		Header = row['Header']
		nFile = row['Filename']
		coName = row['Company_Name']
		fCity = []

		#Remove some clutter from encoding
		query = re.sub(r"\[u\'", '', query)
		query = re.sub(r"\']$", '', query)
		query = re.sub(r"\[\]", 'N/A', query)
		query = re.sub(r"-", '', query)



		#Check if there are mulitple entries (seperated by a comma)
		if re.search(r"(.+),(.+)", query):
			#print 'Multiple Entries:' + query
			for x in query.split(','):
				for cty, score in process.extract(x, choices, scorer = fuzz.ratio, limit = 1):
					if score > 50:
						cty = expandCity(cty)
						fCity.append(cty)
						#print 'Entry: ' + x + '; Corrected City: ' + cty
						correctCnt = correctCnt + 1
						if cty == 'N/A':
							naCnt = naCnt + 1
					else:
						fCity.append(x + 'No Match')
						# print 'Entry: ' + x + ' Couldnt Match'
						noMatchCnt = noMatchCnt + 1

				#set up Row by Row DataFrame		
				og = pd.DataFrame({
					'Text' : [Text],
					'City' : [fCity],
					'Street' : [Street],
					'Header': [Header],
					'Filename': [nFile],
					'Company_Name': [coName]
				})
			result = result.append(og, ignore_index=True)

		else:
			for cty, score in process.extract(query, choices, scorer = fuzz.ratio, limit = 1):
				if score > 50:
					cty = expandCity(cty)
					# print 'Entry: ' + query + '; Corrected City: ' + cty
					correctCnt = correctCnt + 1
					if cty == 'N/A':
						naCnt = naCnt + 1
				else:
					cty = query + 'No Match' 
					# print 'Entry: ' + query + ' Couldnt Match'
					noMatchCnt = noMatchCnt + 1

				#Set up Row by Row DataFrame	
				rowFrame = ({
					'Text': [Text],
					'City': [cty],
					'Street': [Street],
					'Header': [Header],
					'Filename': [nFile],
					'Company_Name': [coName]
			})
			result = result.append(rowFrame, ignore_index=True)	

	result.index = range(1,len(result)+1)
	# print result
	# print 'Correct'
	# print correctCnt
	# print 'N/A' 
	# print naCnt
	# print 'No Match'
	# print noMatchCnt
	# print 'Correct Cities'
	# print correctCnt - naCnt
	return result
