import re, os, csv
import sys, glob
import pandas as pd
import TesserocrEx

def patternRec(res):
	regex = r"(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})"
	regex2 = r"(\d+)(.+)(\s[A-Z].+)(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})"
	regX = r"(\s[A-Z]$)"
	regex3 = r"(\d+.+)(\s[A-Z].+\s$)"
	regex4 = r"(\s.+\s$)"
	loc = []
	pl = []
	companyName = []
	# with open('outfile.txt', 'a') as f:
	# 	sys.stdout = f
	# 	print 'test ********************'
	#for line in res.splitlines():

		#search for Phone Number
	if re.search(regex, res):
		match = re.findall(regex, res)
		#print 'Phone Number: %s' % (match)
		#search for Address/City
		match2 = re.finditer(regex2, res)
		#print match2
		if match2 is not None:
			for match in match2:
				addss = match.group(1) + match.group(2)
				sity = match.group(3)
				out = addss + sity
				coName = re.sub(out, '',res)
				if addss is not None:
					if re.search(regX, addss):
						direction = re.search(regX, addss)
						lttr = direction.group(0)
						addss = re.sub(lttr, '', addss)
						finalSity = lttr+sity
						out1 = addss+finalSity
						comName = re.sub(out1,'',res)
						#print 'Address: %s' % (addss)
						#print 'City: %s' % (finalSity)
						loc.append(addss)
						pl.append(finalSity)
						companyName.append(comName)
					else:
						#print 'Address: %s' % (addss)
						#print 'City: %s' % (sity)
						loc.append(addss)
						pl.append(sity)
						companyName.append(coName)
				else:
					addss = 'N/A'
					sity = 'N/A'
					cName = 'N/A'
					return addss, sity, cName
			return loc, pl, companyName
			#Does the end of the address contain a city direction?
		else:
			addss = 'N/A'
			sity = 'N/A' 
			cName = 'N/A'
			return addss, sity, cName
	else:
		print res
		addss = 'N/A'
		sity = 'N/A'
		cName = 'N/A' 
		return addss, sity, cName
			# else:
			# 	print(line)
			# 	#search for address/city alone in line
			# 	try:
			# 		match3 = re.search(regex3, line)
			# 		trial = match3.group(1)
			# 		print 'Possible Address...?: %s' % trial
			# 		print 'Possible City: %s' % match3.group(2)
			# 	except:
			# 		print 'No match'
			# 	print('****************************')