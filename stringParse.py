import re
import cityMatch
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

#Parses the text into street, city, phone number, and company name.

def split_on_st(string, st):
	words = string.split()
	i = words.index(st)
	if i<2:
		return 'Search','failed',True
	else:
		j = i-2
		while j>0 and (not (re.match('\d+', words[j]))):
			j -= 1
		if j < 0:
			return 'Search','failed',True
		else:
			rtuple = string.partition(' ' + words[j] + ' ')
			return rtuple[0],(rtuple[1] + rtuple[2]), False

def search(string):
	regex = '(\D+)(\s\d+\s)(.+)'

	print('Parsing: ' + string)
	do_regex = True
	if re.match('.+\sAv\s.*', string) or re.match('.+\sAv$', string):
		companyName, street, do_regex = split_on_st(string,'Av')
	elif re.match('.+\sAve\s.*', string) or re.match('.+\sAve$\s.*', string):
		companyName, street, do_regex = split_on_st(string,'Ave')
	elif re.match('.+\sSt\s.*', string) or re.match('.+\sSt$.*', string):
		companyName, street, do_regex = split_on_st(string,'St')
	elif re.match('.+\sRd\s.*', string) or re.match('.+\sRd$', string):
		companyName, street, do_regex = split_on_st(string,'Rd')
	elif re.match('.+\sCt\s.*', string) or re.match('.+\sCt$', string):
		companyName, street, do_regex = split_on_st(string,'Ct')
	elif re.match('.+\sLn\s.*', string) or re.match('.+\sLn$', string):
		companyName, street, do_regex = split_on_st(string,'Ln')
	elif re.match('.+\sDr\s.*', string) or re.match('.+\sDr$', string):
		companyName, street, do_regex = split_on_st(string,'Dr')
	
	if do_regex:
		parts = re.search(regex, string)

		if parts:
			street = (parts.group(2) + parts.group(3)).strip()
			companyName = parts.group(1).strip()
			return street, companyName
		elif re.search('(\D+)(\d+)(.+)', string):
			parts = re.search('(\D+)(\d+)(.+)', string)
			street = (parts.group(2) + parts.group(3)).strip()
			companyName = parts.group(1).strip()
			return street, companyName
		else:
			print('Regex failure: no number found')
			return 'N/A', 'N/A'
	else:
		return street, companyName

