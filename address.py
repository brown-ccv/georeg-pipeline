import pandas as pd
import re
import numpy as np
from cityMatch import city_match

from fuzzywuzzy import fuzz, process
import time


ri_streets_table = pd.read_pickle('stZipCty')
street_patt = re.compile(r"(^\d+)(.+)")

def substitute_directions(inwords):
    outwords = inwords[:]
    for i in list(range(len(inwords))):
        if outwords[i] == 'W':
            outwords[i] = 'WEST'
        if outwords[i] == 'E':
            outwords[i] = 'EAST'
        if outwords[i] == 'N':
            outwords[i] = 'NORTH'
        if outwords[i] == 'S':
            outwords[i] = 'SOUTH'
    return outwords

def street_scorer(istr1, istr2):
    str1 = istr1.upper()
    str2 = istr2.upper()
    words1 = substitute_directions(str1.split())
    words2 = substitute_directions(str2.split())
    word1 = sorted(words1, key=len, reverse=True)[0]
    word2 = sorted(words2, key=len, reverse=True)[0]
    return (fuzz.ratio(word1, word2) + fuzz.ratio(' '.join(words1), ' '.join(words2)))/2

class Address(object):
    """Class to hold address information."""

    def __init__(self, street=None, city='PROVIDENCE', streets_table=ri_streets_table):

        self.city = city
        self.street = street
        self.streets_table = streets_table
        self.addr_matches = []

    def set_addr_matches(self, cutoff, limit):
        """Find (street, city, score) fuzzy matches for addresses in streets_table."""

        self.addr_matches = []

        street = self.street.strip()

        if street == 'N/A':
            print('N/A')
            return

        if re.match('.+\(.{1,20}\)$', street):
            print(street)
            city_guess = street.partition('(')[2].partition(')')[0]
            city_guess = re.sub('/d', '', city_guess)
            city_guess = re.sub(';', '', city_guess)
            print(city_guess)
            self.city = city_match(city_guess.strip())
            
        street = street.partition('(')[0]

        print 'Matching: ' + street + ', ' + self.city

        # Get all valid addresses within the matches cities.
        addr_options = self.streets_table[self.streets_table['City'] == self.city]

        # Seperate street number from street name.
        sepr = re.search(street_patt, street)

        if sepr:
            stnum = sepr.group(1).strip()
            stnam = sepr.group(2).strip()
        else:
            stnum = ''
            stnam = street.strip()
        
        stnam = re.sub('Av$', 'Ave', stnam)

        if stnam == '':
            print('EMPTY STREET')
            return
        print(stnam)
        # Look for best fuzzy matches with a score > cutoff.
        t1 = time.time()
        if stnam.upper() in addr_options['Street'].tolist():
            print('Perfect match')
            street_matches = (stnam.upper(), 100.0)
        else:
            street_matches = process.extractOne(stnam, addr_options['Street'], scorer=street_scorer)
        if not street_matches:
            print('No match')
            return
        street, score = (street_matches[0],street_matches[1]) # removes the third dummy value that sometimes shows up in the tuple
        t2 = time.time()
        print('Search time: ' + str(round(t2-t1, 6)) + ' s')

        # Add to addr_matches if score reaches cutoff:
        if score < cutoff:
            print('SCORE LESS THAN CUTOFF: ' + street + ',' + str(score) + ',' + self.city)
            return
        else:
            addr = stnum + ' ' + street
            addr_match = (addr, self.city, score)
            print(addr_match)
            self.addr_matches.append(addr_match)



#address = Address(street='79 Bway', city='PROVIDENCE')
#address.set_city_matches(cutoff=80)
#address.set_addr_matches(cutoff=45, limit=1)
#print('Address matches: ')
#print(address.addr_matches)





