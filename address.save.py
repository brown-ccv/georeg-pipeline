import pandas as pd
import re

from fuzzywuzzy import fuzz, process


ri_streets_table = pd.read_pickle('stZipCty')
street_patt = re.compile(r"(\d+)(.+)")

def cases(regexstr, strlist):
    regex = re.compile('(' + regexstr + ')', re.IGNORECASE)
    return [m.group(0) for lv in strlist for m in [regex.search(lv)] if m]

def abbreviation_scorer(str1, str2):
    if str1[0] == str2[0]:
        fscore = 100.0
    elif str1[0].lower() == str2[0].lower():
        fscore = 75.0
    else:
        fscore = 25.0
    rscore = (fscore + 3 * fuzz.ratio(str1, str2)) / 4.0
    return rscore

class Address(object):
    """Class to hold address information."""

    def __init__(self, street=None, city=None, streets_table=ri_streets_table):

        self.city = city
        self.street = street
        self.streets_table = streets_table
        self.city_matches = []
        self.addr_matches = []

    def set_city_matches(self, cutoff):
        """Find fuzzy matches between city and cities in streets_table."""

        if (not self.city) or (self.city == 'N/A') or ('No Match' in self.city):
            self.city_matches = []
        else:
            valid_cities = self.streets_table['City'].unique()
            matches = process.extractWithoutOrder(self.city, valid_cities,
                                                  scorer=fuzz.partial_ratio,
                                                  score_cutoff=cutoff)

            self.city_matches = [city for city, score in matches]

    def set_addr_matches(self, cutoff, limit):
        """Find (street, city, score) fuzzy matches for addresses in streets_table."""

        self.addr_matches = []

        if not self.city_matches:
            return

        street = self.street.strip()

        print 'Matching: ' + street

        # Get all valid addresses within the matches cities.
        addr_options = self.streets_table[self.streets_table['City'].isin(self.city_matches)]

        # Seperate street number from street name.
        sepr = re.search(street_patt, street)

        if not sepr:
            return

        stnum = sepr.group(1).strip()
        stnam = sepr.group(2).strip()
        # print '********************************'
        # print stnam

        # Look for best fuzzy matches with a score > cutoff.
        street_matches = process.extractBests(stnam, addr_options['Street'],
                                                scorer=fuzz.ratio,
                                                score_cutoff=cutoff,
                                                limit=limit)

        # Look for streets that match the street with only deleted letters
        #street_regex_matches = addr_options[addr_options['Street'].str.match('[a-zA-Z]*'.join(list(stnam)) + '.*', case=False)]
        #print(street_regex_matches)
        if not street_regex_matches.empty:
            street_matches = process.extractBests(stnam, street_regex_matches['Street'],
                                                scorer=abbreviation_scorer,
                                                score_cutoff=30,
                                                limit=limit)
        else:
            street_matches = process.extractBests(stnam, addr_options['Street'],
                                                scorer=abbreviation_scorer,
                                                score_cutoff=cutoff,
                                                limit=limit)


        if not street_matches:
            #print 'No match'
            return

        # removes the third dummy value that sometimes shows up in the tuple
        street_matches = [(li[0],li[1]) for li in street_matches]

        # print(street_matches)

        # Add to addr_matches.
        for street,score in street_matches:
            # print streetls
            # print 'Street Match Conf Score: ' + str(score)
            # print(street)
            matches = addr_options[(addr_options['Street'] == street) & (addr_options['City'] == self.city)]
            # print matches
            firstRun = True
            for ix, match in matches.iterrows():
                
                if firstRun == True:
                    addr = stnum + ' ' + match['Street']
                    addr_match = (addr, match['City'], score)
                    print addr_match
                    self.addr_matches.append(addr_match)
                    firstRun = False



#address = Address(street='79 Newman', city='EAST PROVIDENCE')
#address.set_city_matches(cutoff=80)
#address.set_addr_matches(cutoff=65, limit=1)
#print('Address matches: ')
#print(address.addr_matches)






