import pandas as pd
import re
import pickle as pkl

from address import Address
from multiprocessing import Pool

# Matches string to a database of streets

def search_street(query):
    address = Address(street=query, city='PROVIDENCE')
    address.set_addr_matches(cutoff=80, limit=1)
    rtuple = address.addr_matches[0]
    return query,rtuple[0],rtuple[1],rtuple[2]

def streetMatcher(dataFrame):
    final = []
    mistakes = []
    #dataFrame = pd.read_pickle('ccities')
    #streetTable()
    street_dict = pkl.load(open('street_dict.pkl', 'rb'))
    #street_dict = {}

    search_list = []

    # Check to see if any street strings are not already in the dictionary.
    street_set = set(dataFrame['Street'])
    search_list = list(street_set - set(street_dict.keys()))

    # If there are street strings missing from the dictionary, do the address matching, and add them.
    if search_list:
        pool = Pool(6)
        search_results = pool.map(search_street, search_list)
        for query,addr,city,score in search_results:
            street_dict[query] = (addr, city, score)

    #Get each row of dataframe with corrected cities
    for row in dataFrame.itertuples():

        street = row.Street

        # Get valid addresses from city and street info.
        addr, city, score = street_dict[street]
        if city == 'N/A':
            mistakes.append({
                'Street': addr,
                'Drop_Reason': score,
                'File_List': row.File_List,
                'Text': row.Text,
                })
        else:
            final.append({
                'Address': addr,
                'City': city,
                'Conf_Score': score,
                'Header': row.Header,
                'File_List': row.File_List,
                'Text': row.Text,
                'Company_Name': row.Company_Name
            })    

    final = pd.DataFrame(final)
    drops = pd.DataFrame(mistakes)
    drops.to_csv('drops_address.csv', sep = ',')
    pkl.dump(street_dict, open('street_dict.pkl', 'wb'))

    return final