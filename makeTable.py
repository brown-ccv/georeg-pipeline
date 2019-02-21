import pandas as pd

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


def streetTable():
    """Create DataFrame with streets, Zip Codes, and Cities."""
    print('Creating Zipcode Table')
    #.csv with streets and corresponding zip codes
    street_df = pd.read_csv('streets_by_zip_code.csv', dtype = str)
    zipcode_df = pd.read_csv('zip_code_database.csv', dtype = str)
    zipcode_df = zipcode_df[zipcode_df['state'] == 'RI']
    street_df.columns = ['Street', 'Zip_Code']

    street_df['Street'] = street_df['Street'].apply(lambda x: ' '.join(substitute_directions(x.split())))

    zip_dict = {row.zip:[x for x in ([row.primary_city] + str(row.acceptable_cities).split(',') + str(row.unacceptable_cities).split(',')) if x!= 'nan'] for row in zipcode_df.itertuples()}

    #Find city corresponding to each zip code.
    street_df['City_List'] = street_df['Zip_Code'].apply(lambda x: zip_dict[x])


    expanded_street_list = []
    for index,row in street_df.iterrows():
        for city in row['City_List']:
            new_row = row.copy()
            new_row['City'] = city.upper()
            expanded_street_list.append(new_row.copy())

    matched_street_df = pd.DataFrame(expanded_street_list)[['Street', 'Zip_Code', 'City']]

    matched_street_df.to_csv('StreetZipCity.csv')
    #return matched_street_df


streetTable()


