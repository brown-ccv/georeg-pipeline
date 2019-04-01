import pandas as pd
from fuzzywuzzy import fuzz
import string
import pickle as pkl
import numpy as np
import time

# Global constants, should be in input_Params
THRESHOLD = 85

# can be in input_params, but this is stuff that needs to be replaced or removed.
replace_char = ["*", "%", "/", "\\"]
strip_char = ["'", "-", ".", "!", ":", ";"]
num_char =  ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]
red_flag_char = [" AVE ", " ST ", " AV ", " BLVD ", " BLV ", " RD ", " DR "]
common_errors = {
    "0": "O",
    "1": "l",
    "5": "S",
    "8": "B"
}

def clean_header(h_raw):
    # cleans the header
    h = h_raw.partition(' (')[0]
    red_flag = False
    for s in replace_char:
        h = h.replace(s, "")
    for s in strip_char:
        h = h.strip(s)
    cnt = 0
    hl = []
    for c in list(h):
        if c in common_errors: c = common_errors[c]
        if c in num_char: cnt += 1
        hl.append(c)
    h = ''.join(hl).upper()
    for rf in red_flag_char:
        if rf in h: red_flag = True
    if cnt > 3 or red_flag: 
        h = ""
    return h
    
def assign_clean(D):
    # assigns to dataframe
    return [clean_header(h) for h in D["Header"]]

def score(string1, string2):
    # Scores the fuzzy match between the ocr header and the true header
    return fuzz.ratio(string1,string2)

def match(headers, true_headers, map_dict):
    # Matches ocr headers to true headers
    for header in headers:
        if header not in map_dict:
            # score every true header with current unknown header
            score_list = [(score(header, true_header), true_header) for true_header in true_headers]
            # sort the list
            sorted_score = sorted(score_list, key = lambda tup: tup[0], reverse=True)
            # pick the best tuple
            score_tuple = sorted_score[0]
            # if above threshold, match
            if score_tuple[0] > THRESHOLD:
                #print("LOG: Matched! " + header + " and " + score_tuple[1] + " with score " + str(score_tuple[0]))
                map_dict[header] = (score_tuple[0], score_tuple[1], "TRUE")
            else:
                #print("LOG: Not matched, " + header + " and " + score_tuple[1] + " with score " + str(score_tuple[0]))
                map_dict[header] = (score_tuple[0], "no_header", "FALSE")
    return map_dict


# driver function to create the map_dict
def generate_dict(df, true_headers):

    map_dict = {}
    df = df.drop_duplicates("Header").dropna().assign(clean_headers=assign_clean)
    df = df[df["clean_headers"].map(lambda h: (len(h) < 150) and (len(h) > 2) and (h != ""))].reset_index(drop=True)

    unsure_headers = list(df["clean_headers"])
    map_dict = match(unsure_headers, true_headers, map_dict)
    pkl.dump(map_dict, open('header_match_dict.pkl', 'wb'))

    return map_dict

# driver function to header match given a map_dict
def match_headers(df, map_dict):

    #df = df.drop_duplicates("Header").dropna().assign(clean_headers=assign_clean)
    t1 = time.time()
    header_dict = {}
    print(len(set(df['Header'])))
    for header in set(df['Header']):
        cleaned_header = clean_header(header)
        if cleaned_header in map_dict.keys():
            if map_dict[cleaned_header][1] != "no_header":
                header_dict[header] = map_dict[cleaned_header][1]
            else:
                header_dict[header] = cleaned_header
        else:
            header_dict[header] = cleaned_header
    t2 = time.time()
    print('clean header assigning time: ' + str(round(t2-t1,3)) + ' s')
    print(set(header_dict.keys()) - set(df['Header']))

    t1 = time.time()
    header_list = []
    for row in df.itertuples():
        raw_header = row.Header
        matched_header = header_dict[raw_header]
        header_list.append(matched_header)
    t2 = time.time()
    print('assigning time: ' + str(round(t2-t1, 3)) + ' s')
    df = df.assign(clean_header=header_list)

    return df





