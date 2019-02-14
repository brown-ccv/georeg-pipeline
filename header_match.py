import pandas as pd
from fuzzywuzzy import fuzz
import string
import pickle as pkl
import numpy as np


# Global constants, should be in input_Params
THRESHOLD = 85

# can be in input_params, but this is stuff that needs to be replaced or removed.
replace_char = ["*", "&", "%", "/", "\\"]
strip_char = ["'", "-", ".", "!", ":", ";"]
num_char =  ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]
red_flag_char = [" AVE ", " ST ", " AV ", " BLVD ", " BLV ", " RD ", " DR "]
common_errors = {
    "0": "O",
    "1": "l",
    "5": "S",
    "8": "B"
}

def clean_header(h):
    # cleans the header
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
    try:
        fuzz.ratio(string1,string2)
    except:
        print(string1)
        print(string2)
        exit()
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
                print("LOG: Matched! " + header + " and " + score_tuple[1] + " with score " + str(score_tuple[0]))
                map_dict[header] = (score_tuple[0], score_tuple[1], "TRUE")
            else:
                print("LOG: Not matched, " + header + " and " + score_tuple[1] + " with score " + str(score_tuple[0]))
                map_dict[header] = (score_tuple[0], "no_header", "FALSE")
    return map_dict

def calculate_scores(df):
    # find the total length
    l = len(df["clean_headers"])
    # initialize the matrix
    score_matrix = np.zeros((l,l))
    i = 0
    for h in df["clean_headers"]:
        # create the scored matrix
        score_matrix[i, :] = df["clean_headers"].apply(score, args=(h,)).values
        print("Row number: {} of {}".format(i, l - 1))
        i += 1
    return score_matrix

def remove_repeat(df, scores):
    # remove the repeated unknown headers
    prelist = list(df["clean_headers"])
    i = 0
    for _ in df["clean_headers"]:
        j = 0
        for removal in scores[i, :]:
            if removal and j > i: # the j > i ensures that only one copy of the header to header match is removed. 
                #   A B C
                # A . . .
                # B . . .
                # C . . .
                # AC <==> CA, only one is removed

                # check if can be removed, if yes, remove.
                try:
                    df = df[df["clean_headers"] != prelist[j]]
                    print("LOG: " + prelist[j] + " deleted!")
                except KeyError:
                    print("LOG: " + prelist[j] + " Already deleted")
            j += 1
        i += 1
    return df


# Functions to assign to the dataframe

def assign_matched(D, map_dict):
    matched = []
    for h in D["clean_headers"]:
        if h in map_dict: 
            matched.append(map_dict[h][1])
        else: 
            print("Known: ", h, " not in map_dict")
            matched.append(h)
    return matched

def assign_score(D, map_dict):
    scores = []
    for h in D["clean_headers"]:
        if h in map_dict: 
            scores.append(map_dict[h][0])
        else:
            print("Known: ", h, " not in map_dict")
            scores.append(np.nan)
    return scores

def assign_bool(D, map_dict):
    is_matched = []
    for h in D["clean_headers"]:
        if h in map_dict:
            is_matched.append(map_dict[h][2])
        else:
            print("Known: ", h, " not in map_dict")
            is_matched.append("TRUE")
    return is_matched



# driver function to create the map_dict
def generate_dict(df, true_headers):
    
    map_dict = {}

    df = df.drop_duplicates("Header").dropna().assign(clean_headers=assign_clean)
    df = df[df["clean_headers"].map(lambda h: (len(h) < 150) and (len(h) > 2) and (h != ""))].reset_index(drop=True)

    unsure_headers = list(df['clean_headers'])

    map_dict = match(unsure_headers, true_headers, map_dict)
    pkl.dump(map_dict, open('trueheaders_match_dict.pkl', 'wb'))

    return map_dict

# driver function to header match given a map_dict
def match_headers(df, map_dict):

    df = df.drop_duplicates("Header").dropna().assign(clean_headers=assign_clean)
    df = df[df["clean_headers"].map(lambda h: (len(h) < 150) and (len(h) > 2) and (h != ""))].reset_index(drop=True)

    df = df.assign(
        matched = lambda D: assign_matched(D, map_dict),
        score = lambda D: assign_score(D, map_dict), 
        is_matched = lambda D: assign_bool(D, map_dict))

    known = df.loc[df.matched != "no_header"].reset_index(drop=True)
    unknown = df.loc[df.matched == "no_header"].reset_index(drop=True)

    scores = calculate_scores(unknown)
    removal_matrix = (scores > THRESHOLD) & (scores < 100)
    internal_unmatched = remove_repeat(unknown, removal_matrix)
    internal_unmatched = internal_unmatched.reset_index(drop=True)

    return known, internal_unmatched, df


