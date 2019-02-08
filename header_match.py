import pandas as pd
from fuzzywuzzy import fuzz
import string
import pickle as pkl
import numpy as np

TRUE_CUTOFF = 3150
THRESHOLD = 85
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
    if cnt > 3 or red_flag: h = ""
    return h
    
def assign_clean(D):
    return [clean_header(h) for h in D.Headers]

def score(string1, string2):
    # Scores the fuzzy match between the ocr header and the true header
    return fuzz.ratio(string1,string2)

def match(headers, true_headers, map_dict):
    # Matches ocr headers to true headers
    for header in headers:
        if header not in map_dict:
            score_list = [(score(header, true_header), true_header) for true_header in true_headers]
            sorted_score = sorted(score_list, key = lambda tup: tup[0], reverse=True)
            score_tuple = sorted_score[0]
            if score_tuple[0] > 90:
                print("LOG: Matched! " + header + " and " + score_tuple[1] + " with score " + str(score_tuple[0]))
                map_dict[header] = (score_tuple[0], score_tuple[1], "TRUE")
            else:
                print("LOG: Not matched, " + header + " and " + score_tuple[1] + " with score " + str(score_tuple[0]))
                map_dict[header] = (score_tuple[0], "no_header", "FALSE")
    return map_dict

def calculate_scores(df):
    l = len(df.headers)
    score_matrix = np.zeros((l,l))
    i = 0
    for h in df.headers:
        score_matrix[i, :] = df.headers.apply(score, args=(h,)).values
        print("Row number: {} of {}".format(i, l - 1))
        i += 1
    return score_matrix

def remove_repeat(df, scores):
    prelist = list(df.headers)
    i = 0
    for _ in df.headers:
        j = 0
        for removal in scores[i, :]:
            if removal and j > i:
                try:
                    df = df[df["headers"] != prelist[j]]
                    print("LOG: " + prelist[j] + " deleted!")
                except KeyError:
                    print("LOG: " + prelist[j] + " Already deleted")
            j += 1
        i += 1
    return df

def assign_matched(D, map_dict):
    matched = []
    for h in D.headers:
        if h in map_dict: 
            matched.append(map_dict[h][1])
        else: 
            print("Known: ", h, " not in map_dict")
            matched.append(h)
    return matched

def assign_score(D, map_dict):
    scores = []
    for h in D.headers:
        if h in map_dict: 
            scores.append(map_dict[h][0])
        else:
            print("Known: ", h, " not in map_dict")
            scores.append(np.nan)
    return scores

def assign_bool(D, map_dict):
    is_matched = []
    for h in D.headers:
        if h in map_dict:
            is_matched.append(map_dict[h][2])
        else:
            print("Known: ", h, " not in map_dict")
            is_matched.append("TRUE")
    return is_matched



def generate_dict(df, true_headers, unsure_headers):
    map_dict = {}
    df = df[df.headers.map(lambda h: (len(h) < 150) and (len(h) > 2) and (h is not ""))]
    df = df.drop_duplicates("headers").reset_index(drop=True).sort_values("count", ascending=False)

    map_dict = match(unsure_headers, true_headers, map_dict)
    pkl.dump(map_dict, open('trueheaders_match_dict.pkl', 'wb'))

    return map_dict


def header_match(df, map_dict, unsure_headers):
    df = df[df.headers.map(lambda h: (len(h) < 150) and (len(h) > 2) and (h is not ""))]
    df = df.drop_duplicates("headers").reset_index(drop=True).sort_values("count", ascending=False)
    
    df = df.assign(headers=assign_clean).drop_duplicates("headers")
    df = df[df.headers.map(lambda h: (len(h) < 150) and (len(h) > 2) and (h is not ""))]
    df = df.reset_index(drop=True).sort_values("count", ascending=False)
    df = df.assign(
        matched = lambda D: assign_matched(D, map_dict),
        score = lambda D: assign_score(D, map_dict), 
        is_matched = lambda D: assign_bool(D, map_dict)).sort_values("count", ascending=False)

    known = df.loc[df.matched != "no_header"]
    known = known.reset_index(drop=True).drop_duplicates("headers").sort_values("count", ascending=False)
    unknown = df.loc[df.matched == "no_header"]
    unknown = unknown.reset_index(drop=True).drop_duplicates("headers").sort_values("count", ascending=False)

    scores = calculate_scores(unknown)
    removal_matrix = (scores > THRESHOLD) & (scores < 100)

    internal_unmatched = remove_repeat(unknown, removal_matrix)
    internal_unmatched = internal_unmatched.reset_index(drop=True)

    return known, internal_unmatched


def test_matcher():
    raw = pd.read_csv("RankedHeaders.csv")[["count", "Headers"]].dropna().assign(headers=assign_clean)
    df = raw.drop_duplicates("headers")
    unsure_headers = list(df[TRUE_CUTOFF:].headers)
    map_dict = pkl.load(open('trueheaders_match_dict.pkl', 'rb'))
    known, unmatched = header_match(df, map_dict, unsure_headers)
    print("test_passed")
    print(known)
    print(unmatched)

def test_map_dict():
    raw = pd.read_csv("RankedHeaders.csv")[["count", "Headers"]].dropna().assign(headers=assign_clean)
    df = raw.drop_duplicates("headers")
    true_headers = list(df[:TRUE_CUTOFF].headers)
    unsure_headers = list(df[TRUE_CUTOFF:].headers)
    map_dict = generate_dict(df, true_headers, unsure_headers)
    print(map_dict)
    print("test passed")

test_matcher()