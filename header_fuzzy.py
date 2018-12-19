import pandas as pd
import pickle as pkl
from fuzzywuzzy import fuzz
import string

def preprocess(df):
    # Preprocesses the ocr header
    try:
        map_dict = pkl.load(open('header_match_dict.pkl', 'rb'))
    except:
        print("WARNING: pickle load failed")
        map_dict = {}
    header_list = list(set(df["Header"]) - set(map_dict.keys()))
    return header_list, map_dict


common_errors = {
    "0": "O",
    "1": "l",
    "5": "S",
    "8": "B"
}


def clean_header(header):
    cleaned_word = []
    for let in header.upper().split():
        if let in common_errors:
            let = common_errors[let]
        cleaned_word.append(let)
    return string.join(cleaned_word, "")

def clean(header_list):
    # Cleans the ocr header
    new_list = [clean_header(header) for header in header_list]
    return new_list


def score(string1, string2):
    print(string1, string2)
    # Scores the fuzzy match between the true header and the ocr header
    return fuzz.partial_ratio(string1.upper(),string2.upper())


def match(headers, true_headers, map_dict):
    # Matches ocr headers to true headers
    for header in headers:
        if header not in map_dict:
            score_list = [(score(header, true_header), true_header) for true_header in true_headers]
            sorted_score = sorted(score_list, key = lambda tup: tup[0], reverse=True)
            score_tuple = sorted_score[0]
            map_dict[header] = score_tuple
    return map_dict


def driver(df):
    # Drives the header matching code
    true_headers = list(pd.read_csv("RankedHeaders.csv", usecols=[2]).dropna().rename(lambda x: "true_headers", axis="columns")["true_headers"])
    hl, mpd = preprocess(df)
    print("LOG: Obtained headers")
    cleaned_hl = clean(hl)
    print("LOG: Cleaned headers")
    updated_mpd = match(cleaned_hl, true_headers, mpd)
    print("LOG: Match completed")
    pkl.dump(updated_mpd, open('header_match_dict.pkl', 'wb'))
    def assign_score(D):
        return [updated_mpd[clean_header(h)][0] for h in D.Header]
    def assign_header(D):
        return [updated_mpd[clean_header(h)][1] for h in D.Header]
    updated_df = df.assign(score=assign_score, true_header=assign_header)
    return updated_df


print(driver(pd.read_csv("FOutput.csv"))[["Header", "true_header", "score"]])