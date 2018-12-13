import pandas as pd
import pickle as pkl
from fuzzywuzzy import fuzz


def preprocess(df):
    # Preprocesses the ocr header
    try:
        map_dict = pkl.load(open('header_match_dict.pkl', 'rb'))
    except:
        map_dict = {}
    header_list = list(set(df["header"]) - set(map_dict.keys()))
    return header_list, map_dict


def clean(header_list):
    # Cleans the ocr header
    return header_list


def score(string1, string2):
    # Scores the fuzzy match between the true header and the ocr header
    words1 = string1.upper().split()
    words2 = string2.upper().split()
    # if words1[-1] != words2[-1]:
    #     if words2[-1] in sts:
    #         words2 = words2[:-1]
    #     if words1[-1] in sts:
    #         words1 = words1[:-1]
    word1 = sorted(words1, key=len, reverse=True)[0]
    word2 = sorted(words2, key=len, reverse=True)[0]
    return (fuzz.ratio(word1, word2) + fuzz.ratio(' '.join(words1), ' '.join(words2)))/2


def match(ocr_headers, true_headers, map_dict):
    # Matches ocr headers to true headers
    for ocr_header in ocr_headers:
        score_list = [(score(ocr_header, true_header), true_header) for true_header in true_headers]
        sorted_score = sorted(score_list, key = lambda tup: tup[0])
        score_tuple = sorted_score[0]
        map_dict[ocr_header] = score_tuple
    return map_dict


def driver(df):
    # Drives the header matching code
    true_headers = list(pd.read_csv("true_headers.csv")["true_headers"]) # or to_list() ?
    hl, mpd = preprocess(df)
    cleaned_hl = clean(hl)
    updated_mpd = match(cleaned_hl, true_headers, mpd)
    updated_df = df.assign(score=lambda h: updated_mpd[h][0], true_header=lambda h: updated_mpd[h][1])
    pkl.dump(updated_mpd, open('header_match_dict.pkl', 'wb'))
    return updated_df



