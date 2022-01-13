import re
import pandas as pd
from pathlib import Path
import os

from transformers.tokenization_utils import Trie

from src.data import _RAW_DATA_PATH

# This file is purely for cleaning tweetlike input data, and is made such that
# it can be reused on future tweet-like datasets
# The file make_datasetpy will handle the actual dataloader.
def clean_data():

    raw_data = pd.read_csv(_RAW_DATA_PATH, encoding='latin-1')

    # list of columns which should be rem oved
    waste_col = ['UserName', 'ScreenName', 'Location', 'TweetAt']
    # removing columns
    raw_data = raw_data.drop(waste_col, axis=1)

    # renaming columns to more saying names. 
    raw_data = raw_data.rename(columns={"OriginalTweet": "input", "Sentiment": "label"})

    # converts tweets to lower case
    raw_data["input"] = raw_data["input"].str.lower()

    # Setting up tag for Re implementation
    tags = r"@\w*"

    # replace matches with tags with ''
    raw_data['input'] = raw_data['input'].str.replace(tags, '',regex=True)

    # removing links from the input
    # Regex expression found https://stackoverflow.com/questions/3809401/what-is-a-good-regular-expression-to-match-a-url
    links = r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)" 
    raw_data['input'] = raw_data['input'].str.replace(links, '',regex=True)

    # remove all hashtags
    hashtags = r'#[^#\s\n]+'
    raw_data['input'] = raw_data['input'].str.replace(hashtags, '',regex=True)


    # remove everything that is not a letter, not a white space and note a /
    # for situations were new/used etc. are used

    special_chars = r"[^a-zA-Z\s/]"
    raw_data['input'] = raw_data['input'].str.replace(special_chars, '',regex=True)

    # substiuting / with white space
    forward_slash = r"/"
    raw_data['input'] = raw_data['input'].str.replace(forward_slash, ' ',regex=True)

    raw_data['label'] = raw_data['label'].replace(["Extremely Negative", "Negative", "Neutral", "Positive", "Extremely Positive"], [0,1,2,3,4])

    # for saving the data to a csv for further processing.
    # raw_data.to_csv(f'{Path(os.getcwd()).parents[1]}//data//processed//cleaned_tr.csv')
    
    return raw_data