import pandas as pd 
import numpy as np
import re
from pathlib import Path
import os 
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from transformers.file_utils import replace_return_docstrings

## This file is purely for cleaning tweetlike input data, and is made such that it can be reused on future tweet-like datasets
## The file make_datasetpy will handle the actual dataloader. 



PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)


raw_data = pd.read_csv(f'{Path(os.getcwd()).parents[1]}\\data\\raw\\Corona_NLP_train.csv',encoding='latin-1')

## list of columns which should be removed
waste_col = ['UserName', 'ScreenName','Location','TweetAt']
## removing columns 
raw_data = raw_data.drop(waste_col, axis = 1)

# renaming columns to more saying names. 
raw_data=raw_data.rename(columns={"OriginalTweet": "input", "Sentiment": "label"})

## converts tweets to lower case
raw_data["input"] = raw_data["input"].str.lower()

## Setting up tag for Re implementation
tags = r"@\w*"

#replace matches with tags with ''
raw_data['input'] = raw_data['input'].str.replace(tags, '')

#removing links from the input
## Regex expression found https://stackoverflow.com/questions/3809401/what-is-a-good-regular-expression-to-match-a-url
links = r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)" 
raw_data['input'] = raw_data['input'].str.replace(links, '')

# remove all hashtags 
hashtags = r'#[^#\s\n]+'
raw_data['input'] = raw_data['input'].str.replace(hashtags, '')


## remove everything that is not a letter, not a white space and note a /
## for situations were new/used etc. are used

special_chars = r"[^a-zA-Z\s/]"
raw_data['input'] = raw_data['input'].str.replace(special_chars, '')

## substiuting / with white space
forward_slash = r"/"
raw_data['input'] = raw_data['input'].str.replace(forward_slash, ' ')

raw_data['label'] = raw_data['label'].replace(["Extremely Negative", "Negative", "Neutral", "Positive", "Extremely Positive"], [1, 2, 3, 4, 5])

# for index, value in enumerate(raw_data['label']):
#     if value == "Extremely Negative":
#         raw_data[index]['label'].replace()
print(raw_data)
##raw_data.to_csv(f'{Path(os.getcwd()).parents[1]}//data//processed//cleaned_tr_.csv') 


## testing everything is as it should be
# sample_txt=raw_data.iloc[0]['input']

# tokens = tokenizer.tokenize(sample_txt)
# token_ids = tokenizer.convert_tokens_to_ids(tokens)

# print(f' Sentence: {sample_txt}')
# print(f'   Tokens: {tokens}')
# print(f'Token IDs: {token_ids}')



