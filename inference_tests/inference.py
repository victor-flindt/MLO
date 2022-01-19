# the following script will be for inference testing on a trained model found in model folders
# the script will provide the model with some text and the model will predict the sentiment of the text
from pathlib import Path
from transformers import BertTokenizer

import pandas as pd
import torch
#from src.data.clean_dataset import clean_data
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, SequentialSampler

def inference_dataloader(sentence_max_length, raw_data):
    
    data=raw_data
    sentences = data.input.values
    # Load the BERT tokenizer.
    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    print('BERT tokenizer loaded!')

    max_len = 0

    # For every sentence...
    for sent in sentences:

        # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
        input_ids = tokenizer.encode(sent, add_special_tokens=True)

        # Update the maximum sentence length.
        max_len = max(max_len, len(input_ids))

    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in sentences:
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = sentence_max_length,           # Pad & truncate all sentences.
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                    )
        
        # Add the encoded sentence to the list.    
        input_ids.append(encoded_dict['input_ids'])
        
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)


    # Combine the training inputs into a TensorDataset.
    dataset = TensorDataset(input_ids, attention_masks)

    # Create a 90-10 train-validation split.

    # Calculate the number of samples to include in each set.
    val_size = len(dataset)

    # Divide the dataset by randomly selecting samples.
    val_dataset = dataset

    validation_dataloader = DataLoader(
                val_dataset, # The validation samples.
                sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
                batch_size = 1 # Evaluate with this batch size.
            )
    return validation_dataloader


## getting input from the termnial as the first argument after the file name
input_tweet = ["i am extremely angry i hate everyone"]
#input_tweet = sys.argv[1]

columns = ["input"]
df = pd.DataFrame(input_tweet,columns = columns)

tweet_loader=inference_dataloader(len(input_tweet)+1,df)

path = Path("..\\models\\model.pt")
device = torch.device("cpu")
model = torch.load(path,map_location=torch.device('cpu'))
model.eval()

for step,batch in enumerate(tweet_loader):
    b_input_ids = batch[0].to(device)
    b_input_mask = batch[1].to(device)

    result = model(b_input_ids, 
            token_type_ids=None,
            attention_mask=b_input_mask,
            return_dict=True)

print(result[0])
# print(max(a,key=itemgetter(1))[0])

print("Sentiment Classification scale: \n Extremly negative \t [1] \n Negative \t\t [2] \n Neutral \t\t [3] \n Posetive \t\t [4] \n Extremly posetive \t [5]")
print(f'Input Tweet sentiment classification: {torch.argmax(result[0])+1}')