from os import truncate
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from src.data.clean_dataset import clean_data
from transformers import BertTokenizer
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# this file will handle making of the actual dataset
# which will consist of the processed data from the 
# clean_dataset.py file.
class GPReviewDataset(Dataset):

  def __init__(self, reviews, targets, tokenizer, max_len):
    self.reviews = reviews
    self.targets = targets
    self.tokenizer = tokenizer
    self.max_len = max_len

  def __len__(self):
    return len(self.reviews)

  def __getitem__(self, item):
    review = str(self.reviews[item])
    target = self.targets[item]
    encoding = self.tokenizer.encode_plus(
      review,
      add_special_tokens=True,
      max_length=self.max_len,
      return_token_type_ids=False,
      pad_to_max_length=True,
      return_attention_mask=True,
      return_tensors='pt',
    )
    return {
      'review_text': review,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'targets': torch.tensor(target, dtype=torch.long)
    }

def make_dataset():
    data=clean_data()
    df_train, df_test = train_test_split(
                    data,
                    test_size=0.1,
                    random_state=RANDOM_SEED
                )

    return df_train,df_test

# creating dataloader
def create_data_loader(df, tokenizer, max_len, batch_size):
  ds = GPReviewDataset(
    reviews=df.input.to_numpy(),
    targets=df.label.to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len
  )
  return DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=1
  )

def dataset():
    BATCH_SIZE = 16
    PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

    df_train,df_test=make_dataset()

    BATCH_SIZE = 16
    MAX_LEN = 90
    train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
    test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

    # data = next(iter(train_data_loader))
    # print(data.keys())
    # print(data['input_ids'].shape)
    # print(data['attention_mask'].shape)
    # print(data['targets'].shape)

    # print(data['input_ids'][1])
    # print(data['review_text'][1])
    # print(data['targets'][1])

    # for batch_idx,(afaf,inputs,afaf,labels) in enumerate(next(iter(train_data_loader))):
    #     if batch_idx >= 5:
    #         break
    #     else:
    #         print(inputs)

    return train_data_loader, test_data_loader

# if __name__ == "__main__":
#   dataset()
