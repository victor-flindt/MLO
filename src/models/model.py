from torch import nn
from transformers import BertModel
PRE_TRAINED_MODEL_NAME = 'bert-base-cased'

class SentimentClassifier(nn.Module):

  def __init__(self, n_classes, p_dropout):

    super(SentimentClassifier, self).__init__()

    self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)

    self.drop = nn.Dropout(p=p_dropout)

    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

  def forward(self, input_ids, attention_mask):

    print(type(input_ids))
    print(type(attention_mask))
    _, pooled_output = self.bert(

      input_ids=input_ids,

      attention_mask=attention_mask

    )
    print(pooled_output.shape())
    output = pooled_output

    return self.out(output)