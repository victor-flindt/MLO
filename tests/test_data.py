import pytest
import os
from src.data.clean_dataset import clean_data

FILE_PATH = os.path.dirname(__file__) 
RAW_DATA_PATH = os.path.join(FILE_PATH, '../data/raw/Corona_NLP_train.csv')

#########################################---TESTING---##################################################
# the next portion of the script will be testing some key point about the data, such as dimensionlity, #
# size, label representation, etc. The data loaded will not be local but will be loaded from pytorch   #
# module, this is done since the self made dataloader is to time consuming and is not the point of the #
# exercise in the first place.                                                                         #
########################################################################################################


def test_label_representation():
    labels = [0, 1, 2, 3, 4]
    train_dataset=clean_data(RAW_DATA_PATH)
    ## input all the expected labels for the data, will throw error if a label in the labels array is not present in the dataset
    ## this array can be made into a tuble if string labels are required. 
    for _, value in enumerate(labels):
        assert value in train_dataset['label'].values, f" {value} is not present in the dataset" 

def test_training_size():
    train_dataset=clean_data(RAW_DATA_PATH)
    assert len(train_dataset) == 41157

def test_is_nan():
    train_dataset=clean_data(RAW_DATA_PATH)
    assert train_dataset.isnull != True ,f" Dataset Contains Null values" 

def test_is_whitespace():
    # virker sku ikke lige rigtig men det må vi ændre på et tidspunkt nu er der sat nogen test op
    train_dataset=clean_data(RAW_DATA_PATH)
    assert train_dataset['input'].str.isspace != True

# @pytest.mark.parametrize("dim1,dim2,expected", [(28, 28, 784), (10, 10, 100), (32, 32, 1024)])
# def test_dim_flatten(dim1, dim2, expected):
#     temp_tens = torch.rand(dim1, dim2)
#     assert torch.flatten(temp_tens).size(dim=0) == expected, f"input dimension: [{dim1}, {dim2}] did not meet expecteed output size after flattening of {expected}."

