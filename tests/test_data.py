import pytest
# from src.data import clean_dataset
# from  import make_dataset


#########################################---TESTING---##################################################
# the next portion of the script will be testing some key point about the data, such as dimensionlity, #
# size, label representation, etc. The data loaded will not be local but will be loaded from pytorch   #
# module, this is done since the self made dataloader is to time consuming and is not the point of the #
# exercise in the first place.                                                                         #
########################################################################################################

def test_check():
    # data = make_dataset.dataset()
    assert 1 != 1, f"testen virker du kan gå i seng "
    

# @pytest.mark.parametrize("dim1,dim2,expected", [(28, 28, 784), (10, 10, 100), (32, 32, 1024)])
# def test_dim_flatten(dim1, dim2, expected):
#     temp_tens = torch.rand(dim1, dim2)
#     assert torch.flatten(temp_tens).size(dim=0) == expected, f"input dimension: [{dim1}, {dim2}] did not meet expecteed output size after flattening of {expected}."
