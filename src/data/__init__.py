import os 

_FILE_PATH = os.path.dirname(__file__)  # root of file (models)
_SRC_PATH = os.path.dirname(_FILE_PATH)  # root of models (src)
_PROJECT_ROOT = os.path.dirname(_SRC_PATH) # root of src (project root)

_RAW_DATA_PATH = os.path.join(_PROJECT_ROOT, "data/raw/Corona_NLP_train.csv")