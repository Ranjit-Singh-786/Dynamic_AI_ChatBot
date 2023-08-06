import os , sys
from chatbot.logger import logging
from chatbot.exception import ChatbotException
import json
import pickle


def read_data_from_dir(file_path:str):
    """ this function will return the raw data, 
    and it takes the file path of dataset file."""
    try:
        with open(file_path) as file:
            data = json.load(file)
        logging.info(f"successfully read the dataset from :- {file_path}")
        return data
    except Exception as e:
        raise ChatbotException(e,sys)
    
def save_object(file_path:str,model_obj:object):
    """"this function will save the model object in pickle
    formate.
    """
    try:
        logging.info(f"saving the object path.")
        file_dir = os.path.dirname(file_path)
        os.makedirs(file_dir,exist_ok=True)
        pickle.dump(model_obj,open(file_path,'wb'))
        logging.info(f"successfully saved your object :- {file_path}")
    except Exception as e:
        raise ChatbotException(e,sys)
    

def load_model(file_path:str):

    try:
        if not os.path.exists(file_path):
            raise Exception(f"file not found error {file_path}")
        else:
            model_obj = pickle.load(open(file_path,'rb'))
            logging.info(f"successfully load the model :- {file_path}")
        return model_obj
    except Exception as e:
        raise ChatbotException(e,sys)