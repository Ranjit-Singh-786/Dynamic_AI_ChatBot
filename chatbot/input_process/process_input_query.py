from chatbot.logger import logging
from chatbot.exception import ChatbotException
from chatbot.entity import cofig_entity,artifact_entity
from chatbot.component.data_ingestion import DataIngestion
from chatbot.component.data_processing import PreProcessed
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from chatbot.logger import logging
import numpy as np
import os,sys


class Input_Process:

    def Process_Input(self,query:str,max_sequence_length:int):
        try:
            logging.info(f"user query processing !")
            query_in_lst = [query]
            lowerize_question = [question.lower() for question in query_in_lst]

            vocabulary_size = 5000
            one_hoted = [one_hot(sentence,vocabulary_size) for sentence in lowerize_question]
            x_test_data  = pad_sequences(one_hoted,padding='post',maxlen=max_sequence_length)
            x_test_data = np.array(x_test_data)
            logging.info(f"user query successfully processed, to get intent prediction !")
            return x_test_data
        except Exception as e:
            raise ChatbotException(e,sys)