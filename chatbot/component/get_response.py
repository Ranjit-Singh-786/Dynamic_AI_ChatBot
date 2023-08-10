# 5
from chatbot.exception import ChatbotException
from chatbot.logger import logging
import random
from chatbot import util
import os ,sys


class Get_Response:

    def get_response(self,transformed_dict_path:str,dict_with_label_path:str,intent:str):
        try:
            data_dict = util.load_model(file_path=transformed_dict_path)
            dict_with_label = util.load_model(file_path=dict_with_label_path)

            lst_of_response = data_dict['response']
            index_of_lst_for_response = dict_with_label[intent]
            filter_responses = lst_of_response[index_of_lst_for_response]
            final_response = random.choice(filter_responses)
            logging.info(f"successfully extracted the response from the database !")
            return final_response
        except Exception as e:
            raise ChatbotException(e,sys)