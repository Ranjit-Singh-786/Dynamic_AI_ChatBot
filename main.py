from chatbot.logger import logging
from chatbot.exception import ChatbotException
import os,sys


if __name__=="__main__":
    try:
        a = 5/0
        logging.info('logging and exception handling successfully tested')
        print('successfully tested')
    except Exception as e:
        logging.info('logging and exception handling successfully tested')
        
        raise ChatbotException(error_message=e,error_detail=sys)