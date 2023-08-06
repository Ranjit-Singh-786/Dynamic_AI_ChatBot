# 1
# responsible for loading and transformation of the data
from chatbot.exception import ChatbotException
from chatbot.logger import logging
from chatbot.entity import  cofig_entity,artifact_entity
from chatbot import util
import os,sys

# data_file_path = "Data/intent.json"
class DataIngestion:
    def __init__(self,ingestion_config:cofig_entity.DataIngestionConfig):
        self.ingestion_config = ingestion_config
        

    ## reading and transforming the data
    def Get_Transformed_Data(self,data_file_path:str):
        try:
            data = util.read_data_from_dir(file_path=data_file_path)
            tag = []
            labels = []
            pattern = []
            response = []
            for intent in data['intents']:
                label = intent['tag']
                response.append(intent['responses'])
                tag.append(label)
                for key , value in intent.items():
                    if key == 'patterns':
                        for patter in intent[key]: 
                            pattern.append(patter)
                            labels.append(label)
            logging.info("Data are extracted in with transformation.")
            data_dictionary = {'no_of_class':len(tag),'tag':tag,'question':pattern,
                               'label':labels,'response':response}
            util.save_object(file_path=self.ingestion_config.transformed_file_path,model_obj=data_dictionary)
            logging.info("tranformed dictionary saved at this location :- {self.ingestion_config.transformed_file_path}")
            Data_ingestion_artifact = artifact_entity.DataIngestionArtifact(transformed_file_path=self.ingestion_config.transformed_file_path)
            return Data_ingestion_artifact
        except Exception as e:
            raise ChatbotException(e,sys)