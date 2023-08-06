from  chatbot.exception import ChatbotException
from chatbot.logger import logging
from datetime import datetime
import os, sys









class TrainingPipelineConfig:
    def __init__(self):
        try:
            logging.info('artifact_dir path configuring !')
            self.artifact_dir = os.path.join(os.getcwd(),'artifacts',f"{datetime.now().strftime('%m%d%Y__%H%M%S')}")
        except Exception as e:
            raise ChatbotException(e,sys)
        
class DataIngestionConfig:
    def __init__(self,training_pipeline:TrainingPipelineConfig):
        try:
            self.ingestion_dir_path = os.path.join(training_pipeline.artifact_dir,"Data_ingestion")
            self.transformed_data_dir = os.path.join(self.ingestion_dir_path,"Transformed")
            self.transformed_file_path = os.path.join(self.transformed_data_dir,"transformed_dict.pkl")
            # self.response_dir_path = os.path.join(self.ingestion_dir,"Response")
            # self.intent_dir_path = os.path.join(self.ingestion_dir_path,"intent")
            # self.pattern_dir_path = os.path.join(self.ingestion_dir_path,"Pattern")
            # self.tag_dir_path = os.path.join(self.ingestion_dir_path,"tagg")
        except Exception as e:
            raise ChatbotException(e,sys)