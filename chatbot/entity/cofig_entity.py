from typing import Any
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
        
class Preprocess_config:
    def __init__(self,training_pipeline:TrainingPipelineConfig):
        self.preprocessed_dir = os.path.join(training_pipeline.artifact_dir,"Preprocessed_data")
        self.processed_data_file_path = os.path.join(self.preprocessed_dir,"bert_processed_data.pkl")
        self.processed_labels_file_path = os.path.join(self.preprocessed_dir,"processed_labels.pkl")
        self.dict_with_label_file_path = os.path.join(self.preprocessed_dir,"label_with_tag.pkl")
        self.lstm_processed_dir = os.path.join(self.preprocessed_dir,"lstm_processed")
        self.x_train_data_file_path = os.path.join(self.lstm_processed_dir,"x_train_data.pkl")
        self.y_train_data_file_path = os.path.join(self.lstm_processed_dir,"y_train_data.pkl")
        self.dict_with_label_file_path_lstm = os.path.join(self.lstm_processed_dir,"label_with_tag.pkl")
        

class ModelTrainer_config:
    def __init__(self,training_pipeline:TrainingPipelineConfig):
        self.model_trainer_dir = os.path.join(training_pipeline.artifact_dir,"model_training")
        self.bert_process_model_file_path = os.path.join(self.model_trainer_dir,"bert_process_model.pkl")
        self.bert_base_model_file_path = os.path.join(self.model_trainer_dir,"bert_base_model.pkl")
        self.complete_bert_model_file_path = os.path.join(self.model_trainer_dir,"bert_base_cmplt_arch_model.h5")

        self.lstm_model_file_path = os.path.join(self.model_trainer_dir,"lstm_model.h5")
        self.model_training_logs = os.path.join(self.model_trainer_dir,"lstm_model_training_history.pkl")


        