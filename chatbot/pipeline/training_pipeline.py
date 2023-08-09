from chatbot.logger import logging
from chatbot.exception import ChatbotException
from chatbot.entity import cofig_entity,artifact_entity
from chatbot.component.data_ingestion import DataIngestion
from chatbot.component.data_processing import PreProcessed
from chatbot.component.model_trainer import ModelTrainer
import os,sys

data_file_path = "Data/intent.json"

class Training_Pipeline:

    def Train_the_model(self):
        try:
            training_pipeline_obj = cofig_entity.TrainingPipelineConfig()
            ingestion_config_obj = cofig_entity.DataIngestionConfig(training_pipeline=training_pipeline_obj)
            data_ingestion_obj = DataIngestion(ingestion_config=ingestion_config_obj)
            ingestion_artifact = data_ingestion_obj.Get_Transformed_Data(data_file_path=data_file_path)

            process_config_obj = cofig_entity.Preprocess_config(training_pipeline=training_pipeline_obj)
            processed_obj = PreProcessed(ingestion_artifact=ingestion_artifact,process_config=process_config_obj)
            process_artifact = processed_obj.process_data_for_LSTM()   # <<< for LSTM model
            # process_artifact = processed_obj.processed_data_for_bert()   #   <<< for bert model

            model_trainer_config_obj = cofig_entity.ModelTrainer_config(training_pipeline=training_pipeline_obj)
            Model_trainer_obj = ModelTrainer(process_artifact=process_artifact,ingestion_artifact=ingestion_artifact,model_trainer_config=model_trainer_config_obj)
            LSTM_Model_artifact = Model_trainer_obj.Initiate_LSTM_model()
            lstm_model_file_path = LSTM_Model_artifact.lstm_model_file_path
            max_sequence_length = LSTM_Model_artifact.max_sequence_length
            label_tag_dict_file_path = LSTM_Model_artifact.dict_with_label_file_path
            # Bert_Model_trainer_artifact = Model_trainer_obj.Initiate_Bert_model()
            return lstm_model_file_path,max_sequence_length,label_tag_dict_file_path
        except Exception as e:
            raise ChatbotException(e,sys)