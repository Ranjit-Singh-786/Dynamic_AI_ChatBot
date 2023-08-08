from chatbot.logger import logging
from chatbot.exception import ChatbotException
from chatbot.entity import cofig_entity,artifact_entity
from chatbot.component.data_ingestion import DataIngestion
from chatbot.component.data_processing import PreProcessed
from chatbot.component.model_trainer import ModelTrainer
import os,sys

data_file_path = "Data/intent.json"

if __name__=="__main__":
    # try:
    #     a = 5/0
    #     logging.info('logging and exception handling successfully tested')
    #     print('successfully tested')
    # except Exception as e:
    #     logging.info('logging and exception handling successfully tested')
        
    #     raise ChatbotException(error_message=e,error_detail=sys)

    try:
        training_pipeline_obj = cofig_entity.TrainingPipelineConfig()
        ingestion_config_obj = cofig_entity.DataIngestionConfig(training_pipeline=training_pipeline_obj)
        data_ingestion_obj = DataIngestion(ingestion_config=ingestion_config_obj)
        ingestion_artifact = data_ingestion_obj.Get_Transformed_Data(data_file_path=data_file_path)

        process_config_obj = cofig_entity.Preprocess_config(training_pipeline=training_pipeline_obj)
        processed_obj = PreProcessed(ingestion_artifact=ingestion_artifact,process_config=process_config_obj)
        process_artifact = processed_obj.processed_data()

        model_trainer_config_obj = cofig_entity.ModelTrainer_config(training_pipeline=training_pipeline_obj)
        Model_trainer_obj = ModelTrainer(process_artifact=process_artifact,ingestion_artifact=ingestion_artifact,model_trainer_config=model_trainer_config_obj)
        Model_trainer_artifact = Model_trainer_obj.Initiate_model()

    except Exception as e:
        raise ChatbotException(e,sys)