# 2
# data ingestion will return the list of sentence and lst of labels
# in this file i will process the data and then 
# return the emabadding of the data.
from chatbot.entity import cofig_entity,artifact_entity
from chatbot.exception import ChatbotException
# import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from chatbot import util
from chatbot.logger import logging
import numpy as np
import os,sys

Bert_process_model_name = os.getenv('BERT_PROCESSING_MODEL_NAME')
Bert_model_name = os.getenv('BERT_MODEL_NAME_4_SEQUENCE_CLASSIFACTION')


class PreProcessed:
    def __init__(self,ingestion_artifact:artifact_entity.DataIngestionArtifact,process_config:cofig_entity.Preprocess_config) :
        self.ingestion_artifact = ingestion_artifact
        self.process_config = process_config
    
    def processed_data(self):
        try:
            logging.info(f"loading the data to perform bert pre-processing.")
            transformed_data_dict = util.load_model(self.ingestion_artifact.transformed_file_path)

            bert_process_model = hub.KerasLayer(Bert_process_model_name)
            # bert_model = hub.KerasLayer(Bert_model_name) <<- i will in load in model training

            lst_of_questions = transformed_data_dict['question']
            processed_data = bert_process_model(lst_of_questions)
            logging.info(f"data successfully processed by bert preprocessed model.")

            processed_file_path = self.process_config.processed_data_file_path
            util.save_object(file_path=processed_file_path,model_obj=processed_data)
            logging.info("save preprocessed successfully at this location :- {processed_file_path}")

            ## label encoding
            tag = transformed_data_dict['tag']
            labels = transformed_data_dict['label']
            target = []
            dictionary_tag_with_label = {}
            ## loop for get encoding dict
            for ind , value in enumerate(tag):
                dictionary_tag_with_label[value] = ind
            ## loop for encode the linc
            for label in labels:
                target.append(dictionary_tag_with_label[label])
            target = np.array(target)
            ## saving the target variable
            label_file_path = self.process_config.processed_labels_file_path
            util.save_object(file_path=label_file_path,model_obj=target)
            logging.info(f"succesfully save the target variables at this location :- {label_file_path}")


            ## saving the label with tag dictionary
            dict_tag_with_labl_file_path = self.process_config.dict_with_label_file_path
            util.save_object(file_path=dict_tag_with_labl_file_path,model_obj=dictionary_tag_with_label)
            logging.info(f"successfully save the dictinary, tag with labels at this location :- {dict_tag_with_labl_file_path}")

            process_artifact = artifact_entity.PreprocessedArtifact(processed_data_file_path=processed_file_path,
                                                                    processed_labels_file_path=label_file_path,
                                                                    dict_tag_with_label_file_path=dict_tag_with_labl_file_path
                                                                    no_class_or_intent=len(tag))
            return process_artifact
        except Exception as e:
            raise ChatbotException(e,sys)
        