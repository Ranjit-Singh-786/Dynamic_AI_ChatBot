# 2
from chatbot.entity import cofig_entity,artifact_entity
from chatbot.exception import ChatbotException
# import tensorflow as tf
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences

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
    
    def processed_data_for_bert(self):
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
                                                                    dict_tag_with_label_file_path=dict_tag_with_labl_file_path,
                                                                    no_class_or_intent=len(tag))
            return process_artifact
        except Exception as e:
            raise ChatbotException(e,sys)
        
    def process_data_for_LSTM(self):
        try:
            logging.info(f"loading the data to perform lstm pre-processing.")
            transformed_data_dict = util.load_model(self.ingestion_artifact.transformed_file_path)
            lst_of_questions = transformed_data_dict['question']

            maximum_sequence_length  = max(map(len,lst_of_questions))+8
            lowerize_question = [question.lower() for question in lst_of_questions]

            # one hot encoding x_train_data
            vocabulary_size = 5000
            one_hoted = [one_hot(sentence,vocabulary_size) for sentence in lowerize_question]
            x_train_data  = pad_sequences(one_hoted,padding='post',maxlen=maximum_sequence_length)

            # >>>>>>>>>>>>> to save the data <<<<<<<<<<<<<<<<<
            util.save_object(file_path=self.process_config.x_train_data_file_path,model_obj=x_train_data)
            logging.info(f"x_train processed data successfully saved at this location :- {self.process_config.x_train_data_file_path}")


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

            # <<<<<<<<<<<to save the y_train data>>>>>>>>>>>>>>
            util.save_object(file_path=self.process_config.y_train_data_file_path,model_obj=target)
            logging.info(f"succesfully saved the Y_TRAIN data at this location :- {self.process_config.y_train_data_file_path}")

            util.save_object(file_path=self.process_config.dict_with_label_file_path_lstm,model_obj=dictionary_tag_with_label)
            logging.info(f"dictinary tag with label saved at this location :- {self.process_config.dict_with_label_file_path_lstm}")

            process_artifact = artifact_entity.PreprocessedArtifact(no_class_or_intent=len(tag),
                                                                    x_train_processed_file_path=self.process_config.x_train_data_file_path,
                                                                    y_train_processed_file_path=self.process_config.y_train_data_file_path,
                                                                    dict_tag_with_label_file_path=self.process_config.dict_with_label_file_path_lstm,
                                                                    vocabulary_size=vocabulary_size,
                                                                    maximum_sequence_length=maximum_sequence_length,
                                                                    transformed_data_dict_file_path = self.ingestion_artifact.transformed_file_path)
            return process_artifact
        except Exception as e:
            raise ChatbotException(e,sys)