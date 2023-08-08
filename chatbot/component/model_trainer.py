# 3
# define the function
# write code for training the model
# write the code to save the model and accuracy visualaization chart
# return the model file path
from chatbot.exception import ChatbotException
from chatbot.entity import cofig_entity,artifact_entity
from chatbot import util
from chatbot.logger import logging
import os,sys
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

Bert_process_model_name = os.getenv('BERT_PROCESSING_MODEL_NAME')
Bert_model_name = os.getenv('BERT_MODEL_NAME_4_SEQUENCE_CLASSIFACTION')


class ModelTrainer:
    def __init__(self,process_artifact:artifact_entity.PreprocessedArtifact,
                 ingestion_artifact:artifact_entity.DataIngestionArtifact,
                 model_trainer_config:cofig_entity.ModelTrainer_config):
        self.process_artifact = process_artifact
        self.ingestion_artifact = ingestion_artifact
        self.model_trainer_config = model_trainer_config

    ### functioning to defining the model
    def Get_model(self):
        """ Defining a functional Bert base neural network model"""
        try:
            ## loading the bert model from tensorflow hub
            logging.info("model defining process is starting !")
            bert_process_model = hub.KerasLayer(Bert_process_model_name)
            bert_model = hub.KerasLayer(Bert_model_name)

            # saving the bert models separately for the future need.
            logging.info('saving bert model architecture saperately !')
            util.save_object(file_path=self.model_trainer_config.bert_process_model_file_path,model_obj=bert_process_model)
            util.save_object(file_path=self.model_trainer_config.bert_base_model_file_path,model_obj=bert_model)

            ## defining the model layers
            text_input = tf.keras.layers.Input(shape=(),dtype=tf.string,name='text')
            processed_text = bert_process_model(text_input)
            bert_output = bert_model(processed_text)

            ### adding FFN layers
            drop_out_layer = tf.keras.layers.Dropout(0.3)(bert_output['pooled_output'])
            output_layer = tf.keras.layers.Dense(units=self.process_artifact.no_class_or_intent,activation='softmax')(drop_out_layer)

            ## defining layers in the model
            model = tf.keras.Model(inputs=[text_input], outputs=[output_layer])

            ## defining the metrics 
            metrics = [
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="Recall")
            ]

            ## compiling the model
            model.compile(loss="sparse_categorical_crossentropy",
                          optimizer="adam",
                          metrics=metrics)
            logging.info("model successfully defined and compiled !")

            return model

        except Exception as e:
            raise ChatbotException(e,sys)
        
    def Initiate_model(self):
        try:
            transformed_data_dict_file_path = self.ingestion_artifact.transformed_file_path
            ## dictionary structure for your prefernce
            # data_dictionary = {'no_of_class':len(tag),'tag':tag,'question':pattern,
            #                    'label':labels,'response':response}

            ## getting the input data for the model
            transformed_data_dict = util.load_model(file_path=transformed_data_dict_file_path)
            lst_of_labels = util.load_model(file_path=self.process_artifact.processed_labels_file_path)
            lst_of_questions = transformed_data_dict['question']

            ## getting the model by the function
            model = self.Get_model()
            model.fit(lst_of_questions,lst_of_labels,epochs=5)

            ## saving the bert base complete architecture model
            # >>>>>>>>>>>>>>>> save the model in .h5 format <<<<<<<<<<<<<<<
            model.save(self.model_trainer_config.complete_bert_model_file_path)
            # util.save_object(file_path=self.model_trainer_config.complete_bert_model_file_path,model_obj=model)
            logging.info("success saved the bert base complete architecture model !")


            model_trainer_artifact = artifact_entity.ModelTrainerArtifact(bert_process_model_file_path=self.model_trainer_config.bert_process_model_file_path,                                                                                                                                           
                                                                          bert_base_model_file_path=self.model_trainer_config.bert_base_model_file_path,
                                                                          bert_base_arch_model_file_path=self.model_trainer_config.complete_bert_model_file_path)
            return model_trainer_artifact
            ## saving the complete architecture models

        except Exception as e:
            raise ChatbotException(e,sys)