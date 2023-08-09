# 3
from chatbot.exception import ChatbotException
from chatbot.entity import cofig_entity,artifact_entity
import tensorflow as tf
from chatbot import util
from chatbot.logger import logging
import os,sys
import tensorflow_hub as hub
import tensorflow_text as text
import numpy as np
# from tensorflow.keras.models import Sequential         ### for sequence model
# from tensorflow.keras.layers import Embedding, Dense , Dropout,LSTM
# from tensorflow.keras.models import load_model


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
    def Get_Bert_model(self):
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
        
    def Get_LSTM_Model(self):
        try:
            vocabulary_size = self.process_artifact.vocabulary_size
            maximum_length_of_sentence = self.process_artifact.maximum_sequence_length
            no_class = self.process_artifact.no_class_or_intent
            logging.info(f"max_seq_len - {maximum_length_of_sentence} and no_class :- {no_class} and vocab_size :- {vocabulary_size}")
            # defining the model
            model  = tf.keras.models.Sequential()
            model.add(tf.keras.layers.Embedding(vocabulary_size,80,input_length=maximum_length_of_sentence))
            model.add(tf.keras.layers.Dropout(0.1))
            model.add(tf.keras.layers.LSTM(200))
            model.add(tf.keras.layers.Dense(no_class,activation='softmax'))
            logging.info(f"model defined successfully !")

            ## defining the metrics 
            # metrics = [
            #     tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            #     tf.keras.metrics.Precision(name="precision"),
            #     tf.keras.metrics.Recall(name="Recall")
            # ]

            # model compilations
            model.compile(loss='sparse_categorical_crossentropy',optimizer = 'adam',metrics=['accuracy'])
            logging.info("model successfully compiled !")

            return model
        except Exception as e:
            raise ChatbotException(e,sys)
        
    def Initiate_Bert_model(self):
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
            callback = tf.keras.callbacks.EarlyStopping(monitor="accuracy",min_delta=0.0001,patience=4)
            model.fit(lst_of_questions,lst_of_labels,epochs=100 ,callbacks=[callback])

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
        

    ##initiating lstm model
    def Initiate_LSTM_model(self):
        try:
            x_train = util.load_model(file_path=self.process_artifact.x_train_processed_file_path)
            y_train = util.load_model(file_path=self.process_artifact.y_train_processed_file_path)
            logging.info(f"successfully loaded the processed data to train the model !")
            logging.info(f"x_train shape :- {x_train.shape} and  y_train shape :- {y_train.shape} and no of class :- {self.process_artifact.no_class_or_intent}")


            model = self.Get_LSTM_Model()

            # callback = tf.keras.callbacks.EarlyStopping(monitor="loss",min_delta=0.000001,patience=4,verbose=1)
            # ,callbacks=[callback]
            history = model.fit(x_train,y_train,epochs=10,verbose=0)
            score = model.evaluate(x_train,y_train) 
            print(f"successfully model trained with {round(score[1]*100)}% accuracy !")
            logging.info(f"successfully model trained with {round(score[1]*100)}% accuracy !")

            ## saving the trained model
            model.save(self.model_trainer_config.lstm_model_file_path)
            util.save_object(file_path=self.model_trainer_config.model_training_logs,model_obj=history)
            logging.info(f"successfully saved the LSTM model at this location :- {self.model_trainer_config.lstm_model_file_path}")

            ## returnig model trainer artifact
            model_trainer_artifact = artifact_entity.ModelTrainerArtifact_for_lstm(lstm_model_file_path=self.model_trainer_config.lstm_model_file_path,
                                                                                   lstm_model_history_log_file_path=self.model_trainer_config.model_training_logs,
                                                                                   max_sequence_length=self.process_artifact.maximum_sequence_length,
                                                                                   dict_with_label_file_path=self.process_artifact.dict_tag_with_label_file_path)
            return model_trainer_artifact
        except Exception as e:
            raise ChatbotException(e,sys)