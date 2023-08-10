# 4 Get the intent prediction
from chatbot.exception import ChatbotException
from chatbot.pipeline.training_pipeline import Training_Pipeline
from chatbot.input_process.process_input_query import Input_Process
from chatbot.logger import logging
import tensorflow as tf
from chatbot import util
import numpy as np
import os,sys
Training_Pipeline_obj = Training_Pipeline()
input_processing_obj = Input_Process()
class Intent_Prediction:
    def __init__(self,change_data_parameter:bool,manually_trained_model:bool):
        self.change_data_parameter = change_data_parameter
        self.manually_trained_model = manually_trained_model
        # self.single_training = single_training
        self.max_sequence_length = 47
        self.current_lstm_model_file_path = ''
        self.dict_with_label_file_path = ''
        self.constant_dict_label = {'greeting': 0, 'goodbye': 1, 'thanks': 2, 'noanswer': 3, 'name': 4, 'options': 5, 'india': 6, 'south_africa_info': 7, 'south_africa_facts': 8}
        self.temp = 0
    
    # Training case condition before getting the prediction
    # param changed , model_training
    #      1                 1    =   no need to train
    #      0                 1    =   no need to train
    #      1                 0    =   need to train model
    #      0                 0    =   no need to train

    def Get_Intent_Preditction(self,query:str):
        """This function is responsible to provide the intent as an prediction !"""
        try:
            change_data_parameter = self.change_data_parameter
            manually_model_tained = self.manually_trained_model

            if (change_data_parameter == False)and(manually_model_tained == False) and (self.temp == 0):
                ## need to train condition is True
                current_trained_lstm_model_file_path , max_sequence_length ,label_tag_dict_file_path = Training_Pipeline_obj.Train_the_model()
                current_lstm_model = tf.keras.models.load_model(current_trained_lstm_model_file_path)

                # get processing pipeline for utilize your time
                processed_xtest_data = input_processing_obj.Process_Input(query=query,max_sequence_length=max_sequence_length)

                # pass the process input to the model to get prediction.
                y_pred = current_lstm_model.predict(processed_xtest_data)
                index_of_prediction = np.argmax(y_pred)

                ## load the dictionary of label with tag
                dict_with_label = util.load_model(file_path=label_tag_dict_file_path)

                intents = ""
                for key_name,value in dict_with_label.items():
                    if value == index_of_prediction:
                        intents = key_name
                # logging.info(f"successfully get the prediction of intent eg. {intent}")
                self.temp = 1   # to ignore the again and again training of the model
                self.max_sequence_length = max_sequence_length
                self.current_lstm_model_file_path = current_trained_lstm_model_file_path
                self.dict_with_label_file_path = label_tag_dict_file_path
                logging.info(f"successfully get the prediction of intent customized model !")
                return intents

            
            elif (change_data_parameter == False) and (manually_model_tained == False) and (self.temp == 1):
                # after single training it will get the predicton for the trained model
                # get processing pipeline for utilize your time
                processed_xtest_data = input_processing_obj.Process_Input(query=query,max_sequence_length=self.max_sequence_length)

                # pass the process input to the model to get prediction.
                current_lstm_model = tf.keras.models.load_model(self.current_lstm_model_file_path)
                y_pred = current_lstm_model.predict(processed_xtest_data)
                index_of_prediction = np.argmax(y_pred)

                ## load the dictionary of label with tag
                dict_with_label = util.load_model(file_path=self.dict_with_label_file_path)
                # print(dict_with_label)
                intents = ""
                for key_name,value in dict_with_label.items():
                    if value == index_of_prediction:
                        intents = key_name
                logging.info(f"successfully get the prediction of intent customized model :- {intent}")
                # to ignore the again and again training of the model
                return intent



            elif (change_data_parameter != False) or (manually_model_tained != False):
                # keep the latest pretrained model
                artifact_path = r"C:\Users\Ranjit Singh\Desktop\working project\Dynamic_AI_ChatBot\artifacts"
                item = os.listdir(artifact_path)
                item.sort(reverse=True)
                latest_model_path = os.path.join(artifact_path,item[0],"model_training","lstm_model.h5")
                # print(latest_model_path)
                ## loading the latest trained model
                
                latest_lstm_model_frm_artifact = tf.keras.models.load_model(filepath=latest_model_path)
                logging.info(f"successfully loaded the pretrained model from :- {latest_model_path}")
                logging.info(f"testing going through with our trained model !")


                # get processing pipeline for utilize your time
                processed_xtest_data = input_processing_obj.Process_Input(query=query,max_sequence_length=self.max_sequence_length)

                # pass the process input to the model to get prediction.
                y_pred = latest_lstm_model_frm_artifact.predict(processed_xtest_data)
                index_of_prediction = np.argmax(y_pred)

                ## load the dictionary of label with tag
                dict_with_label = self.constant_dict_label
                intents = ""
                for key_name,value in dict_with_label.items():
                    if value == index_of_prediction:
                        intents = key_name
                logging.info(f"successfully get the prediction Through pretrained model {intent}")
                # to ignore the again and again training of the model
                
                return intents
        except Exception as e:
            raise ChatbotException(e,sys)
        



obj = Intent_Prediction(change_data_parameter=False,manually_trained_model=False)
obj_without_training = Intent_Prediction(change_data_parameter=True,manually_trained_model=False)
# obj.temp+=1

intent  = obj.Get_Intent_Preditction(query="See you later")     # goodbye

print(f"See you later :- {intent}")

intent = obj.Get_Intent_Preditction(query="hii there")
print(intent)
print()
print(f"prediction without training")
no_training_intent = obj_without_training.Get_Intent_Preditction(query="How can you help")
print(f"How can you help :- {no_training_intent}")
