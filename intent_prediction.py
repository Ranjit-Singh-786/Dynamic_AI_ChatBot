# 4 Get the intent prediction
from chatbot.exception import ChatbotException
from chatbot.pipeline.training_pipeline import Training_Pipeline
import tensorflow as tf
from chatbot import util
import os,sys
Training_Pipeline_obj = Training_Pipeline()

class Intent_Prediction:
    def __init__(self,change_data_parameter:bool,manually_trained_model:bool,single_training:bool,user_query=None):
        self.change_data_parameter = change_data_parameter
        self.manually_trained_model = manually_trained_model
        self.user_query = user_query
        self.single_training = single_training
        self.temp = 0
    
    # Training case condition before getting the prediction
    # param changed , model_training
    #      1                 1    =   no need to train
    #      0                 1    =   no need to train
    #      1                 0    =   need to train model
    #      0                 0    =   no need to train

    def Get_Intent_Preditction(self):
        try:
            change_data_parameter = self.change_data_parameter
            manually_model_tained = self.manually_trained_model

            if (change_data_parameter == False)and(manually_model_tained==False):
                ## need to train condition is True
                current_trained_lstm_model_file_path = Training_Pipeline_obj.Train_the_model()
                current_lstm_model = tf.keras.models.load_model(current_trained_lstm_model_file_path)
                self.temp+=1
                if self.temp == 1:
                    # >>>>>>>>>>>>  continue from here  <<<<<<<<<<<<<<<<<<<<
                    # it will execute everytime with this condition think the solution

                    current_lstm_model.predict()

            else:
                # keep the latest pretrained model
                artifact_path = r"C:\Users\Ranjit Singh\Desktop\working project\Dynamic_AI_ChatBot\artifacts"
                item = os.listdir(artifact_path)
                item.sort(reverse=True)
                latest_model_path = os.path.join(artifact_path,item[0],"model_training","lstm_model.h5")

                ## loading the latest trained model
                
                latest_lstm_model_frm_artifact = tf.keras.models.load_model(latest_model_path)
                print("successfully loaded the latest model from artifact")
        except Exception as e:
            raise ChatbotException(e,sys)
    def a(self):
        print(self.user_query)
obj = Intent_Prediction(change_data_parameter=True,manually_trained_model=False)
# obj.Get_Intent_Preditction()
obj.a()


# intent_predictin.py