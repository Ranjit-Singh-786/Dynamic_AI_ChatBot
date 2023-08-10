from dataclasses import dataclass
from typing import Optional

@dataclass
class DataIngestionArtifact:
    transformed_file_path:str

@dataclass
class PreprocessedArtifact:
    no_class_or_intent:int
    x_train_processed_file_path:str
    y_train_processed_file_path:str
    dict_tag_with_label_file_path:str
    vocabulary_size:int
    maximum_sequence_length:int
    transformed_data_dict_file_path:str


@dataclass
class ModelTrainerArtifact_for_lstm:
    lstm_model_file_path:str
    lstm_model_history_log_file_path:str
    max_sequence_length:int
    dict_with_label_file_path:str
    transformed_dict_file_path:str


@dataclass
class ModelTrainerArtifact_for_Bert:
    bert_process_model_file_path:str
    bert_base_model_file_path:str
    bert_base_arch_model_file_path:str