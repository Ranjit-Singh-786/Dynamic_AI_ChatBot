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
@dataclass
class ModelTrainerArtifact:
    model_file_path:str
    lstm_ytrain_data_file_path:str
    lstm_dict_with_label:str