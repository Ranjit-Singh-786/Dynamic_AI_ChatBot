from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    transformed_file_path:str

@dataclass
class PreprocessedArtifact:
    no_class_or_intent:int
    processed_data_file_path:str
    processed_labels_file_path:str
    dict_tag_with_label_file_path:str