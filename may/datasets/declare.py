"""
Code to convert BINet synthetic datasets to intermediate format
which can be processed by iBCM and MINERFul
"""

import gzip
import json
import pandas as pd

class DeclareDataset:
    """
    Dataset for iBCM and MINERFul
    This class is used to load the dataset from a json file and convert it to a formagt readable by iBCM.
    """
    def __init__(self, json_data_path):
        
        if json_data_path is None:
            raise ValueError("json_data_path cannot be None")

        self.json_data_path = json_data_path
        self.dataset = None
        self.dataframe = None
        self.encoded_dataset = None
        # self.event_types = None
        self.event_types_encoding = {"ibcm": {}, "minerful": {}}
        self.event_types_decoding = {"ibcm": {}, "minerful": {}}
        # self.event_types_count = 0
        if json_data_path.endswith(".gz"):
            self.read_json_gzip()
        else:
            self.read_json()
        pass

    def read_json(self):
        with open(self.json_data_path, "r") as f:
            self.dataset = json.load(f)
        pass

    def read_json_gzip(self):
        with gzip.open(self.json_data_path, "rt") as f:
            self.dataset = json.load(f)
        pass

    def dataset_info(self):
        print(self.dataset.keys())
    
    def load_dataframe(self):
        if self.dataset is None:
            raise ValueError("dataset is None")
        
        event_data = []
        
        for case in self.dataset['cases']:
            trace_id = case['id']
            anomaly = case['attributes']['label'] if isinstance(case['attributes']['label'], str) else case['attributes']['label']['anomaly']
            
            for event in case['events']:
                event_record = event.copy()
                event_record['trace_id'] = trace_id
                event_record['anomaly'] = anomaly
                event_data.append(event_record)
        
        self.dataframe = pd.DataFrame(event_data)
        
        if 'attributes' in self.dataframe.columns:
            attributes_df = self.dataframe['attributes'].apply(pd.Series)
            self.dataframe = self.dataframe.drop(columns=['attributes']).join(attributes_df)
        
        self.dataframe = self.dataframe[["trace_id", "name", "user", "day", "anomaly" ]]
        self.event_types = sorted(set(self.dataframe["name"].unique()))
        self.event_types_encoding["ibcm"] = {event: i for i, event in enumerate(self.event_types)}
        print(f"Event types encoding ibcm: {self.event_types_encoding['ibcm']}")
        
    def create_numerical_encoding(self):
        df_grouped = self.dataframe.groupby("trace_id")["name"].agg(list).reset_index()
        def apply_ibcm_numbering(name_list, name_dict = self.event_types_encoding["ibcm"]):
            """
            Apply iBCM numbering to a list of names.
            """
            # sorted_name_list = sorted(set(uniques_list))    
            # name_dict = {name: i for i, name in enumerate(sorted_name_list)}
            return [name_dict[name] for name in name_list]
        df_grouped["name"] = df_grouped["name"].apply(
        lambda x: apply_ibcm_numbering(x, self.event_types_encoding["ibcm"])
        )
        self.df_grouped_ibcm = df_grouped
    
    def numerical_encoding_to_string(self):
        def apply_numbers_to_ibcm_string(numbers_list):
            """
            format is: [1, 2, 3] -> "1 -1 2 -1 3 -1 -2"
            """
            op_str = ""
            for i in numbers_list:
                op_str += str(i)
                op_str += " -1 "
            op_str += "-2"
            return op_str
        self.df_grouped_ibcm["name"] = self.df_grouped_ibcm["name"].apply(
            lambda x: apply_numbers_to_ibcm_string(x)
        )
    
    def dataframe_encoded_string_to_file(self, output_path):
        # if self.df_grouped_ibcm is None:
        self.create_numerical_encoding()
        self.numerical_encoding_to_string()      
            
        with open(output_path+".dat", "w") as f:
            for _, row in self.df_grouped_ibcm.iterrows():
                f.write(f"{row['name']}\n")
        with open(output_path+".lab", "w") as f:
            for i in range(self.df_grouped_ibcm.shape[0]):
                f.write(f"1\n")

    def dataframe_to_csv_gzip(self, output_path):
        if self.dataframe is None:
            raise ValueError("dataframe is None")
        self.dataframe.to_csv(output_path, index=False, compression='gzip')        
        pass