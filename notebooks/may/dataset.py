import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from pprint import pprint

class Event(dict):
    """
    Event class with following keys:
    :name: name of the event
    :user: user who performed the event
    :case_id: case id of the event
    """
    pass

class Dataset():
    def __init__(self, max_cases=None, anomaly_probabilty=None):
        self.event_names = set(["Create SC", "Approve SC", "Create PO", "Approve PO", "Pay"])
        self.user_names = set(["Dev", "Chantal", "Seokju", "Jonas", "Kaly"])
        self.valid_traces = [
            ["Create SC", "Approve SC", "Create PO", "Approve PO", "Pay"],
        ]
        self.invalid_traces_control_flow = [
            ["Approve SC", "Create SC", "Create PO", "Approve PO", "Pay"],
            ["Create SC", "Approve SC", "Approve PO", "Create PO", "Pay"],
            ["Create SC", "Approve SC", "Create PO", "Pay", "Approve PO"],            
            ["Create PO", "Approve PO", "Create SC", "Approve SC", "Pay"]
        ]
        self.event_user_mapping = {
            "Create SC": ["Dev", "Chantal"],
            "Approve SC": ["Kaly"],
            "Create PO": ["Dev", "Jonas"],
            "Approve PO": ["Kaly"],
            "Pay": ["Seokju"]
        }
        self.event_user_mapping_inv = {
            "Dev": ["Create SC", "Create PO"],
            "Chantal": ["Create SC"],
            "Kaly": ["Approve SC", "Approve PO"],
            "Jonas": ["Create PO"],
            "Seokju": ["Pay"]
        }
        self.anomaly_probability = 0.3
        self.max_cases = 1000
        self.actual_cases = 0
        if max_cases is not None:
            self.max_cases = max_cases
        if anomaly_probabilty is not None:
            self.anomaly_probability = anomaly_probabilty
        self.raw_dataset = []
        self.case_id_counter = 0
        self.encoders = {}
        self.encoders["name"] = LabelEncoder()
        self.encoders["name"].fit(list(self.event_names))
        print(f"encoders[name]:\n{[(str(c), str(t)) for c, t in list(zip(self.encoders['name'].classes_, self.encoders['name'].transform(self.encoders['name'].classes_)))]}")
        self.encoders["user"] = LabelEncoder()
        self.encoders["user"].fit(list(self.user_names))
        print(f"encoders[user]:\n{[(str(c), str(t)) for c, t in list(zip(self.encoders['user'].classes_, self.encoders['user'].transform(self.encoders['user'].classes_)))]}")
        
        
    def create_valid_traces(self, num_traces):
        partial_dataset = []
        cases_per_trace = num_traces // len(self.valid_traces)
        for valid_trace in self.valid_traces:
            for _ in range(cases_per_trace):
                event = Event()
                for event_name in valid_trace:
                    event["name"] = event_name
                    event["user"] = np.random.choice(self.event_user_mapping[event_name])
                    event["case_id"] = self.case_id_counter
                    # case.append(event.copy())
                    partial_dataset.append(event.copy())
                self.case_id_counter += 1
                # partial_dataset.append(case)
        return partial_dataset, len(partial_dataset)//5
                
    def create_invalid_traces_control_flow(self, num_traces):
        partial_dataset = []
        cases_per_trace = num_traces // len(self.invalid_traces_control_flow)
        for invalid_trace in self.invalid_traces_control_flow:
            for _ in range(cases_per_trace):
                event = Event()
                for event_name in invalid_trace:
                    event["name"] = event_name
                    event["user"] = np.random.choice(self.event_user_mapping[event_name])
                    event["case_id"] = self.case_id_counter
                    # case.append(event.copy())
                    partial_dataset.append(event.copy())
                self.case_id_counter += 1
                # partial_dataset.append(case)
        return partial_dataset, len(partial_dataset)//5
    
    def create_invalid_traces_attribute(self, num_traces):
        partial_dataset = []
        cases_per_trace = num_traces // len(self.valid_traces)
        for valid_trace in self.valid_traces:
            for _ in range(cases_per_trace):
                event = Event()
                for event_name in valid_trace:
                    event["name"] = event_name
                    wrong_users = set(self.user_names) - set(self.event_user_mapping[event_name])
                    event["user"] = np.random.choice(list(wrong_users))
                    event["case_id"] = self.case_id_counter
                    # case.append(event.copy())
                    partial_dataset.append(event.copy())
                self.case_id_counter += 1
                # partial_dataset.append(case)
        return partial_dataset, len(partial_dataset)//5
    
    def create_dataset(self, max_cases=None, anomaly_probabilty=None):
        if max_cases is None:
            max_cases = self.max_cases
        else:
            self.max_cases = max_cases
        if anomaly_probabilty is None:
            anomaly_probabilty = self.anomaly_probability
        else:
            self.anomaly_probability = anomaly_probabilty
        num_anomalous_cases = int(max_cases * anomaly_probabilty)
        num_normal_cases = max_cases - num_anomalous_cases
        self.actual_cases = []
        raw_dataset, actual_count = self.create_valid_traces(num_normal_cases)
        self.raw_dataset += raw_dataset
        self.actual_cases += [actual_count]
        raw_dataset, actual_count = self.create_invalid_traces_control_flow(num_anomalous_cases // 2)
        self.raw_dataset += raw_dataset
        self.actual_cases += [actual_count]
        raw_dataset, actual_count = self.create_invalid_traces_attribute(num_anomalous_cases // 2)
        self.raw_dataset += raw_dataset
        self.actual_cases += [actual_count]
    
    @property
    def raw_dataset_as_df(self):
        # name, user, case_id"
        return pd.DataFrame(self.raw_dataset)
    
    @property
    def raw_dataset_as_np_array(self):
        np_df = self.raw_dataset_as_df.to_numpy().reshape(-1, 5, 3)
        return np_df
    
    @property
    def encoded_features(self):
        without_case_id = self.raw_dataset_as_np_array[:, :, 0:2] # name, user
        # without_case_id_one_row_one_case = without_case_id.reshape(-1, 10)
        without_case_id_one_row_one_event = without_case_id.reshape(-1, 2)
        just_names = without_case_id_one_row_one_event[:, 0]
        just_users = without_case_id_one_row_one_event[:, 1]
        encoded_names = self.encoders["name"].transform(just_names).reshape(-1, 1)
        encoded_users = self.encoders["user"].transform(just_users).reshape(-1, 1)
        encoded_data = np.hstack((encoded_names, encoded_users)).reshape(-1, 10)
        return encoded_data
    
    @property
    def one_hot_encoded_features(self):
        one_row_one_event = self.encoded_features.reshape(-1, 2)

        name_column = one_row_one_event[:, 0].reshape(-1, 1)  # Get name column
        user_column = one_row_one_event[:, 1].reshape(-1, 1)  # Get user column

        self.one_hot_encoder = {}
        self.one_hot_encoder["name"] = OneHotEncoder(sparse_output=False)
        self.one_hot_encoder["user"]= OneHotEncoder(sparse_output=False)

        one_hot_encoded_name = self.one_hot_encoder["name"].fit_transform(name_column).reshape(-1, 5)
        one_hot_encoded_user = self.one_hot_encoder["user"].fit_transform(user_column).reshape(-1, 5)
        
        one_hot_encoded_data = np.hstack((one_hot_encoded_name, one_hot_encoded_user)).reshape(-1, 50)
        return one_hot_encoded_data
    
    @property
    def one_hot_encoded_features_2d(self):
        return self.one_hot_encoded_features.reshape(-1, 10, 5)
    
    @property
    def x_one_hot_y_int(self):
        return self.one_hot_encoded_features, self.encoded_features.reshape
    
    @property
    def x_one_hot_2d_y_int_2d(self):
        return self.one_hot_encoded_features_2d, self.encoded_features.reshape(-1, 10, 1)
    
    @property
    def x_one_hot_2d_y_int(self):
        #reshape is not needed here because it's already (-1, 10)
        return self.one_hot_encoded_features_2d, self.encoded_features.reshape(-1, 10)
    
    def inverse_one_hot_encoded_features_to_int(self, one_hot_encoded_data):
        """
        There are 50 columns in a completely one hot encoded data
        thats 5 events and 5 users altenately
        first we deconde to integer level
        this function is just for one vector but maybe can be broadcasted over the whole dataset
        """
        one_row_one_event = one_hot_encoded_data.reshape(-1, 10)
        name_column = one_row_one_event[:, 0:5].reshape(-1, 5)
        user_column = one_row_one_event[:, 5:10].reshape(-1, 5)
        de_encoded_name = self.one_hot_encoder["name"].inverse_transform(name_column).reshape(-1, 1)
        de_encoded_user = self.one_hot_encoder["user"].inverse_transform(user_column).reshape(-1, 1)
        de_encoded_data = np.hstack((de_encoded_name, de_encoded_user)).reshape(-1, 2)
        return de_encoded_data.reshape(-1, 10)

    def inverse_one_hot_encoded_features_to_string(self, one_hot_encoded_data):
        """
        There are 50 columns in a completely one hot encoded data
        thats 5 events and 5 users altenately
        first we deconde to integer level
        then to string level
        this function is just for one vector but maybe can be broadcasted over the whole dataset
        """
        one_hot_encoded_data_int = self.inverse_one_hot_encoded_features_to_int(one_hot_encoded_data)
        one_row_one_event = one_hot_encoded_data_int.reshape(-1, 2)
        name_column = one_row_one_event[:, 0]
        user_column = one_row_one_event[:, 1]
        de_encoded_name = self.encoders["name"].inverse_transform(name_column).reshape(-1, 1)
        de_encoded_user = self.encoders["user"].inverse_transform(user_column).reshape(-1, 1)
        de_encoded_data = np.hstack((de_encoded_name, de_encoded_user)).reshape(-1, 2)
        return de_encoded_data.reshape(-1, 10)
        
        
        
    pass
    
# synth_dataset = Dataset(max_cases=1000, anomaly_probabilty=0.3)
# synth_dataset.create_dataset()
# print(f"Cases Distribution: {synth_dataset.actual_cases}")
# one_hot_flat = synth_dataset.one_hot_encoded_features
# print(f"one_hot_flat shape: {one_hot_flat.shape}")
# print(f"Raw features test:\n{synth_dataset.raw_dataset_as_df[:5]}")
# print(f"Encoded Features test:\n{synth_dataset.encoded_features[0]}")
# # pprint(f"One hot Encoded feature test: {one_hot_flat[0]}")
# print(f"De encoding test:\n{synth_dataset.inverse_one_hot_encoded_features_to_int(one_hot_flat[0])}")
# from pprint import pprint
# pprint(f"Encoded Features: {temp.shape}")
# temp = synth_dataset.one_hot_encoded_features
# pprint(f"One hot Encoded Features: {temp.shape}")
# int_de_encode = synth_dataset.inverse_one_hot_encoded_features_to_int(temp[0])
# print(f"De encoding test: {int_de_encode}")
# str_de_encode = synth_dataset.inverse_one_hot_encoded_features_to_string(temp[0])

# synth_dataset = dataset(max_cases=1000, anomaly_probabilty=0.3)
# synth_dataset.create_dataset()
# one_hot_flat = synth_dataset.one_hot_encoded_features
# print(f"one_hot_flat shape: {one_hot_flat.shape}")
# print(f"Raw features test:\n{synth_dataset.raw_dataset_as_df[:5]}")
# print(f"Encoded Features test:\n{synth_dataset.encoded_features[0]}")
# # pprint(f"One hot Encoded feature test: {one_hot_flat[0]}")
# print(f"De encoding test:\n{synth_dataset.inverse_one_hot_encoded_features_to_int(one_hot_flat[0])}")
# from pprint import pprint
# pprint(f"Encoded Features: {temp.shape}")
# temp = synth_dataset.one_hot_encoded_features
# pprint(f"One hot Encoded Features: {temp.shape}")
# int_de_encode = synth_dataset.inverse_one_hot_encoded_features_to_int(temp[0])
# print(f"De encoding test: {int_de_encode}")
# str_de_encode = synth_dataset.inverse_one_hot_encoded_features_to_string(temp[0])
