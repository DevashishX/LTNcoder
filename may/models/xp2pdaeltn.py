# Copyright 2018 Timo Nolle
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# ==============================================================================

import numpy as np

import april
from april.anomalydetection.binet.core import NNAnomalyDetector
from april.anomalydetection.utils.result import AnomalyDetectionResult
from april.enums import Base
from april.enums import Heuristic
from april.enums import Mode
from april.enums import Strategy
from april.fs import ModelFile        

from typing import Union


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import itertools

np.random.seed(0)

import tensorflow as tf
from tensorflow import keras
physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("GPU found")
    print("Memory growth set")
else:
    print("No GPU found")
from tensorflow.keras.layers import Input, Dense, Dropout, GaussianNoise, Reshape, Flatten, Concatenate, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from typing import Union
import ltn
from collections import defaultdict

class P2PDAE(NNAnomalyDetector):
    """
    Implements a denoising autoencoder based anomaly detection algorithm.
    For P2P dataset using LTN
    """
    
    """ 
    inherited from NNAnomalyDetector class:
    load()
    _save()
    model_fn() not implemented
    fit()
    detect()
    
    inherited from AnomalyDetector class:
    model() return self._model
    load()
    save()
    _save()
    fit() not implemented
    detect() not implemented
    
    inherited from DAE()
    model_fn() not implemented
    detect() not implemented
    
    """   
    

    abbreviation = 'p2pdae'
    name = 'P2PDAE'

    supported_heuristics = [Heuristic.BEST, Heuristic.ELBOW_DOWN, Heuristic.ELBOW_UP,
                            Heuristic.LP_LEFT, Heuristic.LP_MEAN, Heuristic.LP_RIGHT,
                            Heuristic.MEAN, Heuristic.MEDIAN, Heuristic.RATIO, Heuristic.MANUAL]
    supported_strategies = [Strategy.SINGLE, Strategy.ATTRIBUTE, Strategy.POSITION, Strategy.POSITION_ATTRIBUTE]
    supported_modes = [Mode.BINARIZE]
    supported_bases = [Base.LEGACY, Base.SCORES]
    supports_attributes = True

    config = dict(hidden_layers=2,
                  hidden_size_factor=.2,
                  noise=None)

    def __init__(self, model=None):
        """Initialize DAE model.

        Size of hidden layers is based on input size. The size can be controlled via the hidden_size_factor parameter.
        This can be float or a list of floats (where len(hidden_size_factor) == hidden_layers). The input layer size is
        multiplied by the respective factor to get the hidden layer size.

        :param model: Path to saved model file. Defaults to None.
        :param hidden_layers: Number of hidden layers. Defaults to 2.
        :param hidden_size_factor: Size factors for hidden layer base don input layer size.
        :param epochs: Number of epochs to train.
        :param batch_size: Mini batch size.
        """
        # super(P2PDAELTN, self).__init__(model=model)
        self.model:Model = None # Main model which will be trained on one hot encoded features of dataset
        self._model_path = None
        self.input_layer = None # keras input layer of the model
        self.latent_space_last_layer = None # shared latent space/decoder layer of the models 
        self.individual_outputs = [] # individual outputs of the model
        self.output_layer = [] # Single output of the model 
        self.individual_models = [] # models for each output, which have shared input and latent spcae layers
        self.ltn_predicates = [] # LTN predicates for each output
        self.history = None # training history of the model
        
        
        # print(f"self.input_shape: {self.input_shape}")     

    def _build_main_model(self):
        
        # Layers
        self.input_layer = Input(shape=(self.input_shape, ), name='input')
        self.flat_input = Flatten(name="flatten")(self.input_layer)
        
        # Noise layer
        if self.noise is not None:
            x = GaussianNoise(self.noise, name="noise")(self.flat_input)

        # Hidden layers
        for i in range(self.hidden_layers):
            if isinstance(self.hidden_size_factor, list):
                factor = self.hidden_size_factor[i]
            else:
                factor = self.hidden_size_factor
            x = Dense(int(self.input_shape * factor), activation='elu', name=f'hidden-{i}')(x)
            x = Dropout(0.5, name=f'dropout-{i}')(x)

        # latent space
        self.latent_space_last_layer = x

        # Output layer(s)
        self.output_layer = Dense(self.input_shape, activation='sigmoid', name='output')(self.latent_space_last_layer)
        
        # Build model
        self.model = Model(inputs=self.input_layer, outputs=self.output_layer, name="main-DAE")
        self.model.compile(
            optimizer=Adam(learning_rate=0.0001, beta_2=0.99),
            loss='mean_squared_error'
        )
        # self.model._name = "Main DAE"
        pass

    def model_fn(self, dataset:april.Dataset, **kwargs):
        
        hidden_layers = kwargs.pop('hidden_layers')
        hidden_size_factor = kwargs.pop('hidden_size_factor')
        noise = kwargs.pop('noise')

        # Parameters
        self.dataset = dataset
        self.hidden_layers = hidden_layers if hidden_layers is not None else 2 # 2
        self.hidden_size_factor = hidden_size_factor if hidden_size_factor is not None else 0.2 # 0.2
        self.noise = noise if noise is not None else True # for denoising autoencoder
        self.input_shape = self.dataset.flat_onehot_features_2d.shape[1]
        self.dataset_attribute_dims = self.dataset.attribute_dims
        # features = dataset.flat_onehot_features_2d
        
        self._build_main_model() # build main model
        
        return self.model, dataset.flat_onehot_features_2d, dataset.flat_onehot_features_2d  # Features are also targets

    def save(self, file_name: str):
        """Save model to path.

        :param path: Path to save model to.
        """
        if self.model is not None:
            model_file = ModelFile(file_name)
            self._model_path = model_file.str_path
            # self._save(model_file.str_path)
            self.model.save(model_file.str_path)
            return model_file
        else:
            raise RuntimeError(
                'Saving not possible. No model has been trained yet.')
        pass
    
    def load(self, file_name: str):
        """Load model from path.

        :param path: Path to load model from.
        """
        file_name = ModelFile(file_name)
        
        self._model_path = file_name.str_path
        self.model = keras.models.load_model(file_name.str_path)
        pass
    
    def fit(self, dataset, epochs=30, batch_size=500, validation_split=0.1, **kwargs):
        # Build model
        self._model, features, targets = self.model_fn(dataset, **self.config)

        # Train model
        self.history = self._model.fit(
            features,
            targets,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            **kwargs
        )

        return self.history
    
    def detect(self, dataset):
        """
        Calculate the anomaly score for each event attribute in each trace.
        Anomaly score here is the mean squared error.

        :param traces: traces to predict
        :return:
            scores: anomaly scores for each attribute;
                            shape is (#traces, max_trace_length - 1, #attributes)

        """
        # Get features
        _, features, _ = self.model_fn(dataset, **self.config)

        # Parameters
        input_size = int(self.model.input.shape[1])
        features_size = int(features.shape[1])
        if input_size > features_size:
            features = np.pad(features, [(0, 0), (0, input_size - features_size), (0, 0)], mode='constant')
        elif input_size < features_size:
            features = features[:, :input_size]

        # Init anomaly scores array
        scores = np.zeros(dataset.binary_targets.shape)

        # Get predictions
        predictions = self.model.predict(features)

        # Calculate error
        errors = np.power(features - predictions, 2)

        # Split the errors according to the attribute dims
        split = np.cumsum(np.tile(dataset.attribute_dims, [dataset.max_len]), dtype=int)[:-1]
        errors = np.split(errors, split, axis=1)
        errors = np.array([np.mean(a, axis=1) if len(a) > 0 else 0.0 for a in errors])

        for i in range(len(dataset.attribute_dims)):
            error = errors[i::len(dataset.attribute_dims)]
            scores[:, :, i] = error.T

        return AnomalyDetectionResult(scores=scores)

class P2PDAELTN(NNAnomalyDetector):
    """
    Implements a denoising autoencoder based anomaly detection algorithm.
    For P2P dataset using LTN
    """
    
    """ 
    inherited from NNAnomalyDetector class:
    load()
    _save()
    model_fn() not implemented
    fit()
    detect()
    
    inherited from AnomalyDetector class:
    model() return self._model
    load()
    save()
    _save()
    fit() not implemented
    detect() not implemented
    
    inherited from DAE()
    model_fn() not implemented
    detect() not implemented
    
    """   
    

    abbreviation = 'p2pdaeltn'
    name = 'P2PDAELTN'

    supported_heuristics = [Heuristic.BEST, Heuristic.ELBOW_DOWN, Heuristic.ELBOW_UP,
                            Heuristic.LP_LEFT, Heuristic.LP_MEAN, Heuristic.LP_RIGHT,
                            Heuristic.MEAN, Heuristic.MEDIAN, Heuristic.RATIO, Heuristic.MANUAL]
    supported_strategies = [Strategy.SINGLE, Strategy.ATTRIBUTE, Strategy.POSITION, Strategy.POSITION_ATTRIBUTE]
    supported_modes = [Mode.BINARIZE]
    supported_bases = [Base.LEGACY, Base.SCORES]
    supports_attributes = True

    config = dict(hidden_layers=2,
                  hidden_size_factor=.2,
                  noise=None)

    def __init__(self, dataset:april.Dataset, model=None, hidden_layers=None, hidden_size_factor=None, noise=None):
        """Initialize DAE model.

        Size of hidden layers is based on input size. The size can be controlled via the hidden_size_factor parameter.
        This can be float or a list of floats (where len(hidden_size_factor) == hidden_layers). The input layer size is
        multiplied by the respective factor to get the hidden layer size.

        :param model: Path to saved model file. Defaults to None.
        :param hidden_layers: Number of hidden layers. Defaults to 2.
        :param hidden_size_factor: Size factors for hidden layer base don input layer size.
        :param epochs: Number of epochs to train.
        :param batch_size: Mini batch size.
        """
        # super(P2PDAELTN, self).__init__(model=model)
        self.model:Model = None # Main model which will be trained on one hot encoded features of dataset
        self._model_path = None
        self.input_layer = None # keras input layer of the model
        self.latent_space_last_layer = None # shared latent space/decoder layer of the models 
        self.individual_outputs = [] # individual outputs of the model
        self.output_layer = None # Single output of the model 
        self.individual_models = [] # models for each output, which have shared input and latent spcae layers
        self.ltn_predicates = [] # LTN predicates for each output
        self.history = None # training history of the model

        # Parameters
        self.dataset = dataset
        self.hidden_layers = hidden_layers if hidden_layers is not None else 2 # 2
        self.hidden_size_factor = hidden_size_factor if hidden_size_factor is not None else 0.2 # 0.2
        self.noise = noise if noise is not None else True # for denoising autoencoder
        self.input_shape = self.dataset.flat_onehot_features_2d.shape[1]
        self.dataset_attribute_dims = self.dataset.attribute_dims
        
        # print(f"self.input_shape: {self.input_shape}")     

    def _build_main_model(self):
        
        # Layers
        self.input_layer = Input(shape=(self.input_shape, ), name='input')
        self.flat_input = Flatten(name="flatten")(self.input_layer)
        
        # Noise layer
        if self.noise is not None:
            x = GaussianNoise(self.noise, name="noise")(self.flat_input)

        # Hidden layers
        for i in range(self.hidden_layers):
            if isinstance(self.hidden_size_factor, list):
                factor = self.hidden_size_factor[i]
            else:
                factor = self.hidden_size_factor
            x = Dense(int(self.input_shape * factor), activation='elu', name=f'hidden-{i}')(x)
            x = Dropout(0.5, name=f'dropout-{i}')(x)

        # latent space
        self.latent_space_last_layer = x

        # Output layer(s)
        self.output_layer = Dense(self.input_shape, activation='sigmoid', name='output')(self.latent_space_last_layer)
        
        # Build model
        self.model = Model(inputs=self.input_layer, outputs=self.output_layer, name="main-DAE")
        self.model.compile(
            optimizer=Adam(learning_rate=0.0001, beta_2=0.99),
            loss='mean_squared_error'
        )
        # self.model._name = "Main DAE"
        return self.model
    
    def _fit_main_model(self, epochs=30, batch_size=100, validation_split=0.1, **kwargs):
        """
        Fit the main model on the dataset.
        Args:
            dataset: Dataset object containing the training data.
            epochs: Number of epochs to train.
            batch_size: Size of the mini-batch.
            validation_split: Fraction of the training data to use for validation.
        """
        # Train the main model
        self._history_main_model = self.model.fit(self.dataset.flat_onehot_features_2d, self.dataset.flat_onehot_features_2d, 
                       epochs=epochs, batch_size=batch_size, validation_split=validation_split, **kwargs)
    
    def _build_individual_models(self):
        """
        Build sub-models from a main model, each returning a specific attribute slice.
        
        Parameters:
            self.model: The original Keras model.
            self.dataset.attribute_dims: List of ints — the size of each one-hot encoded attribute.
            self.dataset.attribute_keys: List of strings — names of each attribute.

        Returns:
            A list of Keras models with proper .name set (e.g., 'name_0', 'user_0', ...).
        """
        assert len(self.dataset.attribute_dims) == len(self.dataset.attribute_keys), "Dims and names must match."

        total_output_dim = self.model.output.shape[-1]
        offset = 0
        block_count = 0
        self.individual_models = []
        # self.individual_models_trainable_parameters = []
        self.individual_predicates = []
        self.individual_predicates_trainable_parameters = []
        self.activity_predicates = []
        self.user_predicates = []

        while offset < total_output_dim:
            for dim, name in zip(self.dataset.attribute_dims, self.dataset.attribute_keys):
                if offset + dim > total_output_dim:
                    break  # Avoid slicing beyond output size

                # Slice the output
                sliced_output = Lambda(lambda x, s=offset, e=offset+dim: x[:, s:e])(self.model.output)

                # Create a model with a custom name
                model_name = f"{name}_{block_count}"
                sub_model = Model(inputs=self.model.input, outputs=sliced_output, name=model_name)
                sub_model.compile(optimizer=Adam(learning_rate=0.0001, beta_2=0.99), loss='mean_squared_error')
                sub_predicate = ltn.Predicate.FromLogits(sub_model, activation_function="softmax", with_class_indexing=True)
                self.individual_models.append(sub_model)
                # self.individual_models_trainable_parameters.append(sub_model.trainable_variables)
                self.individual_predicates.append(sub_predicate)
                self.individual_predicates_trainable_parameters.append(sub_predicate.trainable_variables)
                if name == "name":
                    self.activity_predicates.append(sub_predicate)
                elif name == "user":
                    self.user_predicates.append(sub_predicate)
                offset += dim
            block_count += 1

        self.individual_predicates_trainable_parameters = self.individual_predicates_trainable_parameters[0] # Just for now
        self.individual_predicates_trainable_parameters = list(itertools.chain.from_iterable(self.individual_predicates_trainable_parameters))
        
        return self.individual_models, self.individual_predicates
    
    def _build_ltn_constants(self):
        """
        Build/Save LTN constants for each attribute in a list
        """
        dataset_activity_mapping = dict(zip(self.dataset.encoders["name"].classes_, self.dataset.encoders["name"].transform(self.dataset.encoders["name"].classes_)))
        for name, mapping in dataset_activity_mapping.items():
            dataset_activity_mapping[name] = ltn.Constant(mapping, trainable=False)
        dataset_user_mapping = dict(zip(self.dataset.encoders["user"].classes_, self.dataset.encoders["user"].transform(self.dataset.encoders["user"].classes_)))
        for user, mapping in dataset_user_mapping.items():
            dataset_user_mapping[user] = ltn.Constant(mapping, trainable=False)
        self.activity_constants = dataset_activity_mapping
        self.user_constants = dataset_user_mapping
        return self.activity_constants, self.user_constants
    
    def _build_ltn_axioms(self):
        """Generate LTN axioms function for the model.

        Returns:
            axioms: tf.function that takes features and labels as input and returns the satisfaction level of the axioms.
        """
        Not = ltn.Wrapper_Connective(ltn.fuzzy_ops.Not_Std())
        And = ltn.Wrapper_Connective(ltn.fuzzy_ops.And_Prod())
        Or = ltn.Wrapper_Connective(ltn.fuzzy_ops.Or_ProbSum())
        Implies = ltn.Wrapper_Connective(ltn.fuzzy_ops.Implies_Reichenbach())
        Forall = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMeanError(p=2),semantics="forall")
        formula_aggregator = ltn.Wrapper_Formula_Aggregator(ltn.fuzzy_ops.Aggreg_pMeanError(p=2))
        
        @tf.function
        def axioms(features, labels=None, training=False):
            traces = ltn.Variable("traces", features)
            axioms = [
                Forall(traces, self.activity_predicates[0]([traces, self.activity_constants["▶"]], training=training))           
            ]
            sat_level = formula_aggregator(axioms).tensor
            return sat_level
        
        self.axioms = axioms
        return self.axioms
    
    def _build_train_test_steps(self):
        self.metrics_dict = {
            'train_sat_kb': tf.keras.metrics.Mean(name='train_sat_kb'),
            'test_sat_kb': tf.keras.metrics.Mean(name='test_sat_kb')
        }
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_2=0.99)
        
        @tf.function
        def train_step(features, labels):
            with tf.GradientTape() as tape:
                sat = self.axioms(features, labels, training=True)
                loss = 1.-sat
            gradients = tape.gradient(loss, self.individual_predicates_trainable_parameters)
            optimizer.apply_gradients(zip(gradients, self.individual_predicates_trainable_parameters))
            sat = self.axioms(features, labels) # compute sat without dropout
            self.metrics_dict['train_sat_kb'](sat)
            
        @tf.function
        def test_step(features, labels):
            # sat
            sat = self.axioms(features, labels)
            self.metrics_dict['test_sat_kb'](sat)
        
        return train_step, test_step
        
    def _split_dataset_2d(self, batch_size):
        """
        Create train and test datasets from the original dataset.
        Args:
            batch_size: int, size of the mini-batch.
        """
        permuted_indices = np.random.permutation(self.dataset.flat_onehot_features_2d.shape[0])
        x_one_hot_2d = self.dataset.flat_onehot_features_2d[permuted_indices]
        split_point = int(x_one_hot_2d.shape[0] * 0.8 )
        x_one_hot_2d_train = x_one_hot_2d[:split_point]
        x_one_hot_2d_test = x_one_hot_2d[split_point:]
        ds_train = tf.data.Dataset.from_tensor_slices((x_one_hot_2d_train,x_one_hot_2d_train)).batch(batch_size)
        ds_test = tf.data.Dataset.from_tensor_slices((x_one_hot_2d_test,x_one_hot_2d_test)).batch(batch_size)
        return ds_train, ds_test

    def fit(
            self, 
            epochs=5,
            batch_size=500,
            track_metrics=10,
            csv_path=None,
            scheduled_parameters=defaultdict(lambda : {})
        ):
        """
        Args:
            epochs: int, number of training epochs.
            metrics_dict: dict, {"metrics_label": tf.keras.metrics instance}.
            ds_train: iterable dataset, e.g. using tf.data.Dataset.
            ds_test: iterable dataset, e.g. using tf.data.Dataset.
            train_step: callable function. the arguments passed to the function
                are the itered elements of ds_train.
            test_step: callable function. the arguments passed to the function
                are the itered elements of ds_test.
            csv_path: (optional) path to create a csv file, to save the metrics.
            scheduled_parameters: (optional) a dictionary that returns kwargs for
                the train_step and test_step functions, for each epoch.
                Call using scheduled_parameters[epoch].
        """
        
        ds_train, ds_test = self._split_dataset_2d(batch_size)
        self._build_main_model()
        # To do: fit main model on data beforehand
        self._fit_main_model(epochs=30, batch_size=batch_size, validation_split=0.1)
        self._build_individual_models()
        self._build_ltn_constants()
        axioms = self._build_ltn_axioms()
        train_step, test_step = self._build_train_test_steps()
        
        
        template = "Epoch {}"
        for metrics_label in self.metrics_dict.keys():
            template += ", %s: {:.4f}" % metrics_label
        # if csv_path is not None:
        #     csv_file = open(csv_path,"w+")
        #     headers = ",".join(["Epoch"]+list(self.metrics_dict.keys()))
        #     csv_template = ",".join(["{}" for _ in range(len(self.metrics_dict)+1)])
        #     csv_file.write(headers+"\n")
        
        for epoch in range(epochs):
            for metrics in self.metrics_dict.values():
                metrics.reset_states()

            for batch_elements in ds_train:
                train_step(*batch_elements)
                # train_step(*batch_elements, axioms, self.individual_predicates_trainable_parameters, **scheduled_parameters[epoch])
            for batch_elements in ds_test:
                test_step(*batch_elements)
                # test_step(*batch_elements, axioms, **scheduled_parameters[epoch])
            metrics_results = [metrics.result() for metrics in self.metrics_dict.values()]
            if epoch%track_metrics == 0:
                print(template.format(epoch,*metrics_results))
        #     if csv_path is not None:
        #         csv_file.write(csv_template.format(epoch,*metrics_results)+"\n")
        #         csv_file.flush()
        # if csv_path is not None:
        #     csv_file.close()
        
    # def model_fn(self, dataset:april.Dataset, **kwargs):
        
    #     hidden_layers = kwargs.pop('hidden_layers')
    #     hidden_size_factor = kwargs.pop('hidden_size_factor')
    #     noise = kwargs.pop('noise')

    #     # Parameters
    #     self.dataset = dataset
    #     self.hidden_layers = hidden_layers if hidden_layers is not None else 2 # 2
    #     self.hidden_size_factor = hidden_size_factor if hidden_size_factor is not None else 0.2 # 0.2
    #     self.noise = noise if noise is not None else True # for denoising autoencoder
    #     self.input_shape = self.dataset.flat_onehot_features_2d.shape[1]
    #     self.dataset_attribute_dims = self.dataset.attribute_dims
    #     # features = dataset.flat_onehot_features_2d
        
    #     self._build_main_model() # build main model
        
    #     return self.model, dataset.flat_onehot_features_2d, dataset.flat_onehot_features_2d  # Features are also targets

    def save(self, file_name: str):
        """Save model to path.

        :param path: Path to save model to.
        """
        if self.model is not None:
            model_file = ModelFile(file_name)
            self._model_path = model_file.str_path
            # self._save(model_file.str_path)
            self.model.save(model_file.str_path)
            return model_file
        else:
            raise RuntimeError(
                'Saving not possible. No model has been trained yet.')
        pass
    
    def load(self, file_name: str):
        """Load model from path.

        :param path: Path to load model from.
        """
        file_name = ModelFile(file_name)
        
        self._model_path = file_name.str_path
        self.model = keras.models.load_model(file_name.str_path)
        
        pass
    
        
    
    def detect(self, dataset):
        """
        Calculate the anomaly score for each event attribute in each trace.
        Anomaly score here is the mean squared error.

        :param traces: traces to predict
        :return:
            scores: anomaly scores for each attribute;
                            shape is (#traces, max_trace_length - 1, #attributes)

        """
        # Get features
        # _, features, _ = self.model_fn(dataset, **self.config)
        features = dataset.flat_onehot_features_2d

        # Parameters
        input_size = int(self.model.input.shape[1])
        features_size = int(features.shape[1])
        if input_size > features_size:
            features = np.pad(features, [(0, 0), (0, input_size - features_size), (0, 0)], mode='constant')
        elif input_size < features_size:
            features = features[:, :input_size]

        # Init anomaly scores array
        scores = np.zeros(dataset.binary_targets.shape)

        # Get predictions
        predictions = self.model.predict(features)

        # Calculate error
        errors = np.power(features - predictions, 2)

        # Split the errors according to the attribute dims
        split = np.cumsum(np.tile(dataset.attribute_dims, [dataset.max_len]), dtype=int)[:-1]
        errors = np.split(errors, split, axis=1)
        errors = np.array([np.mean(a, axis=1) if len(a) > 0 else 0.0 for a in errors])

        for i in range(len(dataset.attribute_dims)):
            error = errors[i::len(dataset.attribute_dims)]
            scores[:, :, i] = error.T

        return AnomalyDetectionResult(scores=scores)
