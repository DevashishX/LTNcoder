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

import itertools
import numpy as np

from april.anomalydetection.binet.core import NNAnomalyDetector
from april.anomalydetection.utils.result import AnomalyDetectionResult
from april.enums import Base
from april.enums import Heuristic
from april.enums import Mode
from april.enums import Strategy
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, GaussianNoise, Reshape, Flatten, Concatenate, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("GPU found")
    print("Memory growth set")
else:
    print("No GPU found")
# from keras.layers import Lambda
# from keras.models import Model
# from keras.optimizers import Adam
import ltn

class DAE(NNAnomalyDetector):
    """Implements a denoising autoencoder based anomaly detection algorithm."""

    abbreviation = 'dae'
    name = 'DAE'

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
        super(DAE, self).__init__(model=model)

    @staticmethod
    def model_fn(dataset, **kwargs):
        # Import keras locally
        from keras.layers import Input, Dense, Dropout, GaussianNoise
        from keras.models import Model
        from keras.optimizers import Adam

        hidden_layers = kwargs.pop('hidden_layers')
        hidden_size_factor = kwargs.pop('hidden_size_factor')
        noise = kwargs.pop('noise')

        features = dataset.flat_onehot_features_2d

        # Parameters
        input_size = features.shape[1]

        # Input layer
        input = Input(shape=(input_size,), name='input')
        x = input

        # Noise layer
        if noise is not None:
            x = GaussianNoise(noise)(x)

        # Hidden layers
        for i in range(hidden_layers):
            if isinstance(hidden_size_factor, list):
                factor = hidden_size_factor[i]
            else:
                factor = hidden_size_factor
            x = Dense(int(input_size * factor), activation='relu', name=f'hid{i + 1}')(x)
            x = Dropout(0.5)(x)

        # Output layer
        output = Dense(input_size, activation='sigmoid', name='output')(x)

        # Build model
        model = Model(inputs=input, outputs=output)

        # Compile model
        model.compile(
            # optimizer=Adam(lr=0.0001, beta_2=0.99),
            optimizer=Adam(learning_rate=0.0001, beta_2=0.99),
            loss='mean_squared_error',
            metrics=['accuracy']
        )

        return model, features, features  # Features are also targets

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

class DAELTN(NNAnomalyDetector):
    """Implements a denoising autoencoder based anomaly detection algorithm."""

    abbreviation = 'daeltn'
    name = 'DAELTN'

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
        super(DAELTN, self).__init__(model=model)

    @staticmethod
    def model_fn(dataset, **kwargs):
        # Import keras locally

        hidden_layers = kwargs.pop('hidden_layers')
        hidden_size_factor = kwargs.pop('hidden_size_factor')
        noise = kwargs.pop('noise')

        features = dataset.flat_onehot_features_2d

        # Parameters
        input_size = features.shape[1]

        # Input layer
        input = Input(shape=(input_size,), name='input')
        x = input

        # Noise layer
        if noise is not None:
            x = GaussianNoise(noise, name="noise")(x)

        # Hidden layers
        for i in range(hidden_layers):
            if isinstance(hidden_size_factor, list):
                factor = hidden_size_factor[i]
            else:
                factor = hidden_size_factor
            x = Dense(int(input_size * factor), activation='elu', name=f'hidden-{i}')(x)
            x = Dropout(0.5, name=f"dropout-{i}")(x)

        # Output layer
        output = Dense(input_size, activation='sigmoid', name='output')(x)

        # Build model
        model = Model(inputs=input, outputs=output, name="main-DAE")

        # Compile model
        model.compile(
            # optimizer=Adam(lr=0.0001, beta_2=0.99),
            optimizer=Adam(learning_rate=0.0001, beta_2=0.99),
            loss='mean_squared_error',
            metrics=['accuracy']
        )

        return model, features, features  # Features are also targets

    def _build_individual_models(self, dataset):
        """
        Build sub-models from a main model, each returning a specific attribute slice.
        
        Parameters:
            self.model: The original Keras model.
            self.dataset.attribute_dims: List of ints — the size of each one-hot encoded attribute.
            self.dataset.attribute_keys: List of strings — names of each attribute.

        Returns:
            A list of Keras models with proper .name set (e.g., 'name_0', 'user_0', ...).
        """
        
        
        assert len(dataset.attribute_dims) == len(dataset.attribute_keys), "Dims and names must match."

        total_output_dim = self._model.output.shape[-1]
        offset = 0
        block_count = 0
        self.individual_models = []
        # self.individual_models_trainable_parameters = []
        self.individual_predicates = []
        self.individual_predicates_trainable_parameters = []
        self.activity_predicates = []
        self.user_predicates = []

        while offset < total_output_dim:
            for dim, name in zip(dataset.attribute_dims.astype(int), dataset.attribute_keys):
                if offset + dim > total_output_dim:
                    break  # Avoid slicing beyond output size

                # Slice the output
                sliced_output = Lambda(lambda x, s=offset, e=offset+dim: x[:, s:e])(self._model.output)

                # Create a model with a custom name
                model_name = f"{name}_{block_count}"
                sub_model = Model(inputs=self._model.input, outputs=sliced_output, name=model_name)
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

        # self.individual_predicates_trainable_parameters = self.individual_predicates_trainable_parameters[0]
        # print(self.individual_predicates_trainable_parameters)# Just for now
        self.individual_predicates_trainable_parameters = list(itertools.chain.from_iterable(self.individual_predicates_trainable_parameters))
        
        return self.individual_models, self.individual_predicates

    def _build_ltn_constants(self, dataset):
        """
        Build/Save LTN constants for each attribute in a list
        """
        dataset_activity_mapping = dict(zip(dataset.encoders["name"].classes_, dataset.encoders["name"].transform(dataset.encoders["name"].classes_)))
        for name, mapping in dataset_activity_mapping.items():
            dataset_activity_mapping[name] = ltn.Constant(mapping, trainable=False)
        dataset_user_mapping = dict(zip(dataset.encoders["user"].classes_, dataset.encoders["user"].transform(dataset.encoders["user"].classes_)))
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
            for i in range(1, len(self.activity_predicates)):
                axioms.append(
                    Forall(traces, Not(self.activity_predicates[i]([traces, self.activity_constants["▶"]], training=training)))
                )
            sat_level = formula_aggregator(axioms).tensor
            return sat_level
        
        self.axioms = axioms
        return self.axioms

    def _split_dataset_2d(self, dataset, batch_size):
        """
        Create train and test datasets from the original dataset.
        Args:
            batch_size: int, size of the mini-batch.
        """
        permuted_indices = np.random.permutation(dataset.flat_onehot_features_2d.shape[0])
        x_one_hot_2d = dataset.flat_onehot_features_2d[permuted_indices]
        split_point = int(x_one_hot_2d.shape[0] * 0.8 )
        x_one_hot_2d_train = x_one_hot_2d[:split_point]
        x_one_hot_2d_test = x_one_hot_2d[split_point:]
        ds_train = tf.data.Dataset.from_tensor_slices((x_one_hot_2d_train,x_one_hot_2d_train)).batch(batch_size)
        ds_test = tf.data.Dataset.from_tensor_slices((x_one_hot_2d_test,x_one_hot_2d_test)).batch(batch_size)
        return ds_train, ds_test

    def _build_train_test_steps(self):
        self.metrics_dict = {
            'train_sat_kb': tf.keras.metrics.Mean(name='train_sat_kb'),
            'test_sat_kb': tf.keras.metrics.Mean(name='test_sat_kb')
        }
        self.train_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_2=0.99)
        
        @tf.function
        def train_step(features, labels):
            with tf.GradientTape() as tape:
                sat = self.axioms(features, labels, training=True)
                loss = 1.-sat
            gradients = tape.gradient(loss, self.individual_predicates_trainable_parameters)
            self.train_optimizer.apply_gradients(zip(gradients, self.individual_predicates_trainable_parameters))
            sat = self.axioms(features, labels) # compute sat without dropout
            self.metrics_dict['train_sat_kb'](sat)
            
        @tf.function
        def test_step(features, labels):
            # sat
            sat = self.axioms(features, labels)
            self.metrics_dict['test_sat_kb'](sat)
        
        return train_step, test_step    


    def fit(self, dataset, epochs=20, batch_size=100, validation_split=0.2, epochs_ltn=5, **kwargs):
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
        
        ds_train, ds_test = self._split_dataset_2d(dataset, batch_size)
        self._build_individual_models(dataset)
        self._build_ltn_constants(dataset)
        axioms = self._build_ltn_axioms()
        train_step, test_step = self._build_train_test_steps()
        
        # Train LTN model
        template = "Epoch {}"
        for metrics_label in self.metrics_dict.keys():
            template += ", %s: {:.4f}" % metrics_label
        
        for epoch in range(epochs_ltn):
            for metrics in self.metrics_dict.values():
                metrics.reset_states()

            for batch_elements in ds_train:
                train_step(*batch_elements)
                # train_step(*batch_elements, axioms, self.individual_predicates_trainable_parameters, **scheduled_parameters[epoch])
            for batch_elements in ds_test:
                test_step(*batch_elements)
                # test_step(*batch_elements, axioms, **scheduled_parameters[epoch])
            metrics_results = [metrics.result() for metrics in self.metrics_dict.values()]
            # if epoch%track_metrics == 0:
            print(template.format(epoch,*metrics_results))

        
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
