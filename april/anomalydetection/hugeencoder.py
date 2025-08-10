import pickle
import sys
import numpy as np
np.random.seed(0)

from april.anomalydetection.binet.core import NNAnomalyDetector
from april.anomalydetection.utils.result import AnomalyDetectionResult
from april.anomalydetection.autoencoder import DAE

from april.dataset import Dataset
from april.enums import Base
from april.enums import Heuristic
from april.enums import Mode
from april.enums import Strategy
from april.anomalydetection.axiombuilder import AxiomBuilder
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, GaussianNoise, Reshape, Flatten, Concatenate, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

import ltn
formula_aggregator = ltn.Wrapper_Formula_Aggregator(ltn.fuzzy_ops.Aggreg_pMeanError(p=2))

ltn_rows_file = 'huge_ltn_rows.pkl'
huge_ltn_class_row_values = [10, 25] + list(range(50, 301, 50))
# huge_ltn_class_row_values = [10, 25]
huge_ltn_row_classes = []
huge_leaky_class_row_values = [10, 25] + list(range(50, 301, 50))
# huge_leaky_class_row_values = [10, 25]
huge_leaky_row_classes = []

class HugeDAE(NNAnomalyDetector):
    """Implements a denoising autoencoder based anomaly detection algorithm."""

    abbreviation = f'hugedae'
    name = f'HugeDAE'
    leaky_ltn_rows = 0 # leaked rows from the LTN dataset

    supported_heuristics = [Heuristic.BEST, Heuristic.ELBOW_DOWN, Heuristic.ELBOW_UP,
                            Heuristic.LP_LEFT, Heuristic.LP_MEAN, Heuristic.LP_RIGHT,
                            Heuristic.MEAN, Heuristic.MEDIAN, Heuristic.RATIO, Heuristic.MANUAL]
    supported_strategies = [Strategy.SINGLE, Strategy.ATTRIBUTE, Strategy.POSITION, Strategy.POSITION_ATTRIBUTE]
    supported_modes = [Mode.BINARIZE]
    supported_bases = [Base.LEGACY, Base.SCORES]
    supports_attributes = True

    config = dict(hidden_layers=3,
                  hidden_size_factor=.2,
                  noise=None)

    def __init__(self, model=None):
        """Initialize DAE model."""
        super(HugeDAE, self).__init__(model=model)

    @classmethod
    def model_fn(cls, dataset, **kwargs):
        # Import keras locally
        from keras.layers import Input, Dense, Dropout, GaussianNoise
        from keras.models import Model
        from keras.optimizers import Adam
        import pickle

        # Load ltn rows
        with open(ltn_rows_file, 'rb') as f:
            ltn_rows = pickle.load(f)
            # print(ltn_rows)
            # print(f"Length of LTN rows which will be excluded: {len(ltn_rows)}")

        # use the leaky_ltn_rows to reduce the number of rows in ltn_rows
        if cls.leaky_ltn_rows > 0:
            # remove sampled leaky_ltn_rows from ltn_rows
            # ltn_rows = np.random.choice(ltn_rows, len(ltn_rows) - cls.leaky_ltn_rows, replace=False)
            # just remove the leaky_ltn_rows from ltn_rows
            ltn_rows = ltn_rows[:-cls.leaky_ltn_rows]
            print(f"Length of LTN rows which will be excluded: {len(ltn_rows)}")

        # Filter out rows in ltn_rows
        all_indices = set(range(len(dataset.flat_onehot_features_2d)))
        non_ltn_rows = sorted(list(all_indices - set(ltn_rows)))
        features = dataset.flat_onehot_features_2d[non_ltn_rows]
        # print(f"Length of training features: {len(features)}")

        # Parameters
        hidden_layers = kwargs.pop('hidden_layers')
        hidden_size_factor = kwargs.pop('hidden_size_factor')
        noise = kwargs.pop('noise')
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
            optimizer=Adam(learning_rate=0.0001, beta_2=0.99),
            loss='mean_squared_error',
            metrics=['accuracy']
        )

        return model, features, features  # Features are also targets

    def detect(self, dataset:Dataset):
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
        # print(f"Length of features in detect loaded directly from dataset: {len(features)}")

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

class HugeLTN(NNAnomalyDetector):
    """Implements a denoising autoencoder based anomaly detection algorithm."""

    abbreviation = f'Hugeltn'
    name = f'HugeLTN'
    ltn_rows = 351 # rows to use for LTN training
    # ltn_fraction = 1.0 # fraction of the dataset to use for LTN training
    
    supported_heuristics = [Heuristic.BEST, Heuristic.ELBOW_DOWN, Heuristic.ELBOW_UP,
                            Heuristic.LP_LEFT, Heuristic.LP_MEAN, Heuristic.LP_RIGHT,
                            Heuristic.MEAN, Heuristic.MEDIAN, Heuristic.RATIO, Heuristic.MANUAL]
    supported_strategies = [Strategy.SINGLE, Strategy.ATTRIBUTE, Strategy.POSITION, Strategy.POSITION_ATTRIBUTE]
    supported_modes = [Mode.BINARIZE]
    supported_bases = [Base.LEGACY, Base.SCORES]
    supports_attributes = True

    config = dict(hidden_layers=3,
                  hidden_size_factor=.2,
                  noise=None)

    def __init__(self, model=None):
        super(HugeLTN, self).__init__(model=model)

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
        self.individul_model_optimizer = Adam(learning_rate=0.0001, beta_2=0.99)
        while offset < total_output_dim:
            for dim, name in zip(dataset.attribute_dims.astype(int), dataset.attribute_keys):
                if offset + dim > total_output_dim:
                    break  # Avoid slicing beyond output size

                # Slice the output
                sliced_output = Lambda(lambda x, s=offset, e=offset+dim: x[:, s:e])(self._model.output)

                # Create a model with a custom name
                model_name = f"{name}_{block_count}"
                sub_model = Model(inputs=self._model.input, outputs=sliced_output, name=model_name)
                sub_model.compile(optimizer=self.individul_model_optimizer, loss='mean_squared_error')
                sub_predicate = ltn.Predicate.FromLogits(sub_model, activation_function="softmax", with_class_indexing=True)
                # self.individual_models.append(sub_model)
                # self.individual_models_trainable_parameters.append(sub_model.trainable_variables)
                self.individual_predicates.append(sub_predicate)
                # self.individual_predicates_trainable_parameters.append(sub_predicate.trainable_variables)
                if name == "name":
                    self.activity_predicates.append(sub_predicate)
                elif name == "user":
                    self.user_predicates.append(sub_predicate)
                offset += dim
            block_count += 1

        # self.individual_predicates_trainable_parameters = self.individual_predicates_trainable_parameters[0]
        # # print(self.individual_predicates_trainable_parameters)# Just for now
        # self.all_trainable_parameters = flatten_iterable(self.individual_predicates_trainable_parameters)
        
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
        # print(f"Activity constants: {dataset_activity_mapping}")
        # print(f"User constants: {dataset_user_mapping}")
        self.activity_constants = dataset_activity_mapping
        self.user_constants = dataset_user_mapping
        return self.activity_constants, self.user_constants

    def _build_ltn_axioms(self):
        """
        Generate and compile LTN axioms function for the model.

        This method ensures that the axioms are compiled only once and reused during training.

        Returns:
            axioms_fn: A precompiled TensorFlow function that evaluates the satisfaction level of the axioms.
        """
        # Initialize the AxiomBuilder with predicates and constants
        self.axiombuilder = AxiomBuilder(self.activity_predicates, self.activity_constants)

        # Build the axioms once and store them
        @tf.function
        def axioms_fn(features, labels=None, training=False):
            traces = ltn.Variable("traces", features)
            axioms = self.axiombuilder._build_axioms(traces, training=training)
            sat_level = formula_aggregator(axioms).tensor
            return sat_level

        # Store the compiled axioms function
        self.axioms_fn = axioms_fn
        return self.axioms_fn

    def _split_dataset_for_LTN(self, dataset: Dataset):
        """
        Remove non-anomalous ltn rows from the dataset and return two subsets.
        
        Args:
            dataset: The original dataset.
        
        Returns:
            x_one_hot_2d: Rows in dataset.flat_onehot_features_2d without the indexes in ltn_rows.
            x_one_hot_2d_LTN: Rows in dataset.flat_onehot_features_2d with the indexes in ltn_rows.
        """
        # Load the ltn rows from the pickle file
        with open(ltn_rows_file, 'rb') as f:
            ltn_rows = pickle.load(f)
        
        # Remove anomaly_indices from ltn_rows
        # ltn_rows = [i for i in ltn_rows if i not in dataset.anomaly_indices]
        
        # Create masks for rows to include and exclude
        all_indices = set(range(len(dataset.flat_onehot_features_2d)))
        non_ltn_rows = list(all_indices - set(ltn_rows))
        
        # Subset the dataset
        x_one_hot_2d = dataset.flat_onehot_features_2d[non_ltn_rows]
        x_one_hot_2d_LTN = dataset.flat_onehot_features_2d[ltn_rows]
        # sample x_one_hot_2d_LTN into a smaller dataset of sample_dataset fraction
        x_one_hot_2d_LTN = x_one_hot_2d_LTN[np.random.choice(x_one_hot_2d_LTN.shape[0], self.ltn_rows, replace=False)]
        
        return x_one_hot_2d, x_one_hot_2d_LTN

    def _split_dataset_2d(self, x_one_hot_2d, batch_size):
        """
        Create train and test datasets from the original dataset.
        Args:
            dataset: The original dataset.
            batch_size: int, size of the mini-batch.
            sample_dataset: float, fraction of the dataset to sample.
        """
        
        permuted_indices = np.random.permutation(x_one_hot_2d.shape[0])
        x_one_hot_2d = x_one_hot_2d[permuted_indices]
        tf.print("Size of LTN training dataset: ", x_one_hot_2d.shape[0])
        # sample x_one_hot_2d into a smaller dataset of sample_dataset fraction
        # x_one_hot_2d = x_one_hot_2d[np.random.choice(x_one_hot_2d.shape[0], int(x_one_hot_2d.shape[0] * sample_dataset), replace=False)]
        split_point = int(x_one_hot_2d.shape[0] * 0.9 )
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
        self.ltn_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_2=0.99)
        
        @tf.function
        def train_step(features, labels, axioms_fn):
            with tf.GradientTape() as tape:
                sat = axioms_fn(features, labels, training=True)
                loss = 1.0 - sat

            gradients = tape.gradient(loss, self.individual_predicates[0].trainable_variables)
            self.ltn_optimizer.apply_gradients(zip(gradients, self.individual_predicates[0].trainable_variables))
            sat = axioms_fn(features, labels, training=False)  # Compute satisfaction without dropout
            self.metrics_dict['train_sat_kb'](sat)
        
        @tf.function
        def test_step(features, labels, axioms_fn):
            """
            Evaluate the satisfaction level of the axioms on the test dataset.

            Args:
                features: Input features for the test dataset.
                labels: Labels for the test dataset (not used in this case).
                axioms_fn: Precompiled TensorFlow function for evaluating axioms.
            """
            # Compute satisfaction level without training mode (no dropout, etc.)
            sat = axioms_fn(features, labels, training=False)
            self.metrics_dict['test_sat_kb'](sat)
        
        return train_step, test_step    


    def fit(self, dataset, epochs=20, batch_size=100, validation_split=0.2, epochs_ltn=5, **kwargs):
        # Build model
        normal_dataset_flat_one_hot_2d, ltn_dataset_flat_one_hot_2d = self._split_dataset_for_LTN(dataset)
        # tf.print("Size of Normal training dataset: ", normal_dataset_flat_one_hot_2d.shape[0])
        # tf.print("Size of LTN training dataset: ", ltn_dataset_flat_one_hot_2d.shape[0])
        
        self._model, _, _ = self.model_fn(dataset, **self.config)

        # Train model
        self.history = self._model.fit(
            normal_dataset_flat_one_hot_2d,
            normal_dataset_flat_one_hot_2d,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            **kwargs
        )
        
        ds_train, ds_test = self._split_dataset_2d(ltn_dataset_flat_one_hot_2d, batch_size)
        self._build_individual_models(dataset)
        self._build_ltn_constants(dataset)
        axioms_fn = self._build_ltn_axioms()  # Compile axioms once
        train_step, test_step = self._build_train_test_steps()
        
        # Train LTN model
        print(f"Training LTN model {self.name}")
        template = "Epoch {}"
        for metrics_label in self.metrics_dict.keys():
            template += ", %s: {:.4f}" % metrics_label
        
        for epoch in range(epochs_ltn):
            for metrics in self.metrics_dict.values():
                metrics.reset_states()

            for batch_elements in ds_train:
                train_step(*batch_elements, axioms_fn)

            for batch_elements in ds_test:
                test_step(*batch_elements, axioms_fn)

            metrics_results = [metrics.result() for metrics in self.metrics_dict.values()]
            print(template.format(epoch, *metrics_results))

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
        # _, features, _ = self.model_fn(dataset, **self.config)
        features = dataset.flat_onehot_features_2d
        # print(f"Length of features in detect loaded directly from dataset: {len(features)}")

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

class HugeLTNFROZEN(HugeLTN):
    """Implements a denoising autoencoder based anomaly detection algorithm."""

    abbreviation = f'hugeltnfrozen'
    name = f'HugeLTNFROZEN'

    def __init__(self, model=None):
        super(HugeLTNFROZEN, self).__init__(model=model)

    def fit(self, dataset, epochs=20, batch_size=100, validation_split=0.2, epochs_ltn=5, **kwargs):
        """
        Train the DAELTNFROZEN model with both normal and LTN datasets.
        
        Args:
            dataset: The dataset to train on.
            epochs: Number of epochs for the normal training phase.
            batch_size: Batch size for training.
            validation_split: Fraction of the dataset to use for validation.
            epochs_ltn: Number of epochs for the LTN training phase.
            **kwargs: Additional arguments for the training process.
        """
        # Split dataset into normal and LTN subsets
        normal_dataset_flat_one_hot_2d, ltn_dataset_flat_one_hot_2d = self._split_dataset_for_LTN(dataset)
        # tf.print("Size of normal training dataset: ", normal_dataset_flat_one_hot_2d.shape[0])
        # tf.print("Size of LTN training dataset: ", ltn_dataset_flat_one_hot_2d.shape[0])
        
        # Build and train the main model
        self._model, _, _ = self.model_fn(dataset, **self.config)
        self.history = self._model.fit(
            normal_dataset_flat_one_hot_2d,
            normal_dataset_flat_one_hot_2d,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            **kwargs
        )
        
        # Freeze all layers except the last one
        for layer in self._model.layers[:-1]:
            layer.trainable = False
        
        # Recompile the model (optimizer here is irrelevant for LTN training)
        self._model.compile(
            optimizer=Adam(learning_rate=0.0001, beta_2=0.99),
            loss=self._model.loss,
            metrics=self._model.metrics
        )
        
        # Prepare LTN training
        ds_train, ds_test = self._split_dataset_2d(ltn_dataset_flat_one_hot_2d, batch_size)
        self._build_individual_models(dataset)
        self._build_ltn_constants(dataset)
        axioms_fn = self._build_ltn_axioms()  # Compile axioms once
        train_step, test_step = self._build_train_test_steps()
        
        # Train the LTN model
        print(f"Training LTN model {self.name}")
        template = "Epoch {}"
        for metrics_label in self.metrics_dict.keys():
            template += ", %s: {:.4f}" % metrics_label
        
        for epoch in range(epochs_ltn):
            # Reset metrics at the start of each epoch
            for metrics in self.metrics_dict.values():
                metrics.reset_states()

            # Train on the training dataset
            for batch_elements in ds_train:
                train_step(*batch_elements, axioms_fn)

            # Evaluate on the test dataset
            for batch_elements in ds_test:
                test_step(*batch_elements, axioms_fn)

            # Log metrics for the epoch
            metrics_results = [metrics.result() for metrics in self.metrics_dict.values()]
            print(template.format(epoch, *metrics_results))
        
        return self.history

# Factory function to create and register classes globally
def create_ltn_classes(base_class, row_values, target_module):
    for rows in row_values:
        class_name = f"{base_class.__name__}-{rows}"
        new_class = type(
            class_name,
            (base_class,),
            {
                '__doc__': f"Implements a denoising autoencoder based anomaly detection algorithm and LTN using only {rows} rows of LTN training data.",
                'abbreviation': f'{base_class.__name__.lower()}-{rows}',
                'name': f'{base_class.__name__}-{rows}',
                'ltn_rows': rows
            }
        )
        setattr(target_module, class_name, new_class)  # Register to actual module
        huge_ltn_row_classes.append(new_class)

# Factory function to create and register classes globally
def create_leaky_classes(base_class, leaky_values, target_module):
    for leaky in leaky_values:
        class_name = f"{base_class.__name__}-Leaky-{leaky}"
        new_class = type(
            class_name,
            (base_class,),
            {
                '__doc__': f"Implements a denoising autoencoder based anomaly detection algorithm with leaked {leaky} rows of LTN training data.",
                'abbreviation': f'{base_class.__name__.lower()}-leaky-{leaky}',
                'name': f'{base_class.__name__}-Leaky-{leaky}',
                'leaky_ltn_rows': leaky
            }
        )
        setattr(target_module, class_name, new_class)  # Register to actual module
        huge_leaky_row_classes.append(new_class)
# Create and register the classes
create_ltn_classes(HugeLTNFROZEN, huge_ltn_class_row_values, sys.modules[__name__])
create_leaky_classes(HugeDAE, huge_leaky_class_row_values, sys.modules[__name__])