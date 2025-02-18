import math
from pathlib import Path
import tensorflow as tf
import keras
from keras import layers
from keras.src.layers import StringLookup

from mlp import BaseModel, Params

from transformer_based_recommendation.datasets import PreProcess


class Encoder(PreProcess):
    def __init__(
            self,
            params,
            lookup_features,
            data_path: str | Path,
            dataset_file_names: dict[str, str],
            data_format: str,
            fields: dict[str, str],
            categorical_features: list[str],
            numerical_features: list[str],
            sequence_length: int,
            step_size: int
    ):
        super().__init__(
            data_path,
            dataset_file_names,
            data_format,
            fields,
            categorical_features,
            numerical_features,
            sequence_length,
            step_size
        )
        self.params = params
        self.target_item_id = f"target_{self.product_field_name}"
        self.sequence_item_ids = 'sequence_' + self.product_field_name + 's'
        self.sequence_length = params.get('sequence_length')
        self.positions = tf.range(start=0, limit=self.sequence_length - 1, delta=1)
        self.lookups = {}
        self.lookup_features = lookup_features
        self.item_lookup_features = [self.target_item_id, self.sequence_item_ids]
        self.embedding_encoders = {}
        self.embedding_dims = {}
        self.item_embedding_processor = None
        self.position_embedding_encoder = None

    @classmethod
    def generate(cls, params: Params):
        _cls = Encoder(
            params=params,
            lookup_features=params.get('categorical_features'),
            data_path=params.get('data_path'),
            dataset_file_names=params.get('dataset_file_names'),
            data_format=params.get('data_format'),
            fields=params.get('fields'),
            categorical_features=params.get('categorical_features'),
            numerical_features=params.get('numeric_features'),
            sequence_length=params.get('sequence_length'),
            step_size=params.get('step_size'),
        )
        _cls.get_lookups(
            _cls.train_product_user_data_merge(),
            _cls.product_data
        )
        return _cls

    def update_lookups_and_embeddings(self, vocabulary, lookup):
        self.lookups[lookup] = StringLookup(
            vocabulary=vocabulary, mask_token=None, oov_token=0,  num_oov_indices=1)
        self.embedding_dims[lookup] = int(math.sqrt(len(vocabulary)))
        self.embedding_encoders[lookup] = layers.Embedding(
            input_dim=len(vocabulary ) +1,
            output_dim=self.embedding_dims[lookup],
            name=f"{lookup}_embedding",
        )

    def get_lookups(self, train_data, products):
        for lookup in self.lookup_features:
            if lookup not in self.item_lookup_features:
                # Convert the string input values into integer indices.
                vocabulary = train_data[lookup].astype(str).unique().tolist()
                self.update_lookups_and_embeddings(vocabulary, lookup)

        # item Id embedding and lookups
        vocabulary = products[self.product_field_name].astype(str).unique().tolist()
        self.update_lookups_and_embeddings(vocabulary, self.product_field_name)
        self.item_embedding_processor = layers.Dense(
            units=self.embedding_dims[self.product_field_name],
            activation="relu",
            name=f"process_{self.product_field_name}_embedding",
        )
        self.position_embedding_encoder = layers.Embedding(
            input_dim=self.sequence_length - 1,
            output_dim=self.embedding_dims[self.product_field_name],
            name="position_embedding",
        )

    def query(self, inp, lookup):
        return self.embedding_encoders[lookup](inp)

    def item_embeddings(self, inputs):
        emb_target = self.query(inputs[self.target_item_id], self.product_field_name)
        emb_target = self.item_embedding_processor(emb_target)
        emb_seq = self.query(inputs[self.sequence_item_ids], self.product_field_name)
        emb_seq = self.item_embedding_processor(emb_seq)
        return emb_target, emb_seq

    def get_embeddings(self, inputs):
        encoded = []
        encoded_transformer = []
        for lookup in self.lookup_features:
            if lookup not in self.item_lookup_features:
                print(inputs[lookup])
                encoded.append(self.query(inputs[lookup], lookup))

        ## Create a single embedding vector for the user features
        if len(encoded) > 1:
            encoded = layers.concatenate(encoded)
        elif len(encoded) == 1:
            encoded = encoded[0]
        else:
            encoded = None

        (
            encoded_target_item,
            encoded_sequence_items
        ) = self.item_embeddings(
            inputs
        )
        encodded_positions = self.position_embedding_encoder(self.positions)
        sequence_ratings = keras.ops.expand_dims(inputs["sequence_relevance_scores"], -1)

        encoded_sequence_items_with_poistion_and_rating = layers.Multiply()(
            [(encoded_sequence_items + encodded_positions), sequence_ratings]
        )

        # Construct the transformer inputs.
        for i in range(self.sequence_length - 1):
            feature = encoded_sequence_items_with_poistion_and_rating[:, i, ...]
            feature = keras.ops.expand_dims(feature, 1)
            encoded_transformer.append(feature)

        encoded_transformer = layers.concatenate(
            encoded_transformer, axis=1
        )

        return encoded_transformer, encoded


class Inputs:
    def __init__(
        self,
        params
    ):
        self.params = params
        self.item_id = params.get('product_field_name')
        self.target_item_id = f"target_{params.get('product_field_name')}"
        self.sequence_item_ids = 'sequence_' + params.get('product_field_name') + 's'
        self.sequence_length = params.get('sequence_length')
        self.categorical_features = params.get('categorical_features')
        self.numeric_features = params.get('numeric_features')
        self.inputs = {}
        self.collect_inputs()

    def collect_inputs(self):
        for cat in self.categorical_features:
            if cat == self.sequence_item_ids:
                self.inputs[cat] = keras.Input(
                    name=cat, shape=(self.sequence_length - 1,)
                )
            elif cat == "sequence_relevance_scores":
                self.inputs[cat] = keras.Input(
                    name="sequence_relevance_scores", shape=(self.sequence_length - 1,)
                )
            else:
                self.inputs[cat] = keras.Input(name=cat, shape=(1,))

        for num in self.numeric_features:
            self.inputs[num] = keras.Input(name=num, shape=(1,))


class Transformer:
    def __init__(self, params: Params, inputs: Inputs, encoders: Encoder):
        self.params = params
        self.num_heads = params.get('num_heads')
        self.dropout_rate = params.get('dropout_rate')
        self.inputs = inputs
        self.encoders = encoders
        self.hidden_units = BaseModel.cal_hidden_layer_of_units(
            params.get('hidden_layers'),
            params.get('hidden_units')
        )

    def create_model(self):
        transformer_features, other_features = self.encoders.get_embeddings(
            self.inputs.inputs
        )
        attention_output = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=transformer_features.shape[2],
            dropout=self.dropout_rate
        )(transformer_features, transformer_features)

        # Transformer block.
        attention_output = layers.Dropout(self.dropout_rate)(attention_output)
        x1 = layers.Add()([transformer_features, attention_output])
        x1 = layers.LayerNormalization()(x1)
        x2 = layers.LeakyReLU()(x1)
        x2 = layers.Dense(units=x2.shape[-1])(x2)
        x2 = layers.Dropout(self.dropout_rate)(x2)
        transformer_features = layers.Add()([x1, x2])
        transformer_features = layers.LayerNormalization()(transformer_features)
        features = layers.Flatten()(transformer_features)

        # Included the other features.
        if other_features is not None:
            features = layers.concatenate(
                [features, layers.Reshape([other_features.shape[-1]])(other_features)]
            )

        # Fully-connected layers.
        for num_units in self.hidden_units:
            features = layers.Dense(num_units)(features)
            features = layers.BatchNormalization()(features)
            features = layers.LeakyReLU()(features)
            features = layers.Dropout(self.dropout_rate)(features)

        outputs = layers.Dense(units=1)(features)
        model = keras.Model(inputs=self.inputs.inputs, outputs=outputs)
        return model
