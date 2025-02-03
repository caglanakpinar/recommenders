import keras
from mlp import Params


from models import Encoder
from transformer_based_recommendation.datasets import FeatureEng


class Payload:
    candidates_p: list[str]
    query_product: str
    user: str
    ts: str


class Prediction:
    def __init__(
            self,
            params: Params,
            encoders: Encoder,
            features: FeatureEng,
            model: keras.Model
    ):
        self.params = params
        self.encoders = encoders
        self.features = features
        self.model: keras.Model = model
        self.product_field_name = params.get('product_field_name')
        self.user_field_name = params.get('user_field_name')
        self.categorical_features = params.get('categorical_features')
        self.numeric_features = params.get('numerical_features')
        self.feature_names = [f for f in model.input]

    def query_lookups(self, lookup_name, lookup_value):
        return (
            self.encoders
            .lookups
            [lookup_name]
            (lookup_value)
        )

    def get_lookup_key_value_from_payload(
            self, f, payload: Payload
    ) -> tuple[str | None, str | list | None]:
        if f == self.encoders.target_item_id:
            return self.product_field_name, payload.query_product
        if f == self.encoders.sequence_item_ids:
            return self.product_field_name, payload.candidates_p
        if f == self.user_field_name:
            return self.user_field_name, payload.user
        return None, None

    def get_online_features(self, payload: Payload) -> list[dict[str, float]]:
        inputs = []
        for query_product in payload.candidates_p:
            payload.query_product = query_product
            features_mappings = self.features.create_online_features_and_mappings(
                payload
            )
            features = {}
            for f in self.feature_names:
                if f in self.encoders.lookup_features:
                    _key, value = self.get_lookup_key_value_from_payload(f, payload)
                    if _key is not None:
                        features[f] = self.query_lookups(
                            lookup_name=_key,
                            lookup_value=self.features.convert_to_str(value)
                        )
                    else:
                        features[f] = self.query_lookups(
                            lookup_name=f,
                            lookup_value=features_mappings[f]
                        )
                else:
                    features[f] = features_mappings[f]
            inputs.append(features)
        return inputs


    def predict_relevance_scores(self, payload: Payload):
        return self.model.predict(
            self.get_online_features(payload)
        )




