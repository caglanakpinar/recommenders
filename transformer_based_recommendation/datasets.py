from pathlib import Path
import pandas as pd
from statistics import mode

from transformer_based_recommendation.utils import Payload


class Data:
    """datasets for recommendation
    product_data:
        only supports pandas dataframe.
        product ID column must be in the dataset.
        unique products per row.
        product related columns. e.g. product category product name, product value.
    user_data:
        only supports pandas dataframe
        user Id column must be in the dataset.
        unique users per row.
        user related columns. e.g. age, gender.
    train_data:
        only supports pandas dataframe.
        transaction ID, user ID, product ID columns must be in the dataset.
        timestamp column must be in the dataset.
        unique (transaction ID, timestamp). e.g rows of train_data;
            | transaction ID| user_Id | product ID|  quantity  |  product_value | transaction value |
            |---------------|---------|-----------|------------|----------------|-------------------|
            |   order_1     |  u_1    |   p_1     | 1          |    10          |     40            |
            |   order_1     |  u_1    |   p_2     | 2          |    5           |     40            |
            |   order_1     |  u_1    |   p_3     | 1          |    20          |     40            |
            |   order_2     |  u_2    |   p_5     | 1          |    10          |     30            |
            |   order_2     |  u_2    |   p_6     | 2          |    5           |     30            |
            |   order_2     |  u_2    |   p_7     | 5          |    2           |     30            |


        data paths will be given from configs/params.yaml
    """
    product_data: pd.DataFrame = pd.DataFrame()
    user_data: pd.DataFrame = pd.DataFrame()
    train_data: pd.DataFrame = pd.DataFrame()


class ReadData(Data):
    """reading and checking for mandatory columns

    product_data
        product ID column
    user_data
        user ID column
    train_data
        transaction Id column

    this framework only supports csv and parquet files with structured format.
    """
    def __init__(
            self, data_path: str | Path,
            dataset_file_names: dict[str, str],
            data_format: str,

    ):
        self.data_path = Path(data_path)
        self.dataset_file_names = dataset_file_names
        self.data_format = data_format

        if  'product_data' not in self.dataset_file_names.keys():
            raise RuntimeError(
                """
                product_data is missing. 
                In order to run framework product_data dataset is needed.
                please add your product data with product ID column
                """
            )

        if  'user_data' not in self.dataset_file_names.keys():
            raise RuntimeError(
                """
                user_data is missing. 
                In order to run framework user_data dataset is needed.
                please add your user data with product ID column.
                """
            )

        if  'train_data' not in self.dataset_file_names.keys():
            raise RuntimeError(
                """
                train_data are missing. 
                In order to run framework train_datas dataset is needed.
                please add to dataset_file_names to train_data.
                You can only add 4 different train_data datasets. 
                """
            )

        self.read_data()

    def read_data(self) -> None:
        """only supports .csv and .parquet data format
        """
        for  attribute_name, file_name in self.dataset_file_names.items():
            setattr(
                self,
                attribute_name,
                self.reader(
                    self.data_path / Path(f"{file_name}.{self.data_format}")
                )
            )

    @property
    def reader(self) -> callable:
        """only supports .csv and .parquet data format
        :return: pandas csv or parquet reader methods
        """
        if self.data_format == 'csv':
            return pd.read_csv
        if self.data_format == 'parquet':
            return pd.read_parquet


class BasePreProcess:
    """base preprocess that has generic attributes for Preprocess and FeatureEng classes
    product_transaction_cnt:
        number of transaction per product
            key : product ID
            value float -> number of transaction per product
    product_user_cnt:
        number of unique users per product that has transacted with product
            key: product ID
            value: int number of unique users per product
    user_transaction_cnt:
        number of transactions per user
            key: user ID
            value: # of transactions per user
    user_product_cnt:
        number unique products per user
            key: user ID
            value: int number of unique products
    user_relevance_sequence:
        last relevance score per user
            key: user ID
            value: last relevance score sequence, last selected product ID sequence
    """
    product_transaction_cnt: dict[str, float] = {}
    product_user_cnt: dict[str, float] = {}
    user_transaction_cnt: dict[str, float] = {}
    user_product_cnt: dict[str, float] = {}
    null_values: [dict | float] = {}
    user_relevance_sequence: [dict | float] = {}
    numerical_mapping: dict[
        str,
        tuple[
            str | tuple[str, str],
            product_transaction_cnt | product_user_cnt | user_transaction_cnt | user_product_cnt
        ]

    ] = {}
    categorical_mappings: dict[
        str,
        tuple[
            str | tuple[str, str],
            dict
        ]
    ] = {}
    sequence_mapping: dict[
        str,
        tuple[
            str | tuple[str, str],
            dict
        ]
    ] = {}

    @staticmethod
    def min_max_norm(data: pd.Series) -> float:
        _min = data.p_order_cnt.min()
        _max = data.p_order_cnt.max()

        return (
            (data - _min)
            / (_max - _min)
        )

    @staticmethod
    def convert_to_str(x) -> str:
        if 'b' == str(x).split("_")[0]:
            return x
        if type(x) == str:
            return 'b_' + x
        return 'b_' + str(int(x))

    @staticmethod
    def get_ranking(r):
        if r <= .5:
            return 1
        if .5 < r <= 1.5:
            return 2
        if 1.5 < r <= 2:
            return 3
        if 2 < r <= 3.5:
            return 4
        if 3.5 < r:
            return 5

    @staticmethod
    def convert_to_str_sequence(seq):
        return ",".join([str(s) for s in seq])

    @staticmethod
    def create_sequences(values, window_size, step_size):
        sequences = []
        start_index = 0
        while True:
            end_index = start_index + window_size
            seq = values[start_index:end_index]
            if len(seq) < window_size:
                seq = values[-window_size:]
                if len(seq) < window_size:
                    seq = seq + ([seq[-1]] * (window_size - len(seq)))
                    sequences.append(seq)
                break
            sequences.append(seq)
            start_index += step_size
        return sequences


class PreProcess(ReadData, BasePreProcess):
    def __init__(
            self,
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
            data_format
        )
        self.product_field_name = fields['product_field_name']
        self.user_field_name = fields['user_field_name']
        self.transaction_field_name = fields['transaction_field_name']
        self.relevance_field = fields.get('relevance', 'relevance')
        self.timestamp_field_name = fields['timestamp_field_name']
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.sequence_features = [
            "relevance_score",
            self.product_field_name
        ]
        self.sequence_length = sequence_length
        self.step_size = step_size

        if  self.user_field_name not in list(self.user_data.columns):
            raise RuntimeError(
                f"""
                in the file {dataset_file_names["product_data"]},
                user field {self.product_field_name} is missing.
                - User data must have user ID column.
                """
            )

        if  self.product_field_name not in list(self.product_data.columns):
            raise RuntimeError(
                f"""
                in the file {dataset_file_names["user_data"]},
                product field {self.product_field_name} is missing.
                - Product data must have product ID column.
                """
            )

        if (
                self.user_field_name not in list(self.train_data.columns)
                or self.product_field_name not in list(self.train_data.columns)
                or self.transaction_field_name not in list(self.train_data.columns)
                or self.timestamp_field_name not in list(self.train_data.columns)
        ):
            raise RuntimeError(
                f"""
                user ID or product ID or transaction ID or timestamp columns are not train_data.
                """
            )

        if len(
            set(self.categorical_features)
            - set(self.train_data.columns + self.product_data.columns + self.user_data.columns)
        ) != 0:
            raise RuntimeError(
                f"""
                some categorical feature column names are missing one of datasets; here are missing columns;
                    {(
                        set(self.categorical_features) 
                        - set(self.train_data.columns + self.product_data.columns + self.user_data.columns)
                    )}   
                """
            )

        self.get_selection_ordered()
        self.update_categorical_features()
        self.get_product_transaction_cnt()
        self.get_product_user_cnt()
        self.get_user_transaction_cnt()
        self.user_product_cnt()
        self.create_relevance()

        self.get_categorical_mappings()
        self.get_numerical_mappings()

        self.train_data_creation()
        self.get_user_relevance_sequence()
        self.get_sequences_mapping()

    def update_categorical_features(self):
        for cat in self.categorical_features:
            if cat in self.train_data.columns:
                self.train_data[cat] = self.train_data[cat].apply(self.convert_to_str)
            if cat in self.user_data.columns:
                self.user_data[cat] = self.user_data[cat].apply(self.convert_to_str)
            if cat in self.product_data.columns:
                self.product_data[cat] = self.product_data[cat].apply(self.convert_to_str)

    def get_selection_ordered(self):
        """creating selection_ordered column based on position of the clicked or selected products
        """
        self.train_data["selection_ordered"] = (
            self.train_data
            .groupby(self.transaction_field_name)
            .cumcount() + 1
        )

    def get_product_transaction_cnt(self):
        """
        """
        _product_transaction_cnt = (
            self.train_data.groupby(self.product_field_name)
            .agg({self.transaction_field_name: "count"})
            .reset_index()
            .rename(columns={self.transaction_field_name: "p_trans_cnt"})
            .sort_values('p_trans_cnt', ascending=False)
        )

        _product_transaction_cnt['p_trans_cnt_norm'] = self.min_max_norm(
            _product_transaction_cnt['p_trans_cnt']
        )

        self.product_transaction_cnt = (
            _product_transaction_cnt.set_index(self.product_field_name)
            .to_dict()
        )

    def get_product_user_cnt(self):
        """
        """
        _product_user_cnt = (
            self.train_data
            .groupby(self.product_field_name)
            .agg({self.user_field_name: pd.Series.nunique})
            .reset_index()
            .rename(columns={self.user_field_name: "p_user_cnt"})
            .sort_values('p_user_cnt', ascending=False)
        )

        _product_user_cnt['p_user_cnt_norm'] = self.min_max_norm(
            self.product_user_cnt['p_user_cnt']
        )

        self.product_user_cnt = (
            _product_user_cnt.set_index(self.product_field_name)
            .to_dict()
        )

    def get_user_transaction_cnt(self):
        _user_transaction_cnt = (
            self.train_data
            .groupby(self.user_field_name)

            .agg({self.transaction_field_name: pd.Series.nunique})
            .reset_index()
            .rename(columns={self.transaction_field_name: "u_transaction_cnt"})
        )

        _user_transaction_cnt['u_transaction_cnt_norm'] = self.min_max_norm(
            _user_transaction_cnt['u_transaction_cnt']
        )

        self.user_transaction_cnt = (
            _user_transaction_cnt.set_index(self.user_field_name)
            .to_dict()
        )

    def user_product_cnt(self):
        _user_product_cnt = (
            self.train_data
            .groupby(self.user_field_name)
            .agg({self.product_field_name: pd.Series.nunique})
            .reset_index()
            .rename(columns={self.product_field_name: "u_product_cnt"})
        )

        _user_product_cnt['u_product_cnt_norm'] = self.min_max_norm(
            _user_product_cnt['u_product_cnt']
        )

        self.user_product_cnt = (
            _user_product_cnt.set_index(self.user_field_name)
            .to_dict()
        )

    def create_relevance(self):
        self.train_data['relevance_score'] = (
                (.2 * self.train_data['selection_ordered'])
                + (.4 * self.train_data['p_order_cnt_norm'])
                + (.4 * self.train_data['p_user_cnt_norm'])
        )
        self.train_data['rating'] = self.train_data.relevance_score.apply(
            self.get_ranking
        )

    def train_data_creation(self):
        train_data = (
            self.train_data
            .sort_values([
                self.user_field_name,
                self.timestamp_field_name
            ])
            .groupby(self.user_field_name)
        )
        train_data = pd.DataFrame(
            {
                self.user_field_name: list(train_data.groups.keys()),
                self.product_field_name: list(train_data[self.product_field_name].apply(list)),
                "relevance_score": list(train_data['relevance_score'].apply(list))
            }
        )
        train_data[f"sequence_{self.product_field_name}s"] = train_data[self.product_field_name].apply(
            lambda row:
            self.create_sequences(
                row,
                self.sequence_length,
                self.step_size
            )
        )
        train_data[f"sequence_relevance_scores"] = train_data['relevance_score'].apply(
            lambda row:
            self.create_sequences(
                row,
                self.sequence_length,
                self.step_size
            )
        )
        self.train_data = (
            train_data
            [[
                self.user_field_name,
                f"sequence_{self.product_field_name}s",
                "sequence_relevance_scores"
            ]]
            .explode([
                f"sequence_{self.product_field_name}s",
                "sequence_relevance_scores"
            ])
        )
        self.train_data[[
            f"sequence_{self.product_field_name}s",
            'sequence_relevance_scores',
            self.product_field_name,
            'relevance_score'
        ]] = self.train_data.apply(
            lambda row:
            pd.Series([
                row[f"sequence_{self.product_field_name}s"][:-1],
                row['sequence_relevance_scores'][:-1],
                row[f"sequence_{self.product_field_name}s"][-1],
                row['sequence_relevance_scores'][-1]
            ]),
            axis=1
        )

    def get_user_relevance_sequence(self):
        self.user_relevance_sequence = (
            self.train_data
            .sort_values([self.product_field_name, self.timestamp_field_name])
            .groupby(self.user_field_name)
            [['sequence_relevance_score', f'sequence_{self.product_field_name}']]
            .agg("last")
            .reset_index()
            .set_index(self.user_field_name)
            .to_dict()
        )

    def get_mapping_from_datasets(self, f) -> tuple[str | tuple, dict[str | tuple, str | float]]:
        _key = [self.product_field_name, self.user_field_name]
        _data = self.train_data
        if f in self.user_data.columns:
            _key = self.user_field_name
            _data = self.user_data
        if f in self.product_data.solumns:
            _key = self.product_field_name
            _data = self.product_data
        return (
            _key,
            (
                _data
                .groupby(_key)
                [f]
                .agg('first')
                .reset_index()
                .set_index(_key)
            )
        )

    def get_categorical_mappings(self):
        """mappings for online/offline feature creation - categorical
        1. finding which dataset has the feature column.
        check product_field_name and user_field_name
            mapping key:
                if dataset hast both product_field_name & user_field_name -> (product_field_name, user_field_name)
                if data has product_field_name -> product_field_name
                if data has user_field_name -> user_field_name
        """
        for f in self.categorical_features:
            if f != self.product_field_name and f != self.user_field_name:
                self.categorical_mappings[f] = self.get_mapping_from_datasets(f)

    def get_numerical_mappings(self):
        """mappings for online/offline feature creation -numerical
        there are generic numerical features coming from
            - product_transaction_cnt
            - product_user_cnt
            - user_transaction_cnt
            - user_product_cnt
        there are also coming from configs/params.yaml
        how to find features

        1. finding which dataset has the feature column.
        check product_field_name and user_field_name
            mapping key:
                if dataset hast both product_field_name & user_field_name -> (product_field_name, user_field_name)
                if data has product_field_name -> product_field_name
                if data has user_field_name -> user_field_name
        """
        for _key, _mapping in [
            (self.product_field_name, self.product_transaction_cnt),
            (self.product_field_name, self.product_user_cnt),
            (self.user_field_name, self.user_transaction_cnt),
            (self.user_field_name, self.user_product_cnt),
        ]:
            for f, _value in _mapping.items():
                self.numerical_mapping[f] = (
                    _key,
                    _mapping[f]
                )

        for f, _value in self.numerical_features:
            if f not in self.numerical_mapping:
                self.numerical_mapping[f] = self.get_mapping_from_datasets(f)

    def get_sequences_mapping(self):
        self.sequence_mapping = {
            f: (self.user_field_name, m)
            for f, m in self.user_relevance_sequence.items()
        }


class FeatureEng(PreProcess):
    def __init__(
            self,
            data_path: str | Path,
            dataset_file_names: dict[str, str], data_format: str,
            fields: dict[str, str],
            categorical_features: list[str],
            numeric_features: list[str],
            sequence_length: int,
            step_size: int
    ):
        super().__init__(
            data_path,
            dataset_file_names,
            data_format,
            fields,
            categorical_features,
            numeric_features,
            sequence_length,
            step_size
        )
        self.numeric_features = numeric_features
        self.create_offline_features_and_mappings()
        self.get_null_values()

    def create_offline_features_and_mappings(self):
        """training data set creation
        """
        for f, mappings in self.numerical_mapping.items():
            for key, mapping in mappings:
                self.train_data[f] = self.train_data[key].apply(
                    mapping[f]
                )

        for f, mappings in self.categorical_mappings.items():
            for key, mapping in mappings:
                self.train_data[f] = self.train_data[key].apply(
                    mapping[f]
                )

    def create_online_features_and_mappings(self, payload: Payload) -> dict[str, float | str]:
        features_mappings = {}
        for mappings in [self.numerical_mapping, self.categorical_mappings, self.sequence_mapping]:
            for f, key_feature_map in mappings.items():
                for key, mapping in key_feature_map:
                    mapping_key = self.convert_to_str(
                        payload.query_product
                        if key == self.product_field_name
                        else payload.user
                    )
                    features_mappings[f] = (
                        mapping[
                            mapping_key
                        ]
                    )
                    if f in self.categorical_features:
                        features_mappings[f] = features_mappings[f]
        return features_mappings

    def get_null_values(self):
        """this is null value implementation for numerical features
        when numerical features are not available on numeric_mappings, collect from null_values
        :return:
        """
        for key, mapping in self.numerical_mapping:
            for f in mapping.keys():
                self.null_values[f] = mode(self.train_data[self.train_data[f] == self.train_data[f]][f])





