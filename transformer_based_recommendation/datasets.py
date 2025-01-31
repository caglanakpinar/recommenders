from pathlib import Path
import pandas as pd
from statistics import mode
from pandas.core.interchange.dataframe_protocol import DataFrame

from recommender import Payload


class Data:
    product_data: pd.DataFrame = pd.DataFrame()
    user_data: pd.DataFrame = pd.DataFrame()
    transaction_data: pd.DataFrame = pd.DataFrame()
    train_data: pd.DataFrame = DataFrame()


class ReadData(Data):
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

        if  'transaction_data' not in self.dataset_file_names.keys():
            raise RuntimeError(
                """
                transaction_datas are missing. 
                In order to run framework transaction_datas dataset is needed.
                please add to dataset_file_names to transaction_data.
                You can only add 4 different transaction_data datasets. 
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
    def reader(self) -> pd.read_csv | pd.read_parquet:
        """only supports .csv and .parquet data format
        :return: pandas csv or parquet reader methods
        """
        if self.data_format == 'csv':
            return pd.read_csv
        if self.data_format == 'parquet':
            return pd.read_parquet


class BasePreProcess:
    product_transaction_cnt: pd.DataFrame | dict[str, float] = {}
    product_user_cnt: pd.DataFrame | dict[str, float] = {}
    user_transaction_cnt: pd.DataFrame | dict[str, float] = {}
    user_product_cnt: pd.DataFrame | dict[str, float] = {}
    null_values: [dict | float] = {}
    mapping: list[
        tuple[str, product_transaction_cnt | product_user_cnt | user_transaction_cnt | user_product_cnt]
    ] = []

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



class PreProcess(ReadData, BasePreProcess):
    def __init__(
            self,
            data_path: str | Path,
            dataset_file_names: dict[str, str],
            data_format: str,
            fields: dict[str, str],
            categorical_features: list[str]
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
        self.categorical_features = categorical_features

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
                self.user_field_name not in list(self.transaction_data.columns)
                or self.product_field_name not in list(self.transaction_data.columns)
                or self.transaction_field_name not in list(self.transaction_data.columns)
        ):
            raise RuntimeError(
                f"""
                user ID or product ID columns are not transaction_data.
                """
            )

        if len(
            set(self.categorical_features)
            - set(self.transaction_data.columns + self.product_data.columns + self.user_data.columns)
        ) != 0:
            raise RuntimeError(
                f"""
                some categorical feature column names are missing one of datasets; here are missing columns;
                    {(
                        set(self.categorical_features) 
                        - set(self.transaction_data.columns + self.product_data.columns + self.user_data.columns)
                    )}   
                """
            )

        self.get_selection_ordered()
        self.update_categorical_features()
        self.get_product_transaction_cnt()
        self.get_product_user_cnt()
        self.get_user_transaction_cnt()
        self.user_product_cnt()
        if fields.get('relevance') is None:
            self.create_rankings()

        self.mapping = [
            (self.product_field_name, self.product_transaction_cnt),
            (self.product_field_name, self.product_user_cnt),
            (self.user_field_name, self.user_transaction_cnt),
            (self.user_field_name, self.user_product_cnt)
        ]

    def update_categorical_features(self):
        for cat in self.categorical_features:
            if cat in self.transaction_data.columns:
                self.train_data[cat] = self.train_data[cat].apply(self.convert_to_str)
            if cat in self.transaction_data.columns:
                self.transaction_data[cat] = self.transaction_data[cat].apply(self.convert_to_str)
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
        self.product_transaction_cnt = (
            self.transaction_data.groupby(self.product_field_name)
            .agg({self.transaction_field_name: "count"})
            .reset_index()
            .rename(columns={self.transaction_field_name: "p_trans_cnt"})
            .sort_values('p_trans_cnt', ascending=False)
        )

        self.product_transaction_cnt['p_trans_cnt_norm'] = self.min_max_norm(
            self.product_transaction_cnt['p_trans_cnt']
        )

        self.product_transaction_cnt = (
            self.product_transaction_cnt.set_index(self.product_field_name)
            .to_dict()
        )

    def get_product_user_cnt(self):
        """
        """
        self.product_user_cnt = (
            self.transaction_data
            .groupby(self.product_field_name)
            .agg({self.user_field_name: pd.Series.nunique})
            .reset_index()
            .rename(columns={self.user_field_name: "p_user_cnt"})
            .sort_values('p_user_cnt', ascending=False)
        )

        self.product_user_cnt['p_user_cnt_norm'] = self.min_max_norm(
            self.product_user_cnt['p_user_cnt']
        )

        self.product_user_cnt = (
            self.product_user_cnt.set_index(self.product_field_name)
            .to_dict()
        )

    def get_user_transaction_cnt(self):
        self.user_transaction_cnt = (
            Data.transaction_data
            .groupby(self.user_field_name)

            .agg({self.transaction_field_name: pd.Series.nunique})
            .reset_index()
            .rename(columns={self.transaction_field_name: "u_transaction_cnt"})
        )

        self.user_transaction_cnt['u_transaction_cnt_norm'] = self.min_max_norm(
            self.product_transaction_cnt['u_transaction_cnt']
        )

        self.user_transaction_cnt = (
            self.user_transaction_cnt.set_index(self.user_field_name)
            .to_dict()
        )

    def user_product_cnt(self):
        self.user_product_cnt = (
            self.transaction_data
            .groupby(self.user_field_name)
            .agg({self.product_field_name: pd.Series.nunique})
            .reset_index()
            .rename(columns={self.product_field_name: "u_product_cnt"})
        )

        self.user_product_cnt['u_product_cnt_norm'] = self.min_max_norm(
            self.user_product_cnt['u_product_cnt']
        )

        self.user_product_cnt = (
            self.user_product_cnt.set_index(self.user_field_name)
            .to_dict()
        )

    def create_rankings(self):
        self.train_data['relevance_scores'] = (
            (.2 * self.train_data['selection_ordered'])
            + (.4 * self.train_data['p_order_cnt_norm'])
            + (.4 * self.train_data['p_user_cnt_norm'])
        )
        self.train_data['rating'] = self.transaction_data.relevance_scores.apply(
            self.get_ranking
        )


class FeatureEng(PreProcess):
    def __init__(
            self,
            data_path: str | Path,
            dataset_file_names: dict[str, str], data_format: str,
            fields: dict[str, str],
            categorical_features: list[str],
            numeric_features: list[str]
    ):
        super().__init__(
            data_path,
            dataset_file_names,
            data_format,
            fields,
            categorical_features
        )
        self.numeric_features = numeric_features
        self.create_offline_features()
        self.get_null_values()

    def update_train_data(self, features):
        for f in features:
            if f not in self.train_data.columns:
                _merge_data = self.transaction_data
                _merge_column = self.transaction_field_name
                if f in self.product_data.columns:
                    _merge_data = self.product_data
                    _merge_column = self.product_field_name
                if f in self.user_data.columns:
                    _merge_data = self.user_data
                    _merge_column = self.user_field_name
                self.train_data = self.train_data.merge(
                    _merge_data[[_merge_column, f]],
                    on=_merge_column,
                    how='left'
                )

    def get_generic_features(self):
        for key, mapping in self.mapping:
            for f in mapping.keys():
                self.train_data[f] = self.train_data[key].apply(
                    mapping[f]
                )

    def create_offline_features(self):
        self.update_train_data(self.categorical_features)
        self.update_train_data(self.numeric_features)

        # this is the features that are created by the platform automatically
        self.get_generic_features()

    def create_online_features(self, payload: Payload):
        pass

    def get_null_values(self):
        for key, mapping in self.mapping:
            for f in mapping.keys():
                self.null_values[f] = mode(self.train_data[self.train_data[f] == self.train_data[f]][f])
