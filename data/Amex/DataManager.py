import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import OneHotEncoder
from pathlib import Path
from interfaces import IData


class DataManager(IData):
    def __init__(self, cfg):
        super(DataManager, self).__init__(cfg)

        self.root = Path(__file__).parent.resolve()
        self.dataset_path = self.root / 'dataset'
        self.cleaning_cfg_path = self.root / 'cleaning.json'
        self.cleaning_cfg = json.load(open(self.cleaning_cfg_path))

        self.X_train_path = self.dataset_path / "output" / "X_res_RUS.parquet"
        self.Y_train_path = self.dataset_path / "output" / "y_res_RUS.csv"
        self.X_test_path = self.dataset_path / "output" / "X_test.parquet"
        self.Y_test_path = self.dataset_path / "output" / "y_test.csv"

        self.convert_dict = self.cleaning_cfg["convert_dict"]
        self.features_avg = self.cleaning_cfg["features_avg"]
        self.features_min = self.cleaning_cfg["features_min"]
        self.features_max = self.cleaning_cfg["features_max"]
        self.features_var = self.cleaning_cfg["features_var"]
        self.features_last = self.cleaning_cfg["features_last"]
        self.full_list = self.cleaning_cfg["full_list"]
        self.groupby_dict = self.cleaning_cfg["groupby_dict"]

    def load(self):
        print("Loading dataset...")
        X_train = pd.read_parquet(self.X_train_path)
        Y_train = pd.read_csv(self.Y_train_path)
        X_test = pd.read_parquet(self.X_test_path)
        Y_test = pd.read_csv(self.Y_test_path)

        X_train = X_train.drop(['customer_ID'], axis=1)
        Y_train = np.ravel(Y_train)
        X_test = X_test.drop(['customer_ID'], axis=1)
        Y_test = Y_test.to_numpy()

        # return pd.read_parquet(self.X_train_path), pd.read_csv(self.Y_train_path), pd.read_parquet(self.X_test_path), pd.read_csv(self.Y_test_path)
        return X_train, Y_train, X_test, Y_test

    def clean(self):
        print("Cleaning dataset...")

        self.group()
        self.reduce_train()
        self.reduce_test()

    def split(self):
        print("Train test split...")

        if 'X_res_RUS.csv' not in os.listdir(self.dataset_path / "output"):

            data = pd.read_parquet(self.dataset_path /
                                   "output" / "train_data_reduced.parquet")

            X_train, X_test, Y_train, Y_test = train_test_split(
                data.drop('target', axis=1), data['target'], test_size=0.2, random_state=1)

            if self.cfg["sampling"] == 'RUS':
                X_train, Y_train = self._undersample(
                    X_train, Y_train, RUS_method=self.cfg["sampling"])

            X_train.to_parquet(self.X_train_path)
            Y_train.to_csv(self.Y_train_path, index=False)

            X_test.to_parquet(self.X_test_path)
            Y_test.to_csv(self.Y_test_path, index=False)

            del X_test, Y_test

    ##############
    ####utils#####
    ##############
    def group(self):
        for i in self.full_list:
            temp_list = []

            if len(i.split('_')) == 3:
                x = i[:-3]
                if x in set(self.features_avg + self.features_min + self.features_max + self.features_last):
                    if x in self.features_avg:
                        temp_list.append('mean')
                    if x in self.features_min:
                        temp_list.append('min')
                    if x in self.features_max:
                        temp_list.append('max')
                    if x in self.features_last:
                        temp_list.append('last')
                    if x in self.features_var:
                        temp_list.append('var')

                    self.groupby_dict[i] = temp_list

            elif i not in set(self.features_avg + self.features_min + self.features_max + self.features_last):
                self.groupby_dict[i] = 'last'

            else:
                if i in self.features_avg:
                    temp_list.append('mean')
                if i in self.features_min:
                    temp_list.append('min')
                if i in self.features_max:
                    temp_list.append('max')
                if i in self.features_last:
                    temp_list.append('last')
                if i in self.features_var:
                    temp_list.append('var')

                self.groupby_dict[i] = temp_list

    def reduce_train(self):
        if 'train_data_reduced.csv' not in os.listdir(self.dataset_path / "output"):
            print('Reading training data and training labels....')

            raw_X = pd.read_parquet(
                self.dataset_path / "raw" / "train.parquet")
            raw_Y = pd.read_csv(
                self.dataset_path / "raw" / "train_labels.csv")

            data = raw_Y.merge(raw_X, how='inner', on='customer_ID')

            del raw_X
            del raw_Y

            # TODO: Encode categorical variables (Discussion suggests there are more categorical/binary variables then below list)

            print('Changing categorical columns to integer and fill NAs')
            cat_col = ['B_30', 'B_38', 'D_114', 'D_116', 'D_117',
                       'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']

            for col in cat_col:
                print(f'Processing column {col}')
                if data[col].dtype == 'float64':
                    data[col] = data[col].fillna(-1)
                    data[col] = data[col].astype('int8')
                    data[col] = data[col].astype('category')
                else:
                    data[col] = data[col].fillna('na')
                    data[col] = data[col].astype('category')

            print('One-hot encoding')
            enc = OneHotEncoder(handle_unknown='ignore')

            encoded = enc.fit_transform(data[cat_col])

            encoded_df = pd.DataFrame(
                encoded.toarray(), columns=enc.get_feature_names(cat_col)).astype('int8')

            data = data.join(encoded_df)

            data = data.drop(cat_col, axis=1)

            data['S_2'] = pd.to_datetime(data['S_2'])

            data = data.sort_values('S_2').drop('S_2', axis=1)

            data = data.groupby('customer_ID').agg(self.groupby_dict)

            # print(data)

            data.columns = data.columns.get_level_values(
                0) + '_' + data.columns.get_level_values(1)

            data = data.rename(columns={'target_last': 'target'}).reset_index()

            data.to_parquet(self.dataset_path / "output" /
                            "train_data_reduced.parquet")

            del data

    def reduce_test(self):

        if 'test_data_reduced.csv' not in os.listdir(self.dataset_path):
            print('Reading test data...')

            x_to_submit = pd.read_parquet(
                self.dataset_path / "raw" / "test.parquet")
            #################################################################
            print('Changing categorical columns to integer and fill NAs')
            cat_col = ['B_30', 'B_38', 'D_114', 'D_116', 'D_117',
                       'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']

            for col in cat_col:
                print(f'Processing column {col}')
                if x_to_submit[col].dtype == 'float64':
                    x_to_submit[col] = x_to_submit[col].fillna(-1)
                    x_to_submit[col] = x_to_submit[col].astype('int8')
                    x_to_submit[col] = x_to_submit[col].astype('category')
                else:
                    x_to_submit[col] = x_to_submit[col].fillna('na')
                    x_to_submit[col] = x_to_submit[col].astype('category')

            print('One-hot encoding')
            enc = OneHotEncoder(handle_unknown='ignore')

            encoded = enc.fit_transform(x_to_submit[cat_col])

            encoded_df = pd.DataFrame(encoded.toarray(),
                                      columns=enc.get_feature_names(cat_col)
                                      ).astype('int8')

            x_to_submit = x_to_submit.join(encoded_df)

            x_to_submit = x_to_submit.drop(cat_col, axis=1)

            x_to_submit['S_2'] = pd.to_datetime(x_to_submit['S_2'])

            x_to_submit['D_64_1'] = 0
            x_to_submit['D_66_0'] = 0
            x_to_submit['D_68_0'] = 0

            x_to_submit = x_to_submit.sort_values('S_2').drop('S_2', axis=1)

            self.groupby_dict.pop('target')

            x_to_submit = x_to_submit.groupby(
                'customer_ID').agg(self.groupby_dict)

            x_to_submit.columns = x_to_submit.columns.get_level_values(
                0) + '_' + x_to_submit.columns.get_level_values(1)

            x_to_submit = x_to_submit.rename(
                columns={'target_last': 'target'}).reset_index()

            print('Saving reduced test data for submission...')
            x_to_submit.to_parquet(self.dataset_path /
                                   "output" / "test_data_reduced.parquet")

            del x_to_submit

    def _undersample(self,
                     X_train,
                     Y_train,
                     RUS_method
                     ):

        print(f'Undersampling, sampling method is {RUS_method}')

        rus = RandomUnderSampler(random_state=42)

        X_res, Y_res = rus.fit_resample(X_train, Y_train)

        return X_res, Y_res
