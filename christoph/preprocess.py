import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

pd.options.mode.chained_assignment = None


def get_raw_dataframe(test: bool = False) -> pd.DataFrame:
    if test:
        return pd.read_csv('./data/test.csv')
    else:
        return pd.read_csv('./data/train.csv')


def get_dataframe(test=False) -> pd.DataFrame:
    dataframe = get_raw_dataframe(test)
    dataframe.drop('Id', axis=1, inplace=True)
    dataframe.reset_index(drop=True, inplace=True)

    numerical: pd.DataFrame = dataframe.select_dtypes(exclude=['object'])
    categorical: pd.DataFrame = dataframe.select_dtypes(include=['object'])

    numerical.fillna(-1, inplace=True)
    numerical.reset_index(drop=True, inplace=True)
    categorical.reset_index(drop=True, inplace=True)

    for column in categorical:
        categorical[column] = categorical[column].astype('category')
        categorical[column] = categorical[column].cat.codes

    dataframe = pd.concat([categorical, numerical], sort=False, axis=1)
    dataframe.reset_index(drop=True, inplace=True)

    return dataframe


def get_dataset(test=False, remove_outliers=True, scale=True) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    dataframe = get_dataframe(test)

    if remove_outliers:
        for i in range(1):
            clf = IsolationForest()
            clf.fit(dataframe)
            dataframe = dataframe[clf.predict(dataframe) == 1]

    if test:
        labels = None
        features = dataframe.values
    else:
        labels = dataframe['SalePrice'].values
        features = dataframe.drop(columns=['SalePrice']).values

    if scale:
        features = scaler.fit_transform(features)
    return features, labels, scaler
