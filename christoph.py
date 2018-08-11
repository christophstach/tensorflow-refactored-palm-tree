import pandas as pd

from typing import Dict

from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Sequential


def get_train_test():
    df_train: pd.DataFrame = pd.read_csv('data/train.csv')
    df_test: pd.DataFrame = pd.read_csv('data/test.csv')

    df_train = df_train[['LotFrontage', 'LotArea']]


    print(df_train.head(10))
    pass


def train():
    model = Sequential([

        Activation('softmax'),
    ])


get_train_test()
