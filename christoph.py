from typing import Dict

from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Sequential


def get_train_split(ratio: int) -> Dict[int, int, int, int]:
    pass


def train():
    model = Sequential([
        Dense(32, input_shape=(784,)),
        Activation('relu'),
        Dense(10),
        Activation('softmax'),
    ])
