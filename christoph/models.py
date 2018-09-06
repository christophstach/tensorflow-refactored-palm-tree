import os.path

import numpy as np
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping


class AutoEncoder3D:
    def __init__(self):
        self.model_path = './christoph/weights/autoencoder_3d_weights.h5'

        self.input = Input(shape=(75,))
        encoded = Dense(45, activation='relu')(self.input)
        encoded = Dense(15, activation='relu')(encoded)
        encoded = Dense(3, activation='relu', name='encoder')(encoded)

        decoded = Dense(15, activation='relu')(encoded)
        decoded = Dense(45, activation='relu')(decoded)
        decoded = Dense(75)(decoded)

        self.model = Model(self.input, decoded)
        self.model.compile(optimizer='adam', loss='mse', metrics=['accuracy', 'mse'])

        if os.path.isfile(self.model_path):
            self.model.load_weights(self.model_path)

    @property
    def encoder(self) -> Model:
        return Model(inputs=self.input, outputs=self.model.get_layer('encoder').output)

    def fit(self, x: np.ndarray, epochs: int = 100):
        self.model.fit(
            x=x,
            y=x,
            epochs=epochs,
            verbose=2,
            shuffle=True,
            validation_split=0.2,
            callbacks=[
                TensorBoard(log_dir='./christoph/logs/autoencoder-3d'),
                EarlyStopping(patience=10)
            ]
        )
        self.model.save_weights(self.model_path)


class AutoEncoder2D:
    def __init__(self):
        self.model_path = './christoph/weights/autoencoder_2d_weights.h5'

        self.input = Input(shape=(75,))
        encoded = Dense(45, activation='relu')(self.input)
        encoded = Dense(15, activation='relu')(encoded)
        encoded = Dense(2, activation='relu', name='encoder')(encoded)

        decoded = Dense(15, activation='relu')(encoded)
        decoded = Dense(45, activation='relu')(decoded)
        decoded = Dense(75)(decoded)

        self.model = Model(self.input, decoded)
        self.model.compile(optimizer='adam', loss='mse', metrics=['accuracy', 'mae'])

        if os.path.isfile(self.model_path):
            self.model.load_weights(self.model_path)

    @property
    def encoder(self) -> Model:
        return Model(inputs=self.input, outputs=self.model.get_layer('encoder').output)

    def fit(self, x: np.ndarray, epochs: int = 100):
        self.model.fit(
            x=x,
            y=x,
            epochs=epochs,
            verbose=2,
            shuffle=True,
            validation_split=0.2,
            callbacks=[
                TensorBoard(log_dir='./christoph/logs/autoencoder-2d'),
                EarlyStopping(patience=10)
            ]
        )
        self.model.save_weights(self.model_path)


class DnnRegressor:
    def __init__(self):
        self.model_path = './christoph/weights/dnn_regressor_weights.h5'

        self.model = Sequential([
            Dense(79, activation='relu', input_dim=79),
            Dense(79, activation='relu'),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae', 'accuracy'])

        if os.path.isfile(self.model_path):
            self.model.load_weights(self.model_path)

    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int = 1000):
        self.model.fit(
            x=x,
            y=y,
            epochs=epochs,
            verbose=2,
            shuffle=True,
            validation_split=0.33,
            callbacks=[
                TensorBoard(log_dir='./christoph/logs/dnn-regressor'),
                EarlyStopping(patience=10, monitor='val_mean_absolute_error')
            ]
        )
        self.model.save_weights(self.model_path)
