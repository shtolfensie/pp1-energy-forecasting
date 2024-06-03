import keras
import keras.api.layers as layers

def tcn_lstm(num_features: int = 1):
    model = keras.Sequential()
    model.add(layers.Conv1D(filters=64, kernel_size=3, activation="relu", input_shape=(48, num_features), padding="causal"))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Conv1D(filters=32, kernel_size=2, activation="relu", padding="causal"))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Flatten())
    model.add(layers.RepeatVector(24))
    model.add(layers.LSTM(200, activation="relu", return_sequences=True))
    model.add(layers.TimeDistributed(layers.Dense(50, activation="relu")))
    model.add(layers.TimeDistributed(layers.Dense(1)))
    model.compile(optimizer="adam", loss="mse")

    return model

def create_tcn_lstm(num_features: int = 1):
    """
    Create an encoder-decoder TCN-LSTM model

    Multivariate inputs are processed by the same convolutional layer.

    :param num_features: Number of features, set to 1 for univariate operation
    """
    return tcn_lstm(num_features)
