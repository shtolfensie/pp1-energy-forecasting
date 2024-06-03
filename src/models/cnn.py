# %%
import keras
import keras.api.layers as layers
from construct_datasets import DataSet



def tcn_single_branch(num_layers: int = 1):
    """
    Univariate TCN network.

    The input shape is `(samples, lookback_window, 1)`.

    :param num_layers: Number of layers
    """
    model = keras.Sequential()
    model.add(layers.Conv1D(filters=64, kernel_size=2, activation="relu", input_shape=(48, 1), padding="causal"))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Conv1D(filters=32, kernel_size=2, activation="relu", input_shape=(48, 1), padding="causal"))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(50, activation="relu"))
    model.add(layers.Dense(24))
    model.compile(optimizer="adam", loss="mse")

    return model


# %%
def tcn_single_branch_multi(num_features: int = 1):
    """
    TCN network with a single branch and a variable number of input sequences.

    The input shape is `(samples, lookback_window, num_features)`.

    :param num_features: Number of input sequences.
    """
    model = keras.Sequential()
    model.add(layers.Conv1D(filters=64, kernel_size=2, activation="relu", input_shape=(48, num_features), padding="causal"))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Conv1D(filters=32, kernel_size=2, activation="relu", padding="causal"))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(50, activation="relu"))
    model.add(layers.Dense(24))
    model.compile(optimizer="adam", loss="mse")

    return model


def tcn_2var_branch_res(num_layers: int = 1):
    """
    Multivariate TCN network with 2 input variables.

    Each input variable is first processed by a separte branch, with the results concatenated and passed into
    the common part of the network

    :param num_layers: Number of layers in the common part of the network
    """

    inp1 = layers.Input(shape=(48, 1))
    branch1 = layers.Conv1D(filters=64, kernel_size=2, activation="relu", padding="causal", dilation_rate=4)(inp1)

    skip_b1 = layers.Conv1D(filters=1, kernel_size=1, padding="causal")(inp1)
    branch1 = layers.Add()([branch1, skip_b1])


    inp2 = layers.Input(shape=(48, 1))
    branch2 = layers.Conv1D(filters=64, kernel_size=2, activation="relu", padding="causal", dilation_rate=4)(inp2)

    skip_b2 = layers.Conv1D(filters=1, kernel_size=1, padding="causal")(inp2)
    branch2 = layers.Add()([branch2, skip_b2])

    merge = layers.concatenate([branch1, branch2])

    stem = layers.Conv1D(filters=32, kernel_size=2, activation="relu", padding="causal")(merge)
    stem = layers.MaxPooling1D(pool_size=2)(stem)
    stem = layers.Conv1D(filters=16, kernel_size=2, activation="relu", padding="causal")(merge)
    stem = layers.MaxPooling1D(pool_size=2)(stem)
    stem = layers.Flatten()(stem)

    output = layers.Dense(50, activation="relu")(stem)
    output = layers.Dense(24)(output)


    model = keras.Model(inputs=[inp1, inp2], outputs=output)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")

    return model


def tcn_4var_branch_res(num_layers: int = 1):
    """
    Multivariate TCN network with 4 input variables.

    Each input variable is first processed by a separte branch, with the results concatenated and passed into
    the common part of the network

    :param num_layers: Number of layers in the common part of the network
    """

    inp1 = layers.Input(shape=(48, 1))
    branch1 = layers.Conv1D(filters=64, kernel_size=2, activation="relu", padding="causal", dilation_rate=4)(inp1)

    skip_b1 = layers.Conv1D(filters=1, kernel_size=1, padding="causal")(inp1)
    branch1 = layers.Add()([branch1, skip_b1])


    inp2 = layers.Input(shape=(48, 1))
    branch2 = layers.Conv1D(filters=64, kernel_size=2, activation="relu", padding="causal", dilation_rate=4)(inp2)

    skip_b2 = layers.Conv1D(filters=1, kernel_size=1, padding="causal")(inp2)
    branch2 = layers.Add()([branch2, skip_b2])

    inp3 = layers.Input(shape=(48, 1))
    branch3 = layers.Conv1D(filters=64, kernel_size=2, activation="relu", padding="causal", dilation_rate=4)(inp3)

    skip_b3 = layers.Conv1D(filters=1, kernel_size=1, padding="causal")(inp3)
    branch3 = layers.Add()([branch3, skip_b3])

    inp4 = layers.Input(shape=(48, 1))
    branch4 = layers.Conv1D(filters=64, kernel_size=2, activation="relu", padding="causal", dilation_rate=4)(inp4)

    skip_b4 = layers.Conv1D(filters=1, kernel_size=1, padding="causal")(inp4)
    branch4 = layers.Add()([branch4, skip_b4])

    merge = layers.concatenate([branch1, branch2, branch3, branch4])

    stem = layers.Conv1D(filters=32, kernel_size=2, activation="relu", padding="causal")(merge)
    stem = layers.MaxPooling1D(pool_size=2)(stem)
    for _ in range(num_layers):
        stem = layers.Conv1D(filters=16, kernel_size=2, activation="relu", padding="causal")(merge)
        stem = layers.MaxPooling1D(pool_size=2)(stem)
        stem = layers.Flatten()(stem)

    output = layers.Dense(50, activation="relu")(stem)
    output = layers.Dense(24)(output)


    model = keras.Model(inputs=[inp1, inp2, inp3, inp4], outputs=output)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")

    return model


def create_tcn(num_variables: int = 1, num_layers: int = 1):
    """
    Generic function to create a TCN based model.

    The model can either be univariate (num_variables=1), or multivariate, with either 2 or 4 input variables.
    Each variable is processed by a separate branch before being merged into the unified part of the network.

    For a univariate model, the input shape is `(samples, lookback_window, 1)`, for a multivariate model, the shape
    is a list `[(samples, lookback_window, 1), ...]` with one input sequence per specified input variable.

    :param num_variables: Number of input variables
    :param num_layers: Number of common layers (does not effect on the separate branches)
    :raises Exception: If a unsupported number of input variables is requested
    """
    if num_variables == 1:
        return tcn_single_branch(num_layers=num_layers)
    elif num_variables == 2:
        return tcn_2var_branch_res(num_layers=num_layers);
    elif num_variables == 4:
        return tcn_4var_branch_res(num_layers=num_layers)
    else:
        raise Exception("Unsupported number of input variables")
