# %%
from pathlib import Path
import numpy as np
import construct_datasets
from construct_datasets import DataSet


DATA_ROOT = Path(__file__).parent.parent / "data"

# %%
# The following function (get_rnn_inputs()) is taken from https://github.com/albertogaspar/dts
def get_rnn_inputs(data, window_size, horizon,
                   multivariate_output=False, shuffle=False, other_horizon=None):
    """
    Prepare data for feeding a RNN model.
    :param data: numpy.array
        shape (n_samples, n_features) or (M, n_samples, n_features)
    :param window_size: int
        Fixed size of the look-back
    :param horizon: int
        Forecasting horizon, the number of future steps that have to be forecasted
    :param multivariate_output: if True, the target array will not have shape
        (n_samples, output_sequence_len) but (n_samples, output_sequence_len, n_features)
    :param shuffle: if True shuffle the data on the first axis
    :param other_horizon:
    :return: tuple
        Return two numpy.arrays: the input and the target for the model.
        the inputs has shape (n_samples, input_sequence_len, n_features)
        the target has shape (n_samples, output_sequence_len)
    """
    if data.ndim == 2:
        data = np.expand_dims(data, 0)
    inputs = []
    targets = []
    for X in data:  # for each array of shape (n_samples, n_features)
        n_used_samples = X.shape[0] - horizon - window_size + 1
        for i in range(n_used_samples):
            inputs.append(X[i: i + window_size])
            # TARGET FEATURE SHOULD BE THE FIRST
            if multivariate_output:
                if other_horizon is None:
                    targets.append(
                        X[i + window_size: i + window_size + horizon])
                else:
                    targets.append(
                        X[i + 1: i + window_size + 1])
            else:
                if other_horizon is None:
                    targets.append(
                        X[i + window_size: i + window_size + horizon, 0])
                else:
                    targets.append(
                        X[i + 1: i + window_size + 1, 0])
    encoder_input_data = np.asarray(inputs)  # (n_samples, sequence_len, n_features)
    decoder_target_data = np.asarray(targets)  # (n_samples, horizon) or (n_samples, horizon, n_features) if multivariate_output
    idxs = np.arange(encoder_input_data.shape[0])
    if shuffle:
        np.random.shuffle(idxs)
    return encoder_input_data[idxs], decoder_target_data[idxs]

def nist_univariate(path: Path|None = None):
    """
    Univariate NIST dataset

    :param path: Path to raw data, if None, uses default name: nist[year].pkl
    """
    X_train, y_train = _nist_univariate(path, year=1)
    X_test, y_test = _nist_univariate(path, year=2)

    return X_train, y_train, X_test, y_test

# %%
def _nist_univariate(path: Path|None = None, year: int = 1):
    if path is None:
        ds = construct_datasets.load_dataset(DATA_ROOT / f"nist/datasets/year{year}/nist{year}.pkl")
    else:
        ds = construct_datasets.load_dataset(path)

    univ_nist = np.array(ds.data["watts_total"]).reshape(len(ds.data["watts_total"]), 1)
    X, y = get_rnn_inputs(univ_nist, 48, 24)

    return X, y

# %%
def fr_univariate(path: Path|None = None):
    """
    Univariate IHEPC dataset

    :param path: Path to raw data, if None, uses default name: nist[year].pkl
    """
    if path is None:
        ds = construct_datasets.load_dataset(DATA_ROOT / f"frhouse/datasets/fr-house.pkl")
    else:
        ds = construct_datasets.load_dataset(path)


    year1 = ds.data.iloc[:8760]
    year2 = ds.data.iloc[8760:8760*2]

    year1_seq = np.array(year1["Global_active_power"]).reshape(len(year1["Global_active_power"]), 1)
    year2_seq = np.array(year2["Global_active_power"]).reshape(len(year2["Global_active_power"]), 1)

    X_train, y_train = get_rnn_inputs(year1_seq, 48, 24)
    X_test, y_test = get_rnn_inputs(year2_seq, 48, 24)

    return X_train, y_train, X_test, y_test

# %%
def nist_multivariate(path: Path|None = None, exo_list: list[str] = ["temperature"]):
    """
    Multivariate NIST dataset

    :param path: Path to raw data, if None, uses default name: nist[year].pkl
    :param exo_list: List of additional variables to include
    """

    if "history" in exo_list:
        exo_list.remove("history")

    X_exo_list_train = []
    X_train, y_train = None, None
    for var in exo_list:
        X_multi, y_in = _nist_multivariate(path, year=1, exo_var=var)

        X1 = X_multi[:, :, 0].reshape((X_multi.shape[0], X_multi.shape[1], 1))
        X_exo = X_multi[:, :, 1].reshape((X_multi.shape[0], X_multi.shape[1], 1))

        X_train = X1
        y_train = y_in

        X_exo_list_train.append(X_exo)

    X_exo_list_test = []
    X_test, y_test = None, None
    for var in exo_list:
        X_multi, y_in = _nist_multivariate(path, year=1, exo_var=var)  # TODO(filip): change

        X1 = X_multi[:, :, 0].reshape((X_multi.shape[0], X_multi.shape[1], 1))
        X_exo = X_multi[:, :, 1].reshape((X_multi.shape[0], X_multi.shape[1], 1))

        X_test = X1
        y_test = y_in

        X_exo_list_test.append(X_exo)
    
    return X_train, X_exo_list_train, y_train, X_test, X_exo_list_test, y_test



# %%
def _nist_multivariate(path: Path|None = None, year: int = 1, exo_var="temperature"):
    if path is None:
        ds = construct_datasets.load_dataset(DATA_ROOT / f"nist/datasets/year{year}/nist{year}.pkl")
    else:
        ds = construct_datasets.load_dataset(path)
    # print(ds.data.columns)

    if exo_var == "temperature":
        # ==== temp
        # print(ds.data["temp"][~ds.data["temp"].apply(lambda x: isinstance(x, float))])
        ds.data["temp"] = ds.data["temp"].astype(float)
        univ_nist = np.hstack((np.array(ds.data["watts_total"]).reshape((len(ds.data["watts_total"]), 1)), np.array(ds.data["temp"]).reshape((len(ds.data["watts_total"]), 1))))

    elif exo_var == "humidity":
        # ==== humidity
        # print(ds.data["humidity"][~ds.data["humidity"].apply(lambda x: isinstance(x, float))])
        ds.data["humidity"] = ds.data["humidity"].astype(float)
        univ_nist = np.hstack((np.array(ds.data["watts_total"]).reshape((len(ds.data["watts_total"]), 1)), np.array(ds.data["humidity"]).reshape((len(ds.data["watts_total"]), 1))))
    elif exo_var == "wind":
        # ==== wind_speed
        # print(ds.data["wind_speed"][~ds.data["wind_speed"].apply(lambda x: isinstance(x, float))])
        ds.data["wind_speed"] = ds.data["wind_speed"].astype(float)
        univ_nist = np.hstack((np.array(ds.data["watts_total"]).reshape((len(ds.data["watts_total"]), 1)), np.array(ds.data["wind_speed"]).reshape((len(ds.data["watts_total"]), 1))))
    else:
        raise Exception("unknown exo variable: " + exo_var)

    X, y = get_rnn_inputs(univ_nist, 48, 24)

    return X, y
