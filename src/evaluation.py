import numpy as np

def eval_forecast(y, y_hat, arch: str = "tcn"):
    """
    Calculate error metrics for the given prediction in y_hat.

    Each metric is returned in a per-hour format. To get a single number per metric, either
    take the average of each metric, or call average_metrics().

    :param y np.ndarray: Correct values
    :param y_hat np.ndarray: Predicted values
    :return: rmse, mape, nrmse metrics for each hour of the day
    """
    if arch != "tcn":
        y_hat = y_hat[:, :, 0]

    mse = np.mean(np.square(y - y_hat), axis=-1)
    rmse = np.sqrt(mse)
    mape = 100 * np.mean(np.abs((y - y_hat) / y), axis=-1)
    nrmse = 100 * (rmse / (np.max(y) - np.min(y)))

    return rmse, mape, nrmse

def average_metrics(metrics):
    """
    Average per-hour metrics into a single number per metric.

    Given an iterable of metrics, where each metric is a list of per-hour values, calculate the average
    of these values. Resulting in a list of same length of metrics, of single values.

    :param metrics: List of per-hour metrics, will average each metric individually
    """
    res = []
    for m in metrics:
        res.append(np.mean(m))

    return res
