# %%
import numpy as np
import preprocess_dataset as dat
import torch
import evaluation
from construct_datasets import DataSet


# %%
def naive_avg_history(X, lookback=48, horizon=24):
    assert(lookback // 2 == horizon)
    first_half = X[:, :horizon]
    second_half = X[:, horizon:]
    averaged = (first_half + second_half) / 2.0
    return averaged


# %%
X, y = dat.nist_univariate()

# %%
X_train, y_train, X, y = dat.fr_univariate()

# %%
y_hat = naive_avg_history(X)[:, :, 0]
 
# %%

rmse, mape = evaluation.eval_forecast(y, y_hat)
print("rmse:", torch.mean(rmse))
print("mape:", torch.mean(mape))


# %%
def eval_forecast(y, y_hat):


    mse = np.mean(np.square(y - y_hat), axis=-1)
    # mse = mean_squared_error(y, y_hat)

    rmse = np.sqrt(mse)

    # mape = mean_absolute_percentage_error(y, y_hat)
    mape = 100 * np.mean(np.abs((y - y_hat) / y), axis=-1)

    nrmse = 100 * (rmse / (np.max(y) - np.min(y)))

    return rmse, mape, nrmse

# %%
mse, mape, nrmse = eval_forecast(y, y_hat)
print(mape)
print("rmse:", np.mean(mse))
print("mape:", np.mean(mape))
print("nrmse:", np.mean(nrmse))
