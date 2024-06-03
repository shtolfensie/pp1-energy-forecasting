# %%
import keras.api.utils as k_utils
import PIL.Image as img
import io
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# %%
def plot_model(model, save: None|str|Path = None, show=True):
    g_img = k_utils.plot_model(model, show_shapes=True, show_layer_names=True).data
    bytes = io.BytesIO(g_img)

    graph_img = img.open(bytes)

    if show:
        graph_img.show()

    if save is not None:
        graph_img.save(save)

# %%
def plot_year():
    pass

# %%
def power_vs_weather():
    pass


# %%
def plot_prediction_for_day(y, y_hat, day: int = 1, save: None|str|Path = None, show: bool = True):
    # day = 59
    plt.plot(np.arange(24), y_hat[24*day], color="blue", label="y_pred")
    plt.plot(np.arange(24), y[24*day], color="orange", label="y")
    plt.title(f"Den {day}")
    plt.xlabel("hodiny")
    plt.ylabel("Spotřeba el. energie [W]")
    plt.legend()
    if save is not None:
        plt.savefig(save)
    if show:
        plt.show()
