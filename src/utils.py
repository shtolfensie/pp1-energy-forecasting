# %%
from pathlib import Path
import pickle
from construct_datasets import DataSet



# %%


def load_dataset(path: Path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    return d
