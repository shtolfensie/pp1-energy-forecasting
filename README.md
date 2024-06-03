# Residential home energy load forecasting


This repository contains datasets and models for energy forecasting experiments.
Each model can be evaluated by running the following experiments:

1. univariate energy consumption forecasting
    - this can be evaluated on the test part of the given dataset, or on the test part of the other dataset, to check, if the model is transferable
2. multivariate energy consumption forecasting
    - A combination of the following exogenous variables can be tested: temperature, humidity, wind speed


## Environment variables

Before running any experiment, make sure the appropriate environment variables are available:

| variable name    | value          | purpose |
| -------------    | -----          | ------- |
| `KERAS_BACKEND`    | "torch"        | Sets `keras` backend, required for any training or inference.       |
| `OPEN_WEATHER_MAP` | YOUR_API_KEY   | Downloading weather history data. Required when creating a new dataset. |

## Usage

This project uses two datasets with residential house energy consumption data. To get started, download the
raw data into `data/nist` and `data/frhouse` respectively:

- NIST: https://pages.nist.gov/netzero/data.html
- IHEPC (frhouse): https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

To setup an isolated environment, you can run:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```


Then transform the raw data into a dataset with added weather variables:

```bash
python src/construct_datasets.py
```

Experiments can be specified from the CLI by running:

```bash
python src/main.py -f history -f temperature --network tcn -e 1
```

To see all the various options, you can run:

```bash
python src/main.py --help
```


## Weather data


Weather history data is downloaded for each datapoint.

> Contains information from [OpenWeather](https://openweathermap.org/), which is made available
> here under the Open Database License (ODbL).

![openweathermap logo](./images/OpenWeather-Master-Logo RGB.png)
