# %%
from sqlite3.dbapi2 import Connection
import pandas as pd
import numpy as np
import os
import requests
import datetime
from pathlib import Path
import sqlite3
import pickle
import time


# %%
NIST1_DATA_PATH = Path(__file__).parent.parent / "data/nist/year1/All-Subsystems-hour.csv"
NIST1_DATASETS_PATH = Path(__file__).parent.parent / "data/nist/datasets/year1"
NIST1_DATASETS_PATH.mkdir(exist_ok=True, parents=True)

NIST2_DATA_PATH = Path(__file__).parent.parent / "data/nist/year2/All-Subsystems-hour.csv"
NIST2_DATASETS_PATH = Path(__file__).parent.parent / "data/nist/datasets/year2"
NIST2_DATASETS_PATH.mkdir(exist_ok=True, parents=True)

WATHER_DATABASE_PATH = Path(__file__).parent.parent / "data/weather/weather_history.db"

FR_HOUSE_DATA_PATH = Path(__file__).parent.parent / "data/frhouse/household_power_consumption.txt"
FR_HOUSE_DATASETS_PATH = Path(__file__).parent.parent / "data/frhouse/datasets"
FR_HOUSE_DATASETS_PATH.mkdir(exist_ok=True, parents=True)

# %%
class GpsPos():
    def __init__(self, lat: float, lon: float) -> None:
        self.lat = lat
        self.lon = lon

    def __repr__(self) -> str:
        return f"{self.lat} {self.lon}"

    @staticmethod
    def deg_to_dec(p: str) -> tuple[int, int, int]:
        deg, rest = p.split("°")
        min, sec = rest.split("'")
        return (int(deg), int(min), int(sec))


    @staticmethod
    def from_deg(lat: str, lon: str):
        lat_parts = GpsPos.deg_to_dec(lat)
        lon_parts = GpsPos.deg_to_dec(lon)
        lat_decimal = lat_parts[0] + (lat_parts[1] / 60) + (lat_parts[2] / 3600)
        lon_decimal = lon_parts[0] + (lon_parts[1] / 60) + (lon_parts[2] / 3600)
        return GpsPos(lat=lat_decimal, lon=lon_decimal)



# 39°08'17. 77°13'10., West Dr, Gaithersburg, MD 20899, United States
# 39.138235, -77.219694
NIST_POS = GpsPos(lat=39.138235, lon=-77.219694)
FR_HOUSE_POS = GpsPos(lat=48.7783, lon=2.292)



def generate_hourly_timestamps(start_timestamp, end_timestamp):
    """

    >>> start_timestamp = 1617235200
    >>> end_timestamp = 1617321600
    >>> hourly_timestamps = generate_hourly_timestamps(start_timestamp, end_timestamp)
    """
    timestamps = []
    
    start_datetime = datetime.datetime.utcfromtimestamp(start_timestamp)
    end_datetime = datetime.datetime.utcfromtimestamp(end_timestamp)
    
    current_datetime = start_datetime
    while current_datetime < end_datetime:
        timestamps.append(int(current_datetime.timestamp()))
        current_datetime += datetime.timedelta(hours=1)
    
    return timestamps


# ========
# OpenWeather API response objects

class WeatherConditions():
    def __init__(self, obj) -> None:
        self.id: int = obj["id"]  # weather condition code
        self.main: str = obj["main"]
        self.description: str = obj["description"]
        self.icon: str = obj["icon"]


class WeatherHistoryData():
    def __init__(self, obj) -> None:
        self.dt: int = obj["dt"]
        self.sunrise: int = obj["sunrise"]
        self.sunset: int = obj["sunset"]
        self.temp: float = obj["temp"]
        self.feels_like: float = obj["feels_like"]
        self.pressure: int = obj["pressure"]
        self.humidity: int = obj["humidity"]
        self.dew_point: float = obj["dew_point"]

        self.uvi: float|None = obj.get("uvi")
        self.clouds: int = obj["clouds"]
        self.visibility: int = obj.get("visibility", "")
        self.wind_speed: float = obj["wind_speed"]
        self.wind_gust: float|None = obj.get("wind_gust")
        self.wind_deg: int = obj["wind_deg"]

        self.weather: WeatherConditions = WeatherConditions(obj["weather"][0]) 

        self.rain: float|None = obj["rain"]["1h"] if "rain" in obj.keys() and obj["rain"] is not None in obj else None
        self.snow: float|None = obj["snow"]["1h"] if "snow" in obj.keys() and obj["snow"] is not None in obj else None

    def __repr__(self) -> str:
        return f"{datetime.datetime.fromtimestamp(self.dt)}: {self.temp}°C, {self.pressure}hPa, {self.humidity}%, {self.wind_speed}m/s, {self.rain}mm, {self.wind_gust}"

class WeatherHistoryRes():# {{{
    def __init__(self, obj) -> None:
        self.lat: float = obj["lat"]
        self.lon: float = obj["lon"]
        self.timezone: str = obj["timezone"]
        self.timezone_offset: int = obj["timezone_offset"]
        self.data: WeatherHistoryData = WeatherHistoryData(obj["data"][0])

    def __repr__(self) -> str:
        return f"[{self.lat} {self.lon}] {repr(self.data)}"

    @classmethod
    def csv_header(cls):
        return ["temp", "pressure", "humidity", "wind_speed", "rain", "snow", "sunrise", "sunset"]


    def to_csv_row(self) -> list[str]:
        return [
            str(self.data.temp),
            str(self.data.pressure),
            str(self.data.humidity),
            str(self.data.wind_speed),
            str(self.data.rain) if self.data.rain is not None else "",
            str(self.data.snow) if self.data.snow is not None else "",
            str(self.data.sunrise),
            str(self.data.sunset),
        ]

    def to_pd_row(self) -> list:
        return [
            self.data.temp,
            self.data.pressure,
            self.data.humidity,
            self.data.wind_speed,
            self.data.rain,
            self.data.snow,
            self.data.sunrise,
            self.data.sunset,
        ]

def get_opm_res(url: str):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return data
        else:
            print("Error:", response.status_code)
            return None
    except requests.exceptions.RequestException as e:
        return None


def create_connection(db_file: Path) -> Connection|None:
    """
    Create sqlite database connection

    :param db_file: Path to sqlite db file
    :return: Connection, raise error if can't open file
    """
    conn: Connection|None = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except sqlite3.Error as e:
        print(e)
    return conn# }}}


def create_table(conn: Connection):
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS weather_history (
        id INTEGER PRIMARY KEY,
        lat REAL NOT NULL,
        lon REAL NOT NULL,
        timezone TEXT NOT NULL,
        timezone_offset INTEGER NOT NULL,
        dt INTEGER NOT NULL,
        sunrise INTEGER NOT NULL,
        sunset INTEGER NOT NULL,
        temp REAL NOT NULL,
        feels_like REAL NOT NULL,
        pressure INTEGER NOT NULL,
        humidity INTEGER NOT NULL,
        dew_point REAL NOT NULL,
        uvi REAL,
        clouds INTEGER NOT NULL,
        visibility INTEGER NOT NULL,
        wind_speed REAL NOT NULL,
        wind_gust REAL,
        wind_deg INTEGER NOT NULL,
        weather_id INTEGER NOT NULL,
        weather_main TEXT NOT NULL,
        weather_description TEXT NOT NULL,
        weather_icon TEXT NOT NULL,
        rain REAL,
        snow REAL,
        UNIQUE(dt, lat, lon) -- ON CONFLICT REPLACE
    );
    """
    try:
        cursor = conn.cursor()
        cursor.execute(create_table_sql)
    except sqlite3.Error as e:
        print(e)

def init_weather_db(db_file: Path = WATHER_DATABASE_PATH):
    conn = create_connection(db_file)
    if conn is None:
        print("Error! Could not connect to db and create the weather table")
    else:
        create_table(conn)

def insert_weather_data(conn: Connection, weather_history_res: WeatherHistoryRes):
    """
    Insert downloaded weather data into DB

    :param conn: DB connection
    :param weather_history_res: Downloaded data
    """
    insert_sql = """
    INSERT INTO weather_history (
        lat, lon, timezone, timezone_offset, dt, sunrise, sunset, temp, feels_like,
        pressure, humidity, dew_point, uvi, clouds, visibility, wind_speed, wind_gust,
        wind_deg, weather_id, weather_main, weather_description, weather_icon, rain, snow
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """
    try:
        cursor = conn.cursor()
        data = (
            weather_history_res.lat, weather_history_res.lon, weather_history_res.timezone,
            weather_history_res.timezone_offset, weather_history_res.data.dt,
            weather_history_res.data.sunrise, weather_history_res.data.sunset,
            weather_history_res.data.temp, weather_history_res.data.feels_like,
            weather_history_res.data.pressure, weather_history_res.data.humidity,
            weather_history_res.data.dew_point, weather_history_res.data.uvi,
            weather_history_res.data.clouds, weather_history_res.data.visibility,
            weather_history_res.data.wind_speed, weather_history_res.data.wind_gust,
            weather_history_res.data.wind_deg, weather_history_res.data.weather.id,
            weather_history_res.data.weather.main, weather_history_res.data.weather.description,
            weather_history_res.data.weather.icon, weather_history_res.data.rain,
            weather_history_res.data.snow
        )
        cursor.execute(insert_sql, data)
        conn.commit()
        return cursor.lastrowid
    except sqlite3.Error as e:
        print(f"{weather_history_res.lat} {weather_history_res.lon} {weather_history_res.data.dt}", e)
        return -1


def retrieve_weather_db(conn: Connection, pos: GpsPos, dt: int):
    """
    Try finding weather information for a given timestamp and position in the DB

    :param conn: DB connection
    :param pos: Position to query
    :param dt: Timestamp to query
    """
    query = """
    SELECT * FROM weather_history
    WHERE lat BETWEEN ? - ? AND ? + ? AND lon BETWEEN ? - ? AND ? + ? AND dt = ?
    """
    cursor = conn.cursor()
    # cursor.execute(query, (lat, lon, dt))

    tolerance = 0.0001
    cursor.execute(query, (pos.lat, tolerance, pos.lat, tolerance, pos.lon, tolerance, pos.lon, tolerance, dt))
    row = cursor.fetchone()
    if row:
        obj = {
            "lat": row[1],
            "lon": row[2],
            "timezone": row[3],
            "timezone_offset": row[4],
            "data": [{
                "dt": row[5],
                "sunrise": row[6],
                "sunset": row[7],
                "temp": row[8],
                "feels_like": row[9],
                "pressure": row[10],
                "humidity": row[11],
                "dew_point": row[12],
                "uvi": row[13],
                "clouds": row[14],
                "visibility": row[15],
                "wind_speed": row[16],
                "wind_gust": row[17],
                "wind_deg": row[18],
                "weather": [{
                    "id": row[19],
                    "main": row[20],
                    "description": row[21],
                    "icon": row[22]
                }],
                "rain": row[23],
                "snow": row[24]
            }]
        }
        return WeatherHistoryRes(obj)
    else:
        return None


def download_weather(pos: GpsPos, timestamps: list[int]) -> list[WeatherHistoryRes]:
    """
    Download weather information for a given position and a list of timestamp

    :param pos: Position to query
    :param timestamps: Timestamp to query
    :return: List of weather results
    """
    result = []
    api_key = os.environ["OPEN_WEATHER_MAP"]
    for ts in timestamps:
        url = f"https://api.openweathermap.org/data/3.0/onecall/timemachine?lat={pos.lat}&lon={pos.lon}&dt={ts}&appid={api_key}&units=metric"

        res = get_opm_res(url)
        if res is None:
            print("something went wrong, skipping...")
            continue

        result.append(WeatherHistoryRes(res))
    return result


class DataSet():
    def __init__(self, data: pd.DataFrame, y: pd.Series, X: pd.DataFrame) -> None:
        self.y = y
        self.X = X
        self.data = data


def store_dataset(ds: DataSet, name: str, ds_path: Path):
    file_path = ds_path / f"{name}-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.pkl"
    with open(file_path, "wb") as f:
        pickle.dump(ds, f)

    return file_path

def load_dataset(ds_path: Path):
    with open(ds_path, "rb") as f:
        return pickle.load(f)



def load_model_house(data_path) -> DataSet:
    """
    Load and process the NIST dataset

    :param data_path: Path to raw NIST data
    :return: NIST dataset object
    """
    data = pd.read_csv(data_path)
    feats_of_interes = [
        "Timestamp",
        "OutEnv_OutdoorAmbTemp",
        "HVAC_HeatPumpIndoorUnitPower",
        "HVAC_HeatPumpOutdoorUnitPower",
        "Elec_PowerClothesWasher",
        "Elec_PowerDishwasher",
        "Elec_PowerDryer1of2",
        "Elec_PowerDryer2of2",
        "Elec_PowerGarbageDisposal",
        "Elec_PowerHRV",
        "Elec_PowerHeatPumpWH1of2",
        "Elec_PowerHeatPumpWH2of2",
        "Elec_PowerLights1stFloorA",
        "Elec_PowerLights1stFloorB",
        "Elec_PowerLights2ndFloor",
        "Elec_PowerLightsAttic",
        "Elec_PowerLightsBasement",
        "Elec_PowerLightsBasementStair",
        "Elec_PowerMakeUpAirDamper",
        "Elec_PowerMicrowave",
        "Elec_PowerPlugs2ndFloor",
        "Elec_PowerPlugsAttic",
        "Elec_PowerPlugsBA1",
        "Elec_PowerPlugsBA2North",
        "Elec_PowerPlugsBA2South",
        "Elec_PowerPlugsBR2",
        "Elec_PowerPlugsBR3",
        "Elec_PowerPlugsBR4",
        "Elec_PowerPlugsBaseAHeliodyneHXs",
        "Elec_PowerPlugsBaseB",
        "Elec_PowerPlugsBaseC",
        "Elec_PowerPlugsDR",
        "Elec_PowerPlugsDRB",
        "Elec_PowerPlugsEntryHall",
        "Elec_PowerPlugsKitPeninsula",
        "Elec_PowerPlugsKitRange",
        "Elec_PowerPlugsKitSink",
        "Elec_PowerPlugsLR",
        "Elec_PowerPlugsMBAEast",
        "Elec_PowerPlugsMBAWest",
        "Elec_PowerPlugsMBR",
        "Elec_PowerRefrigerator",
        "Elec_PowerSpare1",
        "Elec_PowerSpare2",
        "Elec_PowerSumpPump"
    ]

    data = data[feats_of_interes]

    data["watts_total"] = data["Elec_PowerClothesWasher"] + \
                          data["Elec_PowerDishwasher"] + \
                          data["Elec_PowerDryer1of2"] + \
                          data["Elec_PowerDryer2of2"] + \
                          data["Elec_PowerGarbageDisposal"] + \
                          data["Elec_PowerHRV"] + \
                          data["Elec_PowerHeatPumpWH1of2"] + \
                          data["Elec_PowerHeatPumpWH2of2"] + \
                          data["Elec_PowerLights1stFloorA"] + \
                          data["Elec_PowerLights1stFloorB"] + \
                          data["Elec_PowerLights2ndFloor"] + \
                          data["Elec_PowerLightsAttic"] + \
                          data["Elec_PowerLightsBasement"] + \
                          data["Elec_PowerLightsBasementStair"] + \
                          data["Elec_PowerMakeUpAirDamper"] + \
                          data["Elec_PowerMicrowave"] + \
                          data["Elec_PowerPlugs2ndFloor"] + \
                          data["Elec_PowerPlugsAttic"] + \
                          data["Elec_PowerPlugsBA1"] + \
                          data["Elec_PowerPlugsBA2North"] + \
                          data["Elec_PowerPlugsBA2South"] + \
                          data["Elec_PowerPlugsBR2"] + \
                          data["Elec_PowerPlugsBR3"] + \
                          data["Elec_PowerPlugsBR4"] + \
                          data["Elec_PowerPlugsBaseAHeliodyneHXs"] + \
                          data["Elec_PowerPlugsBaseB"] + \
                          data["Elec_PowerPlugsBaseC"] + \
                          data["Elec_PowerPlugsDR"] + \
                          data["Elec_PowerPlugsDRB"] + \
                          data["Elec_PowerPlugsEntryHall"] + \
                          data["Elec_PowerPlugsKitPeninsula"] + \
                          data["Elec_PowerPlugsKitRange"] + \
                          data["Elec_PowerPlugsKitSink"] + \
                          data["Elec_PowerPlugsLR"] + \
                          data["Elec_PowerPlugsMBAEast"] + \
                          data["Elec_PowerPlugsMBAWest"] + \
                          data["Elec_PowerPlugsMBR"] + \
                          data["Elec_PowerRefrigerator"] + \
                          data["Elec_PowerSpare1"] + \
                          data["Elec_PowerSpare2"] + \
                          data["Elec_PowerSumpPump"]



    data.drop([
        "Elec_PowerClothesWasher",
        "Elec_PowerDishwasher",
        "Elec_PowerDryer1of2",
        "Elec_PowerDryer2of2",
        "Elec_PowerGarbageDisposal",
        "Elec_PowerHRV",
        "Elec_PowerHeatPumpWH1of2",
        "Elec_PowerHeatPumpWH2of2",
        "Elec_PowerLights1stFloorA",
        "Elec_PowerLights1stFloorB",
        "Elec_PowerLights2ndFloor",
        "Elec_PowerLightsAttic",
        "Elec_PowerLightsBasement",
        "Elec_PowerLightsBasementStair",
        "Elec_PowerMakeUpAirDamper",
        "Elec_PowerMicrowave",
        "Elec_PowerPlugs2ndFloor",
        "Elec_PowerPlugsAttic",
        "Elec_PowerPlugsBA1",
        "Elec_PowerPlugsBA2North",
        "Elec_PowerPlugsBA2South",
        "Elec_PowerPlugsBR2",
        "Elec_PowerPlugsBR3",
        "Elec_PowerPlugsBR4",
        "Elec_PowerPlugsBaseAHeliodyneHXs",
        "Elec_PowerPlugsBaseB",
        "Elec_PowerPlugsBaseC",
        "Elec_PowerPlugsDR",
        "Elec_PowerPlugsDRB",
        "Elec_PowerPlugsEntryHall",
        "Elec_PowerPlugsKitPeninsula",
        "Elec_PowerPlugsKitRange",
        "Elec_PowerPlugsKitSink",
        "Elec_PowerPlugsLR",
        "Elec_PowerPlugsMBAEast",
        "Elec_PowerPlugsMBAWest",
        "Elec_PowerPlugsMBR",
        "Elec_PowerRefrigerator",
        "Elec_PowerSpare1",
        "Elec_PowerSpare2",
        "Elec_PowerSumpPump"
    ], axis=1, inplace=True)

    data["hvac_power_total"] = data["HVAC_HeatPumpIndoorUnitPower"] + data["HVAC_HeatPumpOutdoorUnitPower"]
    data.drop(["HVAC_HeatPumpIndoorUnitPower", "HVAC_HeatPumpOutdoorUnitPower"], axis=1, inplace=True)

    data["Timestamp"] = pd.to_datetime(data["Timestamp"], utc=True)


    data.set_index("Timestamp", inplace=True)

    data = data.interpolate()

    assert isinstance(data, pd.DataFrame)

    y = data["watts_total"]
    assert isinstance(y, pd.Series)

    X = data.drop("watts_total", axis=1)
    assert isinstance(X, pd.DataFrame)
    return DataSet(y=y, X=X, data=data)


# %%
def load_fr_house() -> DataSet:
    """
    Load and process the IHEPC raw data into a dataset

    :return: IHEPC dataset object
    """
    # Date;Time;Global_active_power;Global_reactive_power;Voltage;Global_intensity;Sub_metering_1;Sub_metering_2;Sub_metering_3
    data = pd.read_csv(FR_HOUSE_DATA_PATH, sep=";")

    feats_of_interes = [
        "Date",
        "Time", 
        "Global_active_power",
        "Sub_metering_3",
    ]

    data = data[feats_of_interes]

    data.replace({'?': None, 'N/A': None}, inplace=True)
    for col in ["Global_active_power", "Sub_metering_3"]:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    data["Timestamp"] = data["Date"] + " " + data["Time"]
    data.drop(["Date", "Time"], axis=1, inplace=True)

    data["Timestamp"] = pd.to_datetime(data["Timestamp"], format="%d/%m/%Y %H:%M:%S", utc=True)

    data.set_index("Timestamp", inplace=True)

    data = data.interpolate()

    data_hourly = data[["Global_active_power", "Sub_metering_3"]].resample("H").mean()

    data_hourly["Global_active_power"] = data_hourly["Global_active_power"] * 1000 # convert kW to W

    assert isinstance(data_hourly, pd.DataFrame)

    print(data_hourly)
    y = data["Global_active_power"]
    assert isinstance(y, pd.Series)

    X = data.drop("Global_active_power", axis=1)
    assert isinstance(X, pd.DataFrame)
    return DataSet(y=y, X=X, data=data_hourly)
                   

# %%
def get_weather(timestamp: int, pos: GpsPos) -> tuple[WeatherHistoryRes, bool]|None:
    """
    Get weather information for a given timestamp and position

    Will use cached data if available, otherwise will download data from OWM.

    :param timestamp: Timestamp to query
    :param pos: Position to query
    :return: A tuple of [WeatherHistoryRes, cached_flag] or None
    """
    conn = create_connection(WATHER_DATABASE_PATH)
    if conn is None:
        print("Error: Could not connect to DB when getting weather")
        return
    
    w = retrieve_weather_db(conn, pos, dt=timestamp)
    if w is not None:
        return (w, True)

    ws = download_weather(pos, [timestamp])
    if len(ws) == 0:
        print(f"Error! Could not download weather: {pos} {timestamp}")
        return
    insert_weather_data(conn, ws[0])  # store result to local DB
    return (ws[0], False)


def add_weather_data(ds: DataSet, pos: GpsPos):
    """
    Add weather data to a dataset

    :param ds: Dataset without weather information
    :param pos: Position to use when downloading weather history data
    """
    init_weather_db()
    new_cols = WeatherHistoryRes.csv_header()

    for col in new_cols:
        ds.data[col] = pd.Series([None] * len(ds.data))

    MAX_W = 8800

    calls = 0
    max_calls_per_day = 999

    for idx, ts in zip(ds.data.index, ds.data.index.astype(np.int64) // 10**9):
        if MAX_W == 0:
            print("Reached max weather iterations -- this should not happen")
            break
        MAX_W -= 1

        if calls >= 100:
            print(f"{8800 - MAX_W}. Sleeping for 6s...")
            time.sleep(4)
            calls = 0

        r = get_weather(ts, pos)
        if r is None:
            print("Got None from get_weather")
        else:
            w, cached = r


            if not cached:
                calls += 1
                max_calls_per_day -= 1
                if max_calls_per_day == 0:
                    print("Reached max weather calls -- stopping")
                    break
            vals = w.to_pd_row()
            ds.data.loc[idx, new_cols] = vals

def prepare_nist(pos: GpsPos, add_weather: bool = False, year : int = 1):
    """
    Prepare NIST dataset from raw data

    :param pos: Postition of the dataset
    :param add_weather: If should add weather data
    :param year: NIST year (1, 2)
    """
    if year == 1:
        ds = load_model_house(NIST1_DATA_PATH)
        ds_path = NIST1_DATASETS_PATH
    elif year == 2:
        ds = load_model_house(NIST2_DATA_PATH)
        ds_path = NIST2_DATASETS_PATH
    else:
        raise Exception("unknown nist year")

    if add_weather:
        add_weather_data(ds, pos)
    return store_dataset(ds, f"nist{year}", ds_path)

def prepare_fr_house(pos: GpsPos, add_weather: bool = False):
    """
    Prepare IHEPC dataset from raw data

    :param pos: Postition of the dataset
    :param add_weather: If should add weather data
    """
    ds = load_fr_house()

    if add_weather:
        add_weather_data(ds, pos)
    return store_dataset(ds, "fr-house", FR_HOUSE_DATASETS_PATH)



def main(name):
    ds_path = None
    if name == "nist1":
        ds_path = prepare_nist(NIST_POS, True, year=1)
    if name == "nist2":
        ds_path = prepare_nist(NIST_POS, False, year=2)
    if name == "fr-house":
        ds_path = prepare_fr_house(FR_HOUSE_POS, False)

    return ds_path


def test_retrieve_weather_from_db():
    conn = create_connection(WATHER_DATABASE_PATH.parent / "test_db.db")
    if conn is not None:
        w = retrieve_weather_db(conn, GpsPos(lat=40.712834, lon=-74.006783), dt=1649361600)
        if w is None:
            print("Error! could not load expected weather")
            return

        print(w)

    else:
        print("Error! Cannot open DB.")



if __name__ == "__main__":
    ds_path = main("nist1")  # change name of dataset to get different data
    if ds_path is not None:
        ds = load_dataset(ds_path)
        print(ds.y, ds.X)
        print(ds.X.head())
        print(ds.X.iloc[0])
        print(len(ds.X))
        print(ds.data.columns)
