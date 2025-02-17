import requests
import datetime
import os
import json
import time
import pandas as pd
from typing import TypedDict, List

STATION_LIST_URL = "https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly/doc/ghcnh-station-list.txt"
PSV_BASE_URL = "https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly/access/by-year/{year}/psv/GHCNh_{station}_{year}.psv"

class Station(TypedDict):
    id: int
    lat: float
    lon: float
    alt: float
    name: str

def parse_station_list(station_list_file = 'stations.json', _repeat_takes=0) -> List[Station]:
    """
    Downloads and parses the station list, returning a list of dictionaries with station data.
    """

    # Code for determining to make a request for a new file or not
    get_new_file = True

    if os.path.exists(station_list_file):
        try:
            mod_time = os.path.getmtime(station_list_file)
            mod_datetime = datetime.datetime.fromtimestamp(mod_time)

            dt_now = datetime.datetime.now()
            if mod_datetime.month == dt_now.month and mod_datetime.year == dt_now.year:
                get_new_file = False
        except:
            pass


    if get_new_file:
        response = requests.get(STATION_LIST_URL)
        response.raise_for_status()

        stations: List[Station] = []
        lines = response.text.splitlines()

        for line in lines:
            if len(line) < 43:  # Skip malformed lines
                continue

            station_id = line[:11].strip()
            lat = float(line[12:20].strip())
            lon = float(line[21:30].strip())
            alt = float(line[31:37].strip())
            name = line[38:].strip()

            stations.append({"id": station_id, "lat": lat, "lon": lon, "alt": alt, "name": name})
        
        with open(station_list_file, "w") as f:
            json.dump(stations, f)
            print("Saved and cached stations list.")
    else:
        try:
            with open(station_list_file, "r") as f:
                stations = json.load(f)
        except: # Remove and try to retreive data
            if _repeat_takes < 5:
                seconds_to_wait = _repeat_takes + 0.5
                print(f"Error retreiving file data. Will try again in {seconds_to_wait} seconds.")
                if os.path.exists(station_list_file):
                    os.remove(station_list_file)
                    time.sleep(seconds_to_wait)
                    return parse_station_list(station_list_file, _repeat_takes + 1)
            else:
                raise("Too many takes exhausted to retreive forecast data.")

    return stations

def get_stations_between_bbox(lat_min, lat_max, lon_min, lon_max) -> List[Station]:
    """
    Returns a list of stations that are within a bounding box
    """
    stations = parse_station_list()
    selected_stations = [s for s in stations if lat_min <= s["lat"] <= lat_max and lon_min <= s["lon"] <= lon_max]
    return selected_stations

def get_file_not_found_list(file_name = 'file_not_found_list.json') -> List[str]:
    # Code for determining to make a request for a new file or not
    get_new_file = True

    if os.path.exists(file_name):
        try:
            mod_time = os.path.getmtime(file_name)
            mod_datetime = datetime.datetime.fromtimestamp(mod_time)

            dt_now = datetime.datetime.now()
            if mod_datetime.year == dt_now.year and mod_datetime.month == mod_datetime.month:
                get_new_file = False
        except:
            return []


    if os.path.exists(file_name):
        if get_new_file:
            os.remove(file_name)
            return []
        
        try:
            with open(file_name, "r") as f:
                return json.load(f)
        except:
            return []
    return []

def add_to_file_not_found_list(file_url, file_name = 'file_not_found_list.json'):

    file_not_found_list = get_file_not_found_list(file_name)

    if file_url not in file_not_found_list:
        file_not_found_list.append(file_url)

    print(f"Added {file_url} to the file-not-found list.")

    with open(file_name, "w") as f:
        return json.dump(file_not_found_list, f)
    


def download_psv_files(year:int, lat_min:float, lat_max:float, lon_min:float, lon_max:float, save_dir="forecasts", validate=True, month:int|None=None, day:int|None=None, force_download=False) -> List[str]:
    """
    Downloads all PSV files for stations within the given latitude/longitude bounding box.
    """

    # Filter stations within the bounding box
    selected_stations = get_stations_between_bbox(lat_min, lat_max, lon_min, lon_max)
    file_not_found_list = get_file_not_found_list()
    print(f"Estimated valid stations found: {len(selected_stations)}")

    if not selected_stations:
        print("No stations found within the given bounding box.")
        return

    # Ensure directory exists
    os.makedirs(save_dir, exist_ok=True)

    psv_files = []

    now = datetime.datetime.now()
    len_stations = len(selected_stations)
    for i, station in enumerate(selected_stations):
        pct = 100*(i+1)/len_stations
        print(f"{pct:.2f}%")

        station_id = station["id"]
        url = PSV_BASE_URL.format(year=year, station=station_id)
        file_path = os.path.join(save_dir, f"GHCNh_{station_id}_{year}.psv")

        get_new_file = True
        if os.path.exists(file_path):
            try:
                mod_time = os.path.getmtime(file_path)
                mod_datetime = datetime.datetime.fromtimestamp(mod_time)

                same_year_older_month = month and month < mod_datetime.month and mod_datetime.year == year
                same_year_older_day = day and month and same_year_older_month and day < (mod_datetime.day-2) # At least 2-3 days of buffer to allow forecast stations to update their data
                same_day = mod_datetime.date() == now.date()
                older_year = year < mod_datetime.year

                if (older_year or same_day or same_year_older_month or same_year_older_day) and not force_download:
                    print("Using cache")
                    get_new_file = False
            except:
                pass
        
        if not get_new_file:
            if validate:
                print(f"Validating {file_path}")
                try:
                    df = pd.read_csv(file_path, delimiter="|", low_memory=False)
                                            

                    print(f"using cache: {file_path}")
                    psv_files.append(file_path)
                except:
                    print(f"Corruption error trying to validate the data for {file_path}")
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    get_new_file = True
            else:
                print(f"using cache: {file_path}. WARNING: VALIDATION DISABLED")
                psv_files.append(file_path)

        if get_new_file:
            if url in file_not_found_list:
                print(f"Skipped {url} due to previous loading issues.")
                continue

            try:
                print(f"Contacting {url}")
                response = requests.get(url, timeout=20)
                if response.status_code == 200:
                    print("Response received. Downloading...")
                    with open(file_path, "wb") as f:
                        f.write(response.content)
                    print(f"Downloaded: {file_path}")
                    psv_files.append(file_path)
                else:
                    print(f"File not found: {url}\nStatus Code: {response.status_code}")
                    add_to_file_not_found_list(url)

            except requests.RequestException as e:
                print(f"Error downloading {url}: {e}")
                add_to_file_not_found_list(url)
    
    return psv_files


if __name__ == "__main__":
    # Example usage:
    # Download PSV files for 2023 within a bounding box (latitude 17 to 19, longitude -64 to -60)
    lon_min, lat_min, lon_max, lat_max = -106.64719063660635, 25.840437651866516, -93.5175532104321, 36.50050935248352

    psv_file_list = download_psv_files(2023, lat_min, lat_max, lon_min, lon_max)
