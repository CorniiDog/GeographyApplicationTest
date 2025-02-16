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
            if mod_datetime.month == dt_now.month and mod_datetime.year == dt_now.year:
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

def add_to_file_not_found_list(file_url, file_name = 'file_not_found_list.json', file_not_found_list=None):

    if not file_not_found_list:
        file_not_found_list = get_file_not_found_list(file_name)

    if file_url not in file_not_found_list:
        file_not_found_list.append(file_url)

    print(f"Added {file_url} to the file-not-found list.")

    with open(file_name, "w") as f:
        return json.dump(file_not_found_list, f)
    


def download_psv_files(year:int, lat_min:float, lat_max:float, lon_min:float, lon_max:float, save_dir="forecasts", validate=True, month:int|None=None) -> List[str]:
    """
    Downloads all PSV files for stations within the given latitude/longitude bounding box.
    """

    # Filter stations within the bounding box
    selected_stations = get_stations_between_bbox(lat_min, lat_max, lon_min, lon_max)
    file_not_found_list = get_file_not_found_list()

    if not selected_stations:
        print("No stations found within the given bounding box.")
        return

    # Ensure directory exists
    os.makedirs(save_dir, exist_ok=True)

    psv_files = []

    now = datetime.datetime.now()

    for station in selected_stations:
        station_id = station["id"]
        url = PSV_BASE_URL.format(year=year, station=station_id)
        file_path = os.path.join(save_dir, f"GHCNh_{station_id}_{year}.psv")

        get_new_file = True
        if os.path.exists(file_path):
            try:
                mod_time = os.path.getmtime(file_path)
                mod_datetime = datetime.datetime.fromtimestamp(mod_time)

                same_year_older_month = month and month < mod_datetime.month and mod_datetime.year == year
                same_day = mod_datetime.date() == now.date()
                older_year = year < mod_datetime.year

                if older_year or same_day or same_year_older_month:
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
                response = requests.get(url)
                if response.status_code == 200:
                    with open(file_path, "wb") as f:
                        f.write(response.content)
                    print(f"Downloaded: {file_path}")
                    psv_files.append(file_path)
                else:
                    print(f"File not found: {url}")
                    add_to_file_not_found_list(url, file_not_found_list=file_not_found_list)

            except requests.RequestException as e:
                print(f"Error downloading {url}: {e}")
                add_to_file_not_found_list(url, file_not_found_list=file_not_found_list)
    
    return psv_files

# Example usage:
# Download PSV files for 2023 within a bounding box (latitude 17 to 19, longitude -64 to -60)
download_psv_files(2023, 17, 19, -64, -60)
