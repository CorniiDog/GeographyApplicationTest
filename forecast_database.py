import sqlite3
import forecast_downloader
import datetime
import os
import pandas as pd

DB_PATH = "forecasts.db"
FORECAST_DIR = "forecasts"
TABLE_NAME = "climate"  # Table name
VERBOSE = False

def create_database_if_dont_exist():
    if not os.path.exists(DB_PATH):  # Check if the database file exists
        conn = sqlite3.connect(DB_PATH)  # Creates the file if it doesn't exist
        cursor = conn.cursor()
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                DATE TEXT,
                Station_ID TEXT,
                UNIQUE(DATE, Station_ID)  -- Enforces uniqueness across both columns
            )
        """)
        conn.commit()
        conn.close()
        print("Database created successfully.")


def create_columns_if_dont_exist(columns: list[str]):
    """
    Checks if columns exist in the SQLite database table and adds them if they don't.
    
    Parameters:
        columns (list[str]): List of column names to check and add if missing.
    """
    create_database_if_dont_exist()

    # Connect to SQLite
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Get existing columns in the table
    cursor.execute(f"PRAGMA table_info({TABLE_NAME})")
    existing_columns = {col[1] for col in cursor.fetchall()}  # Column names

    # Add missing columns dynamically
    for column in columns:
        if column not in existing_columns:
            cursor.execute(f"ALTER TABLE {TABLE_NAME} ADD COLUMN {column} TEXT")  # Store as TEXT for flexibility
            print(f"Added missing column: {column}")

    conn.commit()
    conn.close()

def get_data_within_one_day(dt: pd.Timestamp, timedelta = pd.Timedelta(days=1)) -> pd.DataFrame:
    """
    Retrieves all data from the SQLite database where the DATE column is within 1 day of the given datetime.

    Parameters:
        dt (pd.Timestamp): The reference datetime object.

    Returns:
        pd.DataFrame: A DataFrame containing the filtered records with DATE as a datetime object.
    """
    create_database_if_dont_exist()

    # Connect to SQLite
    conn = sqlite3.connect(DB_PATH)

    # Compute date range (±1 day)
    date_min = dt - timedelta
    date_max = dt

    # SQL query to fetch records within the date range
    query = f"""
        SELECT * FROM {TABLE_NAME}
        WHERE DATE BETWEEN '{date_min.strftime('%Y-%m-%dT%H:%M:%S')}' 
                      AND '{date_max.strftime('%Y-%m-%dT%H:%M:%S')}'
    """

    # Load results into a Pandas DataFrame
    df = pd.read_sql_query(query, conn)

    # Ensure the DATE column is converted to a datetime object
    if "DATE" in df.columns:
        df["DATE"] = pd.to_datetime(df["DATE"])

    conn.close()
    return df

def add_to_database(psv_file: str):
    """
    Adds data from a PSV file into the SQLite database while ensuring:
    - Unique (DATE, Station_ID) composite key is respected.
    - Missing columns are added dynamically before insertion.
    
    Parameters:
        psv_file (str): Path to the PSV file.
    """
    # Load PSV file into Pandas
    df = pd.read_csv(psv_file, delimiter="|", low_memory=False)

    # Ensure required columns exist
    required_columns = ["DATE", "Station_ID"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in the PSV file.")

    # Add missing columns to database before inserting
    create_columns_if_dont_exist(df.columns.tolist())

    # Connect to SQLite
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Insert data while respecting the composite key constraint
    for _, row in df.iterrows():
        placeholders = ", ".join(["?" for _ in row])  # Create placeholders for query
        columns = ", ".join(row.index)  # Column names
        values = tuple(row)  # Row values
        
        try:
            cursor.execute(f"""
                INSERT INTO {TABLE_NAME} ({columns})
                VALUES ({placeholders})
            """, values)
        except sqlite3.IntegrityError:
            if VERBOSE:
                print(f"Skipped duplicate entry: DATE={row['DATE']}, Station_ID={row['Station_ID']}")
            else:
                pass

    conn.commit()
    conn.close()
    print(f"Data from '{psv_file}' added to the database.")


def get_nearest_station_dt_data(dt: datetime.datetime, lat_min, lat_max, lon_min, lon_max, timedelta = pd.Timedelta(days=1), unique=True, _tries=0) -> pd.DataFrame:


  resultant_db_data = get_data_within_one_day(dt, timedelta)
  print("Resultant db:",resultant_db_data)
  if len(resultant_db_data) == 0 and _tries < 1: # No data and haven't tried before
      print("No data found in database... Adding")
      forecast_files = forecast_downloader.download_psv_files(dt.year, lat_min, lat_max, lon_min, lon_max, month=dt.month, day=dt.day, save_dir=FORECAST_DIR, force_download=True)
      len_files = len(forecast_files)
      for i, file in enumerate(forecast_files):
          pct = 100*(i+1)/len_files
          print(f"{pct:.2f}%")
          add_to_database(file)
      return get_nearest_station_dt_data(dt, lat_min, lat_max, lon_min, lon_max, timedelta, unique, _tries + 1)
  
  # **Ensure unique Station_IDs, keeping the most recent DATE**
  if unique:
      resultant_db_data = resultant_db_data.sort_values(by="DATE", ascending=False).drop_duplicates(subset=["Station_ID"], keep="first")
  
  return resultant_db_data

if __name__ == "__main__":
    lon_min, lat_min, lon_max, lat_max = -106.64719063660635, 25.840437651866516, -93.5175532104321, 36.50050935248352
    dt = datetime.datetime(2023, 6,4, hour=6, minute=30)
    buffer_time = pd.Timedelta(minutes=30)
    unique=True
    data = get_nearest_station_dt_data(dt, lat_min, lat_max, lon_min, lon_max, timedelta=buffer_time, unique=unique)
    
    for key in data.keys():
        print(key)
    print(data)