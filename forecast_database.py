import sqlite3
import forecast_downloader
import datetime
import os
import pandas as pd
import psutil


DB_PATH = "forecasts.db"
FORECAST_DIR = "forecasts"
TABLE_NAME = "climate"  # Table name
VERBOSE = True
USE_CACHE = True
VALIDATE_DATA = True

# The percentage a database can be memory-stored onto RAM while processing new data
# Set to 0.0 to go directly to disk (slow)
#
# The system creates a virtual database in RAM/memory to avoid excessive read/write 
# requests on the disk whenever new data passes. That way it speeds up processing 
# by using the faster read and write speeds of RAM first.
PCT_MEM_USAGE = 0.1

def create_database_if_dont_exist():
    if not os.path.exists(DB_PATH):  # Check if the database file exists
        conn = sqlite3.connect(DB_PATH)  # Creates the file if it doesn't exist
        cursor = conn.cursor()
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                DATE TEXT NOT NULL,  -- Store as ISO 8601 formatted text
                Station_ID TEXT NOT NULL,
                Latitude REAL NOT NULL,
                Longitude REAL NOT NULL,
                UNIQUE(DATE, Station_ID)  -- Enforces uniqueness across both columns
            )
        """)

        cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_date ON {TABLE_NAME} (DATE);")
        cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_lat_lon ON {TABLE_NAME} (Latitude, Longitude);")

        conn.commit()
        conn.close()
        print("Database created successfully.")


def create_columns_if_dont_exist(columns: list[str], conn:sqlite3.Connection | None=None):
    """
    Checks if columns exist in the SQLite database table and adds them if they don't.
    
    Parameters:
        columns (list[str]): List of column names to check and add if missing.
    """
    create_database_if_dont_exist()

    primary_database = False
    if not conn:
        primary_database = True

    # Connect to SQLite
    if primary_database:
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

    if primary_database:
        conn.close()

def get_data_within_one_day(dt: pd.Timestamp, lat_min, lat_max, lon_min, lon_max, timedelta=pd.Timedelta(days=1)) -> pd.DataFrame:
    """
    Retrieves all data from the SQLite database where the DATE column is within 1 day of the given datetime
    and within the specified bounding box (latitude and longitude range).

    Parameters:
        dt (pd.Timestamp): The reference datetime object.
        lat_min (float): Minimum latitude boundary.
        lat_max (float): Maximum latitude boundary.
        lon_min (float): Minimum longitude boundary.
        lon_max (float): Maximum longitude boundary.
        timedelta (pd.Timedelta): Time window around the reference date (default is 1 day).

    Returns:
        pd.DataFrame: A DataFrame containing the filtered records.
    """
    create_database_if_dont_exist()

    # Connect to SQLite
    conn = sqlite3.connect(DB_PATH)

    # Compute date range (±1 day)
    date_min = dt - timedelta
    date_max = dt

    # SQL query to fetch records within the date range and bounding box
    query = f"""
        SELECT * FROM {TABLE_NAME}
        WHERE DATE BETWEEN '{date_min.strftime('%Y-%m-%dT%H:%M:%S')}' 
                      AND '{date_max.strftime('%Y-%m-%dT%H:%M:%S')}'
        AND Latitude BETWEEN {lat_min} AND {lat_max}
        AND Longitude BETWEEN {lon_min} AND {lon_max}
    """

    # Load results into a Pandas DataFrame
    df = pd.read_sql_query(query, conn)

    # Ensure the DATE column is converted to a datetime object
    if "DATE" in df.columns:
        df["DATE"] = pd.to_datetime(df["DATE"])

    conn.close()
    return df

def _add_to_database(psv_file: str, conn:sqlite3.Connection):
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
    create_columns_if_dont_exist(df.columns.tolist()) # Create columns in storage db if necessary
    create_columns_if_dont_exist(df.columns.tolist(), conn) # Create columns in memory

    # Connect to SQLite
    cursor = conn.cursor()

    # Insert data while respecting the composite key constraint
    for _, row in df.iterrows():
        placeholders = ", ".join(["?" for _ in row])  # Create placeholders for query
        columns = ", ".join(row.index)  # Column names
        values = tuple(row)  # Row values
        
        try:
            cursor.execute(f"""
                INSERT OR REPLACE INTO {TABLE_NAME} ({columns})
                VALUES ({placeholders})
            """, values)
        except sqlite3.IntegrityError:
            if VERBOSE:
                print(f"Skipped duplicate entry: DATE={row['DATE']}, Station_ID={row['Station_ID']}")
            else:
                pass

    conn.commit()
    print(f"Data from '{psv_file}' added to the database.")


def _add_to_database_no_RAM(forecast_files: list[str]):
    conn = sqlite3.connect(DB_PATH)

    len_files = len(forecast_files)

    for i, file in enumerate(forecast_files):
          pct = 100*(i+1)/len_files
          print(f"{pct:.2f}%")
          _add_to_database(file, conn)


    conn.close()

def add_to_database(forecast_files: list[str]):
    create_database_if_dont_exist()

    if PCT_MEM_USAGE == 0:
        _add_to_database_no_RAM(forecast_files)
        return

    if not os.path.exists(DB_PATH):
        print("Disk database missing. Recreating...")
        create_database_if_dont_exist()

    disk_conn = sqlite3.connect(DB_PATH)
    disk_cursor = disk_conn.cursor()

    # Get table schema, recreate if missing
    disk_cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (TABLE_NAME,))
    table_ddl_row = disk_cursor.fetchone()
    
    if table_ddl_row is None:
        print(f"Table '{TABLE_NAME}' missing. Recreating in disk DB...")
        create_database_if_dont_exist()
        disk_conn = sqlite3.connect(DB_PATH)
        disk_cursor = disk_conn.cursor()
        disk_cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (TABLE_NAME,))
        table_ddl_row = disk_cursor.fetchone()
        if table_ddl_row is None:
            raise RuntimeError(f"Failed to recreate table '{TABLE_NAME}'.")

    table_ddl = table_ddl_row[0]
    disk_conn.close()

    # Create in-memory database
    mem_conn = sqlite3.connect(":memory:")
    mem_cursor = mem_conn.cursor()
    mem_cursor.execute(table_ddl)
    mem_conn.commit()

    # SQLite memory limits
    mem_cursor.execute("PRAGMA cache_size = -102400;")  # Limit cache to ~100MB
    mem_cursor.execute("PRAGMA temp_store = MEMORY;")  # Use RAM

    total_ram = psutil.virtual_memory().total
    threshold = total_ram * PCT_MEM_USAGE
    accumulated_bytes = 0

    len_files = len(forecast_files)
    for i, file in enumerate(forecast_files):
        file_size = os.path.getsize(file)
        accumulated_bytes += file_size

        pct = 100 * (i + 1) / len_files
        print(f"Processing {pct:.2f}%: {file}")

        _add_to_database(file, mem_conn)

        # Check memory usage
        cursor_mem = mem_conn.cursor()
        cursor_mem.execute("PRAGMA memory_used;")
        result = cursor_mem.fetchone()
        sqlite_mem_used = result[0] if result is not None else 0
        cursor_mem.close()

        accumulated_bytes = max(accumulated_bytes, sqlite_mem_used)

        if accumulated_bytes >= threshold:
            print(f"Dumping {accumulated_bytes / (1024*1024):.2f} MB to disk...")

            mem_cursor.execute("ATTACH DATABASE ? AS disk", (DB_PATH,))
            mem_cursor.execute(f"INSERT OR REPLACE INTO disk.{TABLE_NAME} SELECT * FROM main.{TABLE_NAME}")
            mem_conn.commit()
            mem_cursor.execute("DETACH DATABASE disk")

            # Clear memory by recreating the table
            mem_cursor.execute(f"DROP TABLE IF EXISTS {TABLE_NAME}")
            mem_cursor.execute(table_ddl)
            mem_conn.commit()

            accumulated_bytes = 0  
        
        pct_accumulated = 100 * accumulated_bytes / threshold
        print(f"In-Memory DB Size: {accumulated_bytes / (1024*1024):.2f} MB ({pct_accumulated:.2f}% of allocated memory)")

    # Final dump
    mem_cursor.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}")
    count_remaining = mem_cursor.fetchone()[0]
    if count_remaining > 0:
        print(f"Final dump to disk with {count_remaining} records...")
        mem_cursor.execute("ATTACH DATABASE ? AS disk", (DB_PATH,))
        mem_cursor.execute(f"INSERT OR REPLACE INTO disk.{TABLE_NAME} SELECT * FROM main.{TABLE_NAME}")
        mem_conn.commit()
        mem_cursor.execute("DETACH DATABASE disk")

    mem_conn.close()





def get_nearest_station_dt_data(dt: datetime.datetime, lat_min, lat_max, lon_min, lon_max, timedelta = pd.Timedelta(days=1), unique=True, _tries=0) -> pd.DataFrame:
  print(f"Attempting to get weather data. This may take a few minutes depending on database size. (past tries: {_tries})")
  exhausted_takes = not (_tries < 2)
  if exhausted_takes:
      print("Multiple tries exceeded with getting forecast data. Returning what is possible.")

  resultant_db_data = get_data_within_one_day(dt, lat_min, lat_max, lon_min, lon_max, timedelta)
  print("Resultant db:",resultant_db_data)
  if len(resultant_db_data) == 0 and not exhausted_takes:
      print("No data found in database... Adding")

      forecast_files = forecast_downloader.download_psv_files(dt.year, lat_min, lat_max, lon_min, lon_max, month=dt.month, day=dt.day, save_dir=FORECAST_DIR, force_download=(not USE_CACHE), validate=VALIDATE_DATA)
      print("Files found:", forecast_files)
      add_to_database(forecast_files)
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