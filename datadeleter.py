import shutil
from pathlib import Path

data_dir = Path.home() / "data"

if data_dir.exists() and data_dir.is_dir():
    shutil.rmtree(data_dir)
    print(f"Removed {data_dir} recursively.")
else:
    print(f"{data_dir} does not exist.")