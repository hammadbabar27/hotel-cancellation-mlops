"""Download the ORIGINAL public Kaggle dataset, not synthetic samples."""
from pathlib import Path
import shutil
import kagglehub

PROJECT = Path(__file__).resolve().parents[1]
source_dir = Path(kagglehub.dataset_download("ahsan81/hotel-reservations-classification-dataset"))
files = list(source_dir.rglob("Hotel Reservations.csv"))
if len(files) != 1:
    raise FileNotFoundError(f"Expected one 'Hotel Reservations.csv' under {source_dir}, found {len(files)}")
out = PROJECT / "data" / "hotel_reservations.csv"
out.parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(files[0], out)
print(f"Downloaded real Kaggle data to {out} ({out.stat().st_size:,} bytes)")
