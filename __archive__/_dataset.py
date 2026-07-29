"""
The base/pretraining dataset is a set of parquet files.
"""

import os
import argparse
import time
import requests
import pyarrow.parquet as pq
import pyarrow as pa
import random
import string
from multiprocessing import Pool

from minichat.common import get_base_dir

# The URL on the internet where the data is hosted and downloaded from on demand
BASE_URL = (
    "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main"
)
MAX_SHARD = 1822  # the last datashard is shard_01822.parquet
index_to_filename = lambda index: (
    f"shard_{index:05d}.parquet"
)  # format of the filenames
base_dir = get_base_dir()
DATA_DIR = os.path.join(base_dir, "base_data")
os.makedirs(DATA_DIR, exist_ok=True)


def generate_dummy_dataset(num_shards=1, num_rows=1000):
    """Generates a small dummy dataset for testing without downloads."""
    print(f"Generating {num_shards} dummy shards in {DATA_DIR}...")
    for i in range(num_shards):
        filename = index_to_filename(i)
        filepath = os.path.join(DATA_DIR, filename)
        if os.path.exists(filepath):
            print(f"Skipping {filepath} (already exists)")
            continue

        # Generate random text
        data = []
        for _ in range(num_rows):
            text = "".join(
                random.choices(string.ascii_letters + string.digits + " ", k=1024)
            )
            data.append(text)

        table = pa.Table.from_pydict({"text": data})
        pq.write_table(table, filepath)
        print(f"Generated {filepath}")


def list_parquet_files(data_dir=None):
    """Looks into a data dir and returns full paths to all parquet files."""
    data_dir = DATA_DIR if data_dir is None else data_dir
    parquet_files = sorted(
        [
            f
            for f in os.listdir(data_dir)
            if f.endswith(".parquet") and not f.endswith(".tmp")
        ]
    )
    parquet_paths = [os.path.join(data_dir, f) for f in parquet_files]
    return parquet_paths


def download_single_file(index):
    """Downloads a single file index, with some backoff"""
    filename = index_to_filename(index)
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        print(f"Skipping {filepath} (already exists)")
        return True

    url = f"{BASE_URL}/{filename}"
    print(f"Downloading {filename}...")

    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            temp_path = filepath + f".tmp"
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            os.rename(temp_path, filepath)
            print(f"Successfully downloaded {filename}")
            return True
        except (requests.RequestException, IOError) as e:
            print(f"Attempt {attempt}/{max_attempts} failed for {filename}: {e}")
            for path in [filepath + f".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass
            if attempt < max_attempts:
                time.sleep(2**attempt)
            else:
                return False
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download FineWeb-Edu 100BT dataset shards"
    )
    parser.add_argument(
        "-n", "--num-files", type=int, default=-1, help="Number of shards to download"
    )
    parser.add_argument(
        "-w",
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel download workers",
    )
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Generate dummy data instead of downloading",
    )
    args = parser.parse_args()

    if args.dummy:
        generate_dummy_dataset(num_shards=1)
        exit(0)

    num = MAX_SHARD + 1 if args.num_files == -1 else min(args.num_files, MAX_SHARD + 1)
    ids_to_download = list(range(num))
    with Pool(processes=args.num_workers) as pool:
        pool.map(download_single_file, ids_to_download)
