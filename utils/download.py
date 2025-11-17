import os
import subprocess
from pathlib import Path


def resumable_download(url: str, download_root: str, filename: str):
    download_root = Path(download_root)
    download_root.mkdir(parents=True, exist_ok=True)

    file_path = download_root / filename

    if os.path.exists(file_path):
        print(f"File {filename} already exists. Skipping download.")
        return

    print(f"Starting resumable download for {filename}...")

    command = [
        "powershell",
        "-Command",
        f"Start-BitsTransfer -Source {url} -Destination {file_path}"
    ]

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"Successfully downloaded {filename}.")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading {filename}: {e.stderr}")
        if os.path.exists(file_path):
            os.remove(file_path)
        raise e
