import os
import urllib.request
from pathlib import Path
from rich.console import Console

console = Console()

def download_file(url: str, dest_folder: Path, filename: str) -> Path:
    """Downloads a file if it doesn't exist."""
    dest_folder.mkdir(parents=True, exist_ok=True)
    filepath = dest_folder / filename
    if not filepath.exists():
        console.print(f"Downloading {filename} from {url}...")
        try:
            urllib.request.urlretrieve(url, filepath)
            console.print(f"[green]Successfully downloaded {filepath}[/green]")
        except Exception as e:
            console.print(f"[red]Failed to download {filename}: {e}[/red]")
            if filepath.exists():
                os.remove(filepath)
            raise
    return filepath