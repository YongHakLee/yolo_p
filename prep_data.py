import requests
from tqdm import tqdm
import zipfile
from pathlib import Path
import yaml

session = requests.Session()

sharing_url = "https://iamsheep-nas4.synology.me:5001/sharing/GQeno6KBV"
response = session.get(sharing_url, verify=False)
print(f"Sharing page response: {response.status_code}")


download_url = (
    "https://iamsheep-nas4.synology.me:5001/fsdownload/GQeno6KBV/ArTaxOr_test.zip"
)
file_name = "ArTaxOr_test.zip"


with session.get(download_url, verify=False, stream=True) as r:
    r.raise_for_status()

    total_size = int(r.headers.get("content-length", 0))
    block_size = 8192

    with open(file_name, "wb") as f, tqdm(
        desc=file_name, total=total_size, unit="B", unit_scale=True, unit_divisor=1024
    ) as bar:
        for chunk in r.iter_content(chunk_size=block_size):
            f.write(chunk)
            bar.update(len(chunk))

print(f"File downloaded successfully: {file_name}")


to_path = Path.cwd() / "datasets"
to_path.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(file_name, "r") as zip_ref:
    file_list = zip_ref.namelist()
    total_files = len(file_list)

    with tqdm(total=total_files, unit="files", desc="Extracting") as bar:
        for file in file_list:
            zip_ref.extract(file, to_path)
            bar.update(1)

print(f"File unzipped successfully: {file_name}")


data = {
    "path": str(to_path / "ArTaxOr_test"),
    "train": "train/images",
    "val": "valid/images",
    "nc": 7,
    "names": [
        "Araneae",
        "Coleoptera",
        "Diptera",
        "Hemiptera",
        "Hymenoptera",
        "Lepidoptera",
        "Orthoptera",
    ],
}

yaml_filename = to_path / "ArTaxOr_test" / "data.yaml"
with open(yaml_filename, "w") as file:
    yaml.dump(data, file, default_flow_style=False)

print(f"YAML file created successfully: {yaml_filename}")
