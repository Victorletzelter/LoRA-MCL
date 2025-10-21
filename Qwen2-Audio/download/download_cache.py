import gdown
import os
import sys
import rootutils
import zipfile

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
sys.path.append(os.path.dirname(os.environ["PROJECT_ROOT"]))

def download_zip(url):

    # # Convert to downloadable format
    file_id = url.split("/d/")[1].split("/")[0]
    download_url = f"https://drive.google.com/uc?id={file_id}"

    dataset_path = os.path.join(os.environ["PROJECT_ROOT"], "cache")

    # Create the directory if it doesn't exist

    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path, exist_ok=True)

    # Download
    gdown.download(download_url, output=dataset_path, quiet=False)

    # Check that the .part file is the most recent in the directory
    part_files = [e for e in os.listdir(dataset_path) if '.part' in e]
    if len(part_files) == 0:
        raise ValueError(f"No .part file found in {dataset_path}")
    part_files.sort(key=lambda x: os.path.getmtime(os.path.join(dataset_path, x)))
    downloaded_file = os.path.join(dataset_path, part_files[-1])

    if os.path.exists(downloaded_file):
        # # Unzip the file
        with zipfile.ZipFile(downloaded_file, 'r') as zip_ref:
            zip_ref.extractall(dataset_path)

        try:
            os.remove(downloaded_file) # Remove the zip file after extraction
        except:
            pass
    else:
        print(f"File {downloaded_file} does not exist")

url_aac_metrics = "https://drive.google.com/file/d/1l6Cumbo9MC3WD3cphOyeLd55BYODZ0q4/view?usp=sharing"
download_zip(url_aac_metrics)

url_local_cache_torch = "https://drive.google.com/file/d/1P9D8O_bySlIsBeMSsJuS5MChvvJ_F06b/view?usp=sharing"
download_zip(url_local_cache_torch)

url_local_cache_hf = "https://drive.google.com/file/d/1xF3ycN2IEBWQ4Og63EyAh2JBcPr7BPYd/view?usp=sharing"
download_zip(url_local_cache_hf)