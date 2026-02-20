import gdown
import os
import sys
import rootutils
import zipfile

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
sys.path.append(os.path.dirname(os.environ["PROJECT_ROOT"]))

url = "https://drive.google.com/file/d/1lNg9vgMh4UBN10Bb1TEAxy2aJR-Q_3dC/view?usp=sharing"

# Convert to downloadable format
file_id = url.split("/d/")[1].split("/")[0]
download_url = f"https://drive.google.com/uc?id={file_id}"

dataset_path = os.path.join(os.environ["PROJECT_ROOT"], "LLaVA")

### Create the directory if it doesn't exist

if not os.path.exists(dataset_path):
    os.makedirs(dataset_path, exist_ok=True)

# Download
zip_path = os.path.join(dataset_path, "ckpt.zip")
gdown.download(download_url, output=zip_path, quiet=False)

# # Extract the path of the downloaded file
downloaded_file = zip_path

# # Unzip the file
with zipfile.ZipFile(downloaded_file, 'r') as zip_ref:
    zip_ref.extractall(dataset_path)

try:
    os.remove(downloaded_file) # Remove the zip file after extraction
except:
    pass