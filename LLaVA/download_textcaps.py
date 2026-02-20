import os
import urllib.request
import zipfile
import rootutils

rootutils.setup_root(__file__, indicator=".project-root")

ROOT = os.path.join(os.environ.get("PROJECT_ROOT"), "LLaVA", "textcaps")
# ROOT = "/home/victorletzelter/workspace/LoRA-MCL_cleaned/LLaVA/textcaps"
os.makedirs(ROOT, exist_ok=True)

FILES = {
    # annotations
    "TextCaps_0.1_train.json":
        "https://dl.fbaipublicfiles.com/textvqa/data/textcaps/TextCaps_0.1_train.json",
    "TextCaps_0.1_val.json":
        "https://dl.fbaipublicfiles.com/textvqa/data/textcaps/TextCaps_0.1_val.json",
    "TextCaps_0.1_test.json":
        "https://dl.fbaipublicfiles.com/textvqa/data/textcaps/TextCaps_0.1_test.json",

    # images
    "train_val_images.zip":
        "https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip",
    "test_images.zip":
        "https://dl.fbaipublicfiles.com/textvqa/images/test_images.zip",
}


def download(url, path):
    if os.path.exists(path):
        print(f"Skipping {path}")
        return
    print(f"Downloading {path}")
    urllib.request.urlretrieve(url, path)


# download files
for name, url in FILES.items():
    download(url, os.path.join(ROOT, name))

# extract images directly into ROOT
for zip_name in ["train_val_images.zip", "test_images.zip"]:
    zip_path = os.path.join(ROOT, zip_name)
    if not os.path.exists(zip_path):
        continue
    print(f"Extracting {zip_name}")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(ROOT)
