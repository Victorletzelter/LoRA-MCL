import os
import gdown
import h5py
import argparse

def download_data(url):
    # Convert to downloadable format
    file_id = url.split("/d/")[1].split("/")[0]
    download_url = f"https://drive.google.com/uc?id={file_id}"

    # Extract the filename from the url
    filename = url.split("/")[-1]

    if "split" in filename:
        dataset_path = os.path.join(os.environ["PROJECT_ROOT"], "data", "split", filename)
    else:
        dataset_path = os.path.join(os.environ["PROJECT_ROOT"], "data", filename)

    if not os.path.exists(dataset_path):
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)

    # Download
    gdown.download(download_url, output=dataset_path, quiet=False)

def merge_split_files(output_file_name):
    output_path = os.path.join(os.environ["PROJECT_ROOT"], "data", output_file_name)
    # Get all files in the split directory
    split_dir = os.path.join(os.environ["PROJECT_ROOT"], "data", "split")
    split_files = [f for f in os.listdir(split_dir) if f.endswith(".h5")]

    with h5py.File(split_files[0], "r") as f0:
        keys = list(f0.keys())
        total_rows = 0
        for fn in split_files:
            with h5py.File(fn, "r") as f:
                total_rows += f[keys[0]].shape[0]

        with h5py.File(output_path, "w") as dst:
            # create empty datasets in the destination
            for key in keys:
                shape = list(f0[key].shape)
                shape[0] = total_rows  # replace first dim with full length
                maxshape = (None,) + tuple(shape[1:])
                dst.create_dataset(key, shape=tuple(shape), dtype=f0[key].dtype, maxshape=maxshape)

            # fill datasets sequentially
            offset = 0
            for fn in split_files:
                with h5py.File(fn, "r") as src:
                    n = src[keys[0]].shape[0]
                    for key in keys:
                        dst[key][offset:offset+n] = src[key][:]
                offset += n

def main():

    download_audiocaps = True
    download_clotho = True

    if download_audiocaps is True:
        # (Preprocessed) AudioCaps
        urls_audiocaps = [
            "https://drive.google.com/file/d/125uaTgmoF7Fp5fj7kjO11AmomWHejxmO/view?usp=sharing", # train, split 0
            "https://drive.google.com/file/d/1d5PWthynJ2VcfBUmWizZxxmcNOLUq74c/view?usp=sharing", # train, split 1
            "https://drive.google.com/file/d/1r8rZ46DvoSfvk7MEZpTJrobhkor0kRsJ/view?usp=sharing", # train, split 2
            "https://drive.google.com/file/d/1EUg0qn8J4d4_4KjZpWKS49h84FeK6RSs/view?usp=sharing", # train, split 3
            "https://drive.google.com/file/d/1b3EmvCClBpkgzH2g86qBQ6dyPLwlQFcL/view?usp=sharing", # val
            "https://drive.google.com/file/d/1AlRnYho4ls1R8qJDvJqAC88-KrZYcTm6/view?usp=sharing", # test
        ]

        # Download data
        for url in urls_audiocaps:
            download_data(url=url)

        # Merge split files
        merge_split_files(output_file_name="audiocaps_train_resample_mean_raw_ident_raw.hdf")

    if download_clotho is True:
        # (Preprocessed) Clotho
        urls_clotho = [
            "https://drive.google.com/file/d/12DX_dukIa8JVFxXoH0flCc-qgy_00KpJ/view?usp=sharing", # dev
            "https://drive.google.com/file/d/1mYsJVmTVA1m5zwJVZ9zSqMQlepvQ4P3R/view?usp=sharing", # eval
            "https://drive.google.com/file/d/1SQD1aHOgQcDDYxU3uc5_4tY1qS1hxYs0/view?usp=sharing", # val
        ]

        # Download data
        for url in urls_clotho:
            download_data(url=url)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--download_audiocaps", action="store_true")
    p.add_argument("--download_clotho", action="store_true")
    args = p.parse_args()

    if args.download_audiocaps:
        download_audiocaps = True
    if args.download_clotho:
        download_clotho = True
    main()