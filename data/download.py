from gdown import download
import os
import zipfile
from pathlib import Path
from tqdm import tqdm
import sys

all_files = {
    "RRS": ["1fhgxikKI8ujFpARxHs80dC3pgTCdwIhz", "15 Mb", "data/"],
}

if __name__ == '__main__':
    list_files = list(all_files.keys())

    if len(sys.argv) == 2:
        res = sys.argv[1]
    else:
        for i, k in enumerate(list_files):
            print("{}. {} ({})".format(i + 1, k, all_files[k][1]))

        res = input(
            "\nEnter the file number (1 or 2 for eg.) to download, or multiple numbers separated by a colon (1,3 for eg.):")

    if ',' in res:
        res = res.split(',')
    else:
        res = [res]

    # if we come from argv
    if len(sys.argv) == 2:
        try:
            res = [list_files.index(r.strip()) for r in res]
        except ValueError as e:
            sys.exit("{} of available downloads".format(e))
    else:
        res = [int(r.strip()) for r in res]

    assert all([0 < r <= len(list_files) for r in res]), "Numbers must be between 1 and {}".format(
        len(list_files))

    print("Selected downloads:")
    for r in res:
        print("\t{} [{}] in {}".format(list_files[r - 1],
                                       all_files[list_files[r - 1]][1],
                                       all_files[list_files[r - 1]][2]))

    for r in res:
        key = list_files[r - 1]
        outfile = os.path.join("data", key) + ".zip"
        if not os.path.exists(outfile):
            download("https://drive.google.com/uc?id=" + all_files[key][0], outfile,
                     quiet=False)

        print("Unzipping...")
        with zipfile.ZipFile(outfile, 'r') as zf:
            for member in tqdm(zf.infolist(), desc='Extracting '):
                zf.extract(member, all_files[key][2])
    print("done")
