import os
import urllib
import urllib.request
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DatasetSplit:
    train: pd.DataFrame
    vali: pd.DataFrame
    test: pd.DataFrame

def download_file(url: str, path: str) -> str:
    """ Download url to path if path is not already downloaded! """
    # empty data files were mis-downloaded...
    if os.path.exists(path) and os.path.getsize(path) > 0:
        # don't download multiple times.
        return path
    # try connecting before creating output file...
    with urllib.request.urlopen(url) as f:
        # create output file and download the rest.
        with open(path, "wb") as out:
            out.write(f.read())
    return path

def poetry_id() -> str:
    url = 'http://ciir.cs.umass.edu/downloads/poetry/id_datasets.jsonl'
    path = 'poetry_id.jsonl'
    return download_file(url, path)

def poetry_id_split(seed=12345) -> DatasetSplit:
    df: pd.DataFrame = pd.read_json(poetry_id(), lines=True)

    features = pd.json_normalize(df.features)
    features = features.join([df.poetry, df.words])
    tv_f, test_f = train_test_split(features, test_size=0.25, random_state=seed)
    train_f, vali_f = train_test_split(tv_f, test_size=0.25, random_state=seed)

    return DatasetSplit(train_f, vali_f, test_f)