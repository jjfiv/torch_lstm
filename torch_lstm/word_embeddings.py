from dataclasses import dataclass
from typing import Dict, Optional, TextIO
import numpy as np
import gzip
from tqdm import tqdm

@dataclass
class WordEmbeddings:
    word_to_row: Dict[str, int]
    vectors: np.ndarray

def _read_text_vectors(fp: TextIO, total: Optional[int]) -> WordEmbeddings:
    word_to_row = {}
    vectors = []
    for line in tqdm(fp, total=total):
        split = line.index(" ")
        word = line[:split]
        data = np.fromstring(line[split + 1 :], dtype=np.float32, sep=" ")
        if len(data) == 1:
            continue
        word_to_row[word] = len(vectors)
        vectors.append( data )
    return WordEmbeddings(word_to_row, np.vstack(vectors))

def load_text_vectors(path: str, total: Optional[int]=None) -> WordEmbeddings:
    if path.endswith('.gz'):
        return _read_text_vectors(gzip.open(path, 'rt'), total)
    else:
        return _read_text_vectors(open(path), total)
