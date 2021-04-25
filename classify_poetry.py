#%%
import numpy as np
from typing import Tuple, Dict, List
from numpy.lib.npyio import load
import pandas as pd
import torch
from torch_lstm.word_embeddings import load_text_vectors, WordEmbeddings
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.utils import shuffle
from sklearn import metrics
import os

PATH = os.environ["HOME"] + "/data/glove.6B.50d.txt.gz"
glove_embeddings = load_text_vectors(PATH, 400000)

# start off by seeding random number generators:
RANDOM_SEED = 12345

df: pd.DataFrame = pd.read_json("poetry_id.jsonl", lines=True)

features = pd.json_normalize(df.features)
features = features.join([df.poetry, df.words])

tv_f, test_f = train_test_split(features, test_size=0.25, random_state=RANDOM_SEED)
train_f, vali_f = train_test_split(tv_f, test_size=0.25, random_state=RANDOM_SEED)

textual = TfidfVectorizer(max_df=0.75, min_df=2, dtype=np.float32)
tokenizer = textual.build_analyzer()
numeric = make_pipeline(DictVectorizer(sparse=False), StandardScaler())

y_train = train_f.poetry.values
y_vali = vali_f.poetry.values
y_test = test_f.poetry.values

from torch_lstm.classifier import DatasetConfig, SequenceClassifier
from tqdm import tqdm

text_train: List[str] = train_f.words.to_list()
text_vali: List[str] = vali_f.words.to_list()

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
config = DatasetConfig(embeddings=glove_embeddings)

words_train: List[List[str]] = config.fit_transform(text_train)

print('chars.n: {}, words.n: {}'.format(len(config.character_vocab), len(config.word_vocab)))

#%%
clf = SequenceClassifier(config, char_dim=0, char_lstm_dim=0, lstm_size=16, word_dim=8, gen_layer=16, hidden_layer=0, labels=[0,1])
clf.eval()
print(clf.forward(DEVICE, words_train[:8]))
clf.to(DEVICE)

loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(clf.parameters())

#%%
def train_epoch(
    clf: torch.nn.Module,
    sequence_limit = 32,
    batch_size = 32
):
    clf.train()
    N = len(words_train)
    X, y = shuffle(words_train, y_train)
    epoch_pred = []
    for start in tqdm(range(0, N, batch_size)):
        clf.train()
        end = min(start + batch_size, N)
        X_batch = [x[:sequence_limit] for x in X[start:end]]
        y_batch = torch.tensor(y[start:end], dtype=torch.long).to(DEVICE)

        clf.zero_grad()
        y_scores = clf(DEVICE, X_batch)
        loss = loss_function(y_scores, y_batch)
        loss.backward()
        optimizer.step()

        clf.eval()
        for lbl_scores in y_scores:
            pred = lbl_scores[1] > lbl_scores[0]
            epoch_pred.append(pred)

        print("Acc: {:.3} . Batch-Loss: {:.3}".format(metrics.accuracy_score(y[:len(epoch_pred)], epoch_pred), loss.item()))

train_epoch(clf, 16, 64)
train_epoch(clf, 32, 32)
train_epoch(clf, 32, 32)
train_epoch(clf, 64, 16)
# %%
