#%%
import numpy as np
from typing import Tuple, Dict, List
import torch
from torch_lstm.word_embeddings import load_text_vectors, WordEmbeddings
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.utils import shuffle
from sklearn import metrics
from torch_lstm.examples import wiki_lit_split
import os, random

PATH = os.environ["HOME"] + "/data/glove.6B.100d.txt.gz"
# PATH = 'poetry50k.model.vec.gz'
if os.path.exists(PATH):
    glove_embeddings = load_text_vectors(PATH, 400000)
else:
    glove_embeddings = None

# start off by seeding random number generators:
RANDOM_SEED = 12345
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

data = wiki_lit_split()
train_f = data.train
vali_f = data.vali
test_f = data.test

textual = TfidfVectorizer(max_df=0.75, min_df=2, dtype=np.float32)
tokenizer = textual.build_analyzer()
numeric = make_pipeline(DictVectorizer(sparse=False), StandardScaler())

y_train = train_f.truth_value.values
y_vali = vali_f.truth_value.values
y_test = test_f.truth_value.values

from torch_lstm.classifier import DatasetConfig, SequenceClassifier
from tqdm import tqdm

text_train: List[str] = train_f.body.to_list()
text_vali: List[str] = vali_f.body.to_list()

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
config = DatasetConfig(embeddings=glove_embeddings)

words_train: List[List[str]] = config.fit_transform(text_train)
words_vali = config.transform(text_vali)

print(
    "chars.n: {}, words.n: {}".format(
        len(config.character_vocab), len(config.word_vocab)
    )
)

#%%
clf = SequenceClassifier(
    config,
    DEVICE,
    char_dim=0,
    char_lstm_dim=0,
    word_dim=100,
    word_lstm_layers=2,
    lstm_size=100,
    gen_layer=100 if config.embeddings else 0,
    hidden_layer=100,
    labels=[0, 1],
    dropout=0.0,
    activation="gelu",
    averaging=(6, 4),
)
clf.eval()

# This print statement right here helps debug the classifier (if there are any bugs!)
print(clf.forward(words_train[:8]))
clf.to(DEVICE)

loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(clf.parameters())


def eval_model(
    clf: torch.nn.Module,
    X: List[List[str]] = words_vali,
    y: np.ndarray = y_vali,
    sequence_limit=128,
    batch_size=32,
) -> float:
    clf.eval()
    N = len(X)
    epoch_pred = []
    losses = []
    with tqdm(range(0, N, batch_size)) as progress:
        for start in progress:
            end = min(start + batch_size, N)
            X_batch = [x[:sequence_limit] for x in X[start:end]]
            y_batch = torch.tensor(y[start:end], dtype=torch.long).to(DEVICE)
            y_scores = clf(X_batch)
            loss = loss_function(y_scores, y_batch)
            epoch_pred.extend(((y_scores[:, 1] - y_scores[:, 0]) > 0).tolist())
            losses.append(loss.item())
            progress.set_description(
                "Eval Acc: {:.03} Loss: {:.03}".format(
                    metrics.accuracy_score(y[: len(epoch_pred)], epoch_pred),
                    np.mean(losses),
                )
            )
    return metrics.accuracy_score(y, epoch_pred)


#%%
def train_epoch(clf: torch.nn.Module, sequence_limit=32, batch_size=32):
    clf.train()
    N = len(words_train)
    X, y = shuffle(words_train, y_train)
    epoch_pred = []
    losses = []
    with tqdm(range(0, N, batch_size)) as progress:
        for start in progress:
            clf.train()
            end = min(start + batch_size, N)
            X_batch = [x[:sequence_limit] for x in X[start:end]]
            y_batch = torch.tensor(y[start:end], dtype=torch.long).to(DEVICE)
            clf.zero_grad()
            y_scores = clf(X_batch)
            loss = loss_function(y_scores, y_batch)
            loss.backward()
            optimizer.step()

            clf.eval()
            epoch_pred.extend(((y_scores[:, 1] - y_scores[:, 0]) > 0).tolist())
            losses.append(loss.item())

            progress.set_description(
                "Train Acc: {:.03} Loss: {:.03}".format(
                    metrics.accuracy_score(y[: len(epoch_pred)], epoch_pred),
                    np.mean(losses),
                )
            )


MAX_WIDTH = -1
BATCH_SIZE = 64
for epoch in range(10):
    print("Epoch {}".format(epoch + 1))
    train_epoch(clf, MAX_WIDTH, BATCH_SIZE)
    eval_model(clf, sequence_limit=MAX_WIDTH)
# %%
