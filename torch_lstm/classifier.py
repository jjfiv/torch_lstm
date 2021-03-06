from torch.nn.modules import loss
from torch_lstm.word_embeddings import WordEmbeddings
from torch.nn.functional import dropout, embedding
from .utils import SingleReprLSTM
import torch
from torch import nn
from torch.nn.functional import dropout
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass, field
from .word_embeddings import WordEmbeddings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import shuffle
from tqdm import tqdm


@dataclass
class DatasetConfig:
    analyzer = TfidfVectorizer().build_analyzer()
    character_vocab: Dict[str, int] = field(default_factory=dict)
    word_vocab: Dict[str, int] = field(default_factory=dict)
    embeddings: Optional[WordEmbeddings] = None

    def char_vocab_len(self):
        return len(self.character_vocab) + 1

    def word_vocab_len(self):
        if self.embeddings:
            return len(self.embeddings.word_to_row)
        else:
            return len(self.word_vocab) + 1

    def fit_transform(self, dataset: List[str]) -> List[List[str]]:
        xs = []
        for text in dataset:
            words = ["<>"]
            for w in self.analyzer(text):
                words.append(w)
                self.fit_word(w)
            words.append("</>")
            xs.append(words)
        return xs

    def transform(self, dataset: List[str]) -> List[List[str]]:
        xs = []
        for text in dataset:
            words = ["<>"]
            for w in self.analyzer(text):
                words.append(w)
            words.append("</>")
            xs.append(words)
        return xs

    def fit_word(self, word: str):
        for ch in self.word_to_chars(word):
            if ch not in self.character_vocab:
                self.character_vocab[ch] = len(self.character_vocab) + 1
        if self.embeddings:
            self.word_vocab[word] = self.embeddings.word_to_row.get(word, 0)
        else:
            if word not in self.word_vocab:
                self.word_vocab[word] = len(self.word_vocab) + 1

    def word_to_chars(self, word: str) -> List[str]:
        return [ch for ch in "^{}$".format(word)]

    def word_to_char_indices(self, word: str) -> List[int]:
        return [self.character_vocab.get(ch, 0) for ch in self.word_to_chars(word)]


def activation_layer(name: str):
    if name == "relu6":
        return nn.ReLU6()
    elif name == "relu":
        return nn.ReLU()
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "gelu":
        return nn.GELU()
    elif name == "tanh":
        return nn.Tanh()
    else:
        raise ValueError(name)


def activation_func(name: str):
    import torch.nn.functional

    if name == "relu6":
        return torch.nn.functional.relu6
    elif name == "relu":
        return torch.nn.functional.relu
    elif name == "sigmoid":
        return torch.nn.functional.sigmoid
    elif name == "gelu":
        return torch.nn.functional.gelu
    elif name == "tanh":
        return torch.nn.functional.tanh
    else:
        raise ValueError(name)


@dataclass
class SlidingAverage:
    width: int
    stride: int
    weighted: bool = False  # average

    def to_layer(self, dim: int) -> torch.nn.Module:
        if not self.weighted:
            return nn.AvgPool1d(
                kernel_size=self.width, stride=self.stride, ceil_mode=True
            )
        else:
            return nn.Conv1d(dim, dim, kernel_size=self.width, stride=self.stride)


class SequenceClassifier(nn.Module):
    """ This module represents a text classification algorithm. """

    def __init__(
        self,
        config: DatasetConfig,
        device: torch.device,
        char_dim: int = 16,
        char_lstm_dim: int = 16,
        lstm_size: int = 32,
        word_dim: int = 300,  # inferred from pretrained_words
        word_lstm_layers: int = 1,
        gen_layer: int = 100,
        hidden_layer: int = 0,
        dropout: float = 0.1,
        labels: List[int] = [0, 1],
        activation: str = "gelu",
        averaging: Optional[Union[Tuple[int, int], SlidingAverage]] = None,
    ):
        super(SequenceClassifier, self).__init__()
        self.config = config
        self.device = device
        self.dropout = dropout
        self.labels = labels
        self.activation = activation_func(activation)
        word_repr_size = 0
        if char_dim > 0:
            self.char_embed = nn.Embedding(len(config.character_vocab) + 1, char_dim)
            self.char_lstm = SingleReprLSTM(
                device, char_dim, char_lstm_dim, bidirectional=True
            )
            word_repr_size += char_lstm_dim * 2
        if config.embeddings is not None:
            (NW, ND) = config.embeddings.vectors.shape
            self.word_embed = nn.Embedding(NW, ND)
            self.word_embed.weight.data.copy_(
                torch.from_numpy(config.embeddings.vectors)
            )
            self.word_embed.requires_grad_(False)
            word_dim = ND
        else:
            self.word_embed = nn.Embedding(config.word_vocab_len(), word_dim)
        # word_repr directly, or through gen_layer
        if word_dim > 0:
            if gen_layer > 0:
                self.gen_layer = nn.Linear(word_repr_size + word_dim, gen_layer)
                word_repr_size = gen_layer
            else:
                word_repr_size += word_dim
        if averaging:
            if isinstance(averaging, SlidingAverage):
                self.kernel_size = averaging.width
                self.word_avg = averaging.to_layer(word_repr_size)
            else:
                (size, stride) = averaging
                self.kernel_size = size
                self.word_avg = nn.AvgPool1d(
                    kernel_size=size, stride=stride, ceil_mode=True
                )
        self.word_lstm = SingleReprLSTM(
            device,
            word_repr_size,
            lstm_size,
            bidirectional=True,
            layers=word_lstm_layers,
        )
        lstm_output_size = self.word_lstm.get_output_width()
        if hidden_layer > 0:
            self.prediction_layer = nn.Linear(lstm_output_size, hidden_layer)
            self.output_layer = nn.Linear(hidden_layer, len(labels))
        else:
            self.output_layer = nn.Linear(lstm_output_size, len(labels))
        self.to(device)

    def forward(self, xs: List[List[str]]) -> torch.Tensor:
        activate = self.activation
        word_outputs = []
        word_vocab = self.config.word_vocab
        for words in xs:
            word_reprs: List[torch.Tensor] = []
            if hasattr(self, "word_embed"):
                words_i = torch.tensor(
                    [word_vocab.get(w, 0) for w in words], dtype=torch.long
                ).to(self.device)
                words_e = self.word_embed(words_i).reshape(1, len(words), -1)
                word_reprs.append(words_e)
            if hasattr(self, "char_embed"):
                word_char_reprs: List[torch.Tensor] = []
                for w in words:
                    chars_i = torch.tensor(
                        self.config.word_to_char_indices(w), dtype=torch.long
                    ).to(self.device)
                    chars_e = self.char_embed(chars_i).to(self.device)
                    word_char_reprs.append(chars_e)

                char_reprs = self.char_lstm(word_char_reprs).reshape(1, len(words), -1)
                word_reprs.append(char_reprs)

            # concat embeddings if needed:
            if len(word_reprs) > 1:
                word_output = torch.cat(word_reprs, dim=2)
            else:
                word_output = word_reprs[0]

            if hasattr(self, "gen_layer"):
                word_vecs = dropout(
                    activate(self.gen_layer(dropout(word_output, p=self.dropout))),
                    p=self.dropout,
                )
            else:
                word_vecs = dropout(word_output, p=self.dropout)

            if hasattr(self, "word_avg") and len(words) > self.kernel_size:
                # average adjacent words to speed up LSTM.
                avg_word_vecs = self.word_avg(word_vecs.transpose(1, 2))
                lstm_input = avg_word_vecs.reshape(
                    avg_word_vecs.shape[1], avg_word_vecs.shape[2]
                )
                word_outputs.append(lstm_input.transpose(0, 1))
            else:
                lstm_input = word_vecs.transpose(1, 2).reshape(
                    word_vecs.shape[1], word_vecs.shape[2]
                )
                word_outputs.append(lstm_input)

        # do the LSTM in bulk; it really matters:
        lstm_output = self.word_lstm(word_outputs)

        if hasattr(self, "prediction_layer"):
            layer = activate(
                self.prediction_layer(dropout(lstm_output, p=self.dropout))
            )
            return dropout(self.output_layer(layer), p=self.dropout)
        else:
            return dropout(self.output_layer(lstm_output), p=self.dropout)


class AverageEmbeddingClassifier(torch.nn.Module):
    """ This classifier is more efficient; built around a Average of Embeddings. """

    def __init__(
        self,
        config: DatasetConfig,
        device: torch.device,
        word_dim: int = 300,  # inferred from pretrained_words
        hidden_layers: List[int] = [300],
        dropout: float = 0.0,
        labels: List[int] = [0, 1],
        activation: str = "gelu",
    ):
        super(AverageEmbeddingClassifier, self).__init__()
        self.config = config
        self.device = device
        self.dropout = dropout
        self.labels = labels
        if config.embeddings is not None:
            (NW, ND) = config.embeddings.vectors.shape
            self.embeddings = nn.Embedding(NW, ND)
            self.embeddings.weight.data.copy_(
                torch.from_numpy(config.embeddings.vectors)
            )
            self.embeddings.requires_grad_(False)
            word_dim = ND
        else:
            self.embeddings = nn.Embedding(config.word_vocab_len(), word_dim)

        layers = []
        for i, dim in enumerate(hidden_layers):
            input_dim = word_dim
            if i > 0:
                input_dim = hidden_layers[i - 1]
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.Dropout(p=self.dropout))
            layers.append(activation_layer(activation))
        if len(hidden_layers) > 0:
            layers.append(nn.Linear(hidden_layers[-1], len(labels)))
        else:
            layers.append(nn.Linear(word_dim, len(labels)))
        self.mlp = nn.Sequential(*layers)
        self.to(self.device)

    def forward(self, xs: List[List[str]]) -> torch.Tensor:
        word_vocab = self.config.word_vocab
        avg_embed = []
        for words in xs:
            words_i = torch.tensor(
                [word_vocab.get(w, 0) for w in words], dtype=torch.long
            ).to(self.device)
            words_e = self.embeddings(words_i)
            avg_embed.append(torch.mean(words_e, dim=0).view(1, -1))
        return self.mlp(torch.cat(avg_embed, dim=0))


#%%
def train_epoch(
    clf: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_function: torch.nn.Module,
    words_train: List[List[str]],
    y_train: List[int],
    sequence_limit=32,
    batch_size=32,
    device="cpu",
) -> List[float]:
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
            y_batch = torch.tensor(y[start:end], dtype=torch.long).to(device)
            clf.zero_grad()
            y_scores = clf(X_batch)
            loss = loss_function(y_scores, y_batch)
            loss.backward()
            optimizer.step()

            clf.eval()
            epoch_pred.extend(((y_scores[:, 1] - y_scores[:, 0]) > 0).tolist())
            losses.append(loss.item())
            progress.set_description("Train Loss: {:.03}".format(np.mean(losses[-10:])))
    return losses


def test_tiny():
    device = torch.device("cpu")
    y_train = [1, 1, 0, 0]
    X_train = ["I am happy.", "This is great!", "I am sad.", "This is bad."]
    config = DatasetConfig()
    X_ready = config.fit_transform(X_train)
    clf_args = {
        "config": config,
        "device": device,
        "char_dim": 4,
        "char_lstm_dim": 4,
        "word_dim": 10,
        "lstm_size": 10,
        "gen_layer": 10,
        "hidden_layer": 10,
        "labels": [0, 1],
        "dropout": 0.0,
        "activation": "gelu",
    }
    for clf in [
        SequenceClassifier(
            **clf_args,
            averaging=(2, 1),
        ),
        SequenceClassifier(
            **clf_args,
            averaging=None,
        ),
        SequenceClassifier(
            **clf_args,
            averaging=SlidingAverage(2, 1),
        ),
        SequenceClassifier(
            **clf_args,
            averaging=SlidingAverage(2, 1, True),
        ),
    ]:
        optimizer = torch.optim.Adam(params=clf.parameters())
        loss_function = torch.nn.CrossEntropyLoss()
        train_epoch(clf, optimizer, loss_function, X_ready, y_train)


def test_tiny_avg():
    device = torch.device("cpu")
    y_train = [1, 1, 0, 0]
    X_train = ["I am happy.", "This is great!", "I am sad.", "This is bad."]
    config = DatasetConfig()
    X_ready = config.fit_transform(X_train)
    clf_args = {
        "config": config,
        "device": device,
        "word_dim": 10,
        "labels": [0, 1],
        "dropout": 0.0,
        "activation": "gelu",
    }
    for clf in [
        AverageEmbeddingClassifier(
            **clf_args,
            hidden_layers=[],
        ),
        AverageEmbeddingClassifier(
            **clf_args,
            hidden_layers=[10],
        ),
        AverageEmbeddingClassifier(
            **clf_args,
            hidden_layers=[10, 10],
        ),
    ]:
        optimizer = torch.optim.Adam(params=clf.parameters())
        loss_function = torch.nn.CrossEntropyLoss()
        train_epoch(clf, optimizer, loss_function, X_ready, y_train)
