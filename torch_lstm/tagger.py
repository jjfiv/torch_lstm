import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from typing import List
from torch_lstm.classifier import DatasetConfig, activation_func
from torch.nn.functional import dropout
from torch import sigmoid
from .utils import EachReprLSTM, SingleReprLSTM
from sklearn import preprocessing
import random


class SequenceTagger(nn.Module):
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
        output_labels: List[str] = ["POSITIVE", "NEGATIVE"],
        activation: str = "gelu",
    ):
        super(SequenceTagger, self).__init__()
        self.config = config
        self.device = device
        self.dropout = dropout
        self.output_labels = output_labels
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
        self.word_lstm = EachReprLSTM(
            device,
            word_repr_size,
            lstm_size,
            bidirectional=True,
            layers=word_lstm_layers,
        )
        lstm_output_size = self.word_lstm.get_output_width()
        if hidden_layer > 0:
            self.prediction_layer = nn.Linear(lstm_output_size, hidden_layer)
            self.output_layer = nn.Linear(hidden_layer, len(output_labels))
        else:
            self.output_layer = nn.Linear(lstm_output_size, len(output_labels))
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

            lstm_input = word_vecs.transpose(1, 2).reshape(
                1, word_vecs.shape[1], word_vecs.shape[2]
            )
            word_outputs.append(lstm_input)

        # do the LSTM in bulk; it really matters:
        N = len(word_outputs)
        print(word_outputs[0].shape)
        lstm_output = self.word_lstm(torch.hstack(word_outputs))

        if hasattr(self, "prediction_layer"):
            layer = activate(
                self.prediction_layer(dropout(lstm_output, p=self.dropout))
            )
            return sigmoid(dropout(self.output_layer(layer), p=self.dropout))
        else:
            return sigmoid(dropout(self.output_layer(lstm_output), p=self.dropout))


def test_learn_max_ranges():
    DEVICE = torch.device("cpu")
    N = 200
    D = 10
    X = np.random.randn(N, D)
    maxes = np.max(X, axis=1)
    assert maxes.shape == (N,)
    labels = ["..0.25", "0.25..0.75", "0.75.."]
    ys = []
    for i in range(N):
        seq = []
        for _ in range(D):
            val = maxes[i]
            if val <= 0.25:
                seq.append(np.array([1, 0, 0]))
            elif val <= 0.75:
                seq.append(np.array([0, 1, 0]))
            else:
                seq.append(np.array([0, 0, 1]))
        pred = np.vstack(seq).reshape(1, D, 3)
        ys.append(pred)
    print(ys[0].shape)
    y = np.vstack(ys)
    print(y.shape)
    assert y.shape == (N, D, len(labels))

    # X is now NxDx1 (every element is of size... 1)
    X = torch.from_numpy(X).float().to(DEVICE).reshape(N, D, 1)
    y = torch.from_numpy(y).float().to(DEVICE).reshape(N, D, len(labels))

    m = torch.nn.Sequential(
        EachReprLSTM(DEVICE, 1, D),
        torch.nn.Linear(D * 2, len(labels)),
        torch.nn.Sigmoid(),
    )
    optim = torch.optim.Adam(m.parameters())
    loss_fn = torch.nn.BCELoss()

    losses = []
    m.train()
    for i in range(100):
        m.zero_grad()
        yp = m.forward(X).reshape(N, D, len(labels))
        loss = loss_fn(yp, y)
        loss.backward()
        optim.step()
        losses.append(loss.item())
        if i % 10 or (i + 1 == 100):
            print(loss.item())

    assert losses[0] > losses[-1]


def test_learn_capitalized():
    DEVICE = torch.device("cpu")
    config = DatasetConfig()
    examples = []
    labels = []
    D = 5
    N = 1000
    for _ in range(N):
        ex = []
        lbl = []
        for _ in range(D):
            if random.choice([True, False]):
                ex.append("Word")
                lbl.append("UPPER")
            else:
                ex.append("word")
                lbl.append("LOWER")
        examples.append(ex)
        labels.append(lbl)
    m = SequenceTagger(
        config,
        DEVICE,
        char_dim=8,
        char_lstm_dim=8,
        word_dim=8,
        lstm_size=16,
        gen_layer=0,
        hidden_layer=16,
        output_labels=["UPPER", "LOWER"],
    )

    optim = torch.optim.Adam(m.parameters())
    loss_fn = torch.nn.BCELoss()
    losses = []
    m.train()
    le = preprocessing.OneHotEncoder(dtype=np.float32)
    y = (
        torch.from_numpy(le.fit_transform(labels).todense())
        .reshape(N, D, len(m.output_labels))
        .to(DEVICE)
    )
    for i in range(100):
        m.zero_grad()
        yp = m.forward(examples).reshape(N, D, len(m.output_labels))
        loss = loss_fn(yp, y)
        loss.backward()
        optim.step()
        losses.append(loss.item())
        if i % 10 or (i + 1 == 100):
            print(loss.item())


if __name__ == "__main__":
    test_learn_capitalized()
    test_learn_max_ranges()
