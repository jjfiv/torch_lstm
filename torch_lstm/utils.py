import numpy as np
import torch
import torch.nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from typing import List


class SingleReprLSTM(torch.nn.Module):
    """ This module represents a LSTM that takes in a bunch of input and produces 1 single output. """

    def __init__(
        self, device: torch.device, input_dim: int, output_dim: int, bidirectional=True
    ):
        super(SingleReprLSTM, self).__init__()
        self.device = device
        self.lstm = torch.nn.LSTM(
            input_dim, output_dim, batch_first=True, bidirectional=bidirectional
        )
        self.lstm.flatten_parameters()

    def forward(self, xs: List[List[torch.Tensor]]) -> torch.Tensor:
        return pack_lstm(xs, self.lstm, self.device)


def pack_lstm(
    items: List[List[torch.Tensor]],
    lstm: torch.nn.Module,
    device: torch.device,
) -> torch.Tensor:
    """
    Pack items to more efficiently use the LSTM.
    """
    N = len(items)
    reorder_args = np.argsort([len(it) for it in items])[::-1]
    origin_args = torch.from_numpy(np.argsort(reorder_args)).to(device)
    ordered = [items[i] for i in reorder_args]
    packed_items = pack_padded_sequence(
        pad_sequence(ordered, batch_first=True),
        [len(od) for od in ordered],
        batch_first=True,
    )
    _, (hn, _) = lstm(packed_items)
    by_inst_repr = hn.transpose(0, 1).reshape(N, -1)
    # Now untwizzle
    return torch.index_select(by_inst_repr, 0, origin_args)


def test_learn_max():

    DEVICE = torch.device("cpu")
    N = 200
    D = 5
    X = np.random.randn(N, D)
    y = np.max(X, axis=1)
    assert y.shape == (N,)

    # X is now NxDx1 (every element is of size... 1)
    X = torch.from_numpy(X).float().to(DEVICE).reshape(N, D, 1)
    y = torch.from_numpy(y).float().to(DEVICE)

    m = torch.nn.Sequential(SingleReprLSTM(DEVICE, 1, D), torch.nn.Linear(D * 2, 1))
    optim = torch.optim.Adam(m.parameters())
    loss_fn = torch.nn.MSELoss()

    losses = []
    m.train()
    for _ in range(20):
        m.zero_grad()
        yp = m.forward(X).reshape(N)
        loss = loss_fn(yp, y)
        loss.backward()
        optim.step()
        losses.append(loss.item())
        print(loss.item())

    assert losses[0] > losses[-1]


if __name__ == "__main__":
    test_learn_max()