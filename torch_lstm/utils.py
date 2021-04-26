import numpy as np
import torch
import torch.nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from typing import List


class SingleReprLSTM(torch.nn.Module):
    """ This module represents a LSTM that takes in a bunch of input and produces 1 single output. """

    def __init__(self, input_dim: int, output_dim: int, bidirectional=True):
        super(SingleReprLSTM, self).__init__()
        self.lstm = torch.nn.LSTM(
            input_dim, output_dim, batch_first=True, bidirectional=bidirectional
        )
        self.lstm.flatten_parameters()

    def forward(
        self, device: torch.device, xs: List[List[torch.Tensor]]
    ) -> torch.Tensor:
        print(xs[0][0].shape)
        return pack_lstm(xs, self.lstm, device)


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
