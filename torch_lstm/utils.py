import numpy as np
import torch
import torch.nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from typing import List


def pack_lstm(items: List[List[torch.Tensor]], lstm: torch.nn.Module) -> torch.Tensor:
    """
    Pack items to more efficiently use the LSTM.
    """
    N = len(items)
    reorder_args = np.argsort([len(it) for it in items])[::-1]
    origin_args = torch.from_numpy(np.argsort(reorder_args))
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
