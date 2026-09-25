"""Small CPU stand-in for the DataProto methods used in coding exercises."""

from __future__ import annotations

import torch


class MiniDataProto:
    """Preserve tensor rows and non-tensor metadata through selection and chunking."""

    def __init__(self, batch, non_tensor_batch=None, meta_info=None):
        self.batch = {key: value.clone() for key, value in batch.items()}
        self.non_tensor_batch = {
            key: list(value) for key, value in (non_tensor_batch or {}).items()
        }
        self.meta_info = dict(meta_info or {})
        lengths = {value.shape[0] for value in self.batch.values()}
        lengths.update(len(value) for value in self.non_tensor_batch.values())
        if len(lengths) != 1:
            raise ValueError("all fields must have the same batch size")
        self._length = lengths.pop()

    def __len__(self):
        return self._length

    def select_idxs(self, indices):
        selected = [int(index) for index in indices]
        return MiniDataProto(
            {key: value[selected] for key, value in self.batch.items()},
            {key: [value[index] for index in selected]
             for key, value in self.non_tensor_batch.items()},
            self.meta_info,
        )

    def chunk(self, chunks):
        if chunks <= 0 or len(self) % chunks:
            raise ValueError("chunk count must divide batch size")
        width = len(self) // chunks
        return [
            self.select_idxs(range(start, start + width))
            for start in range(0, len(self), width)
        ]
