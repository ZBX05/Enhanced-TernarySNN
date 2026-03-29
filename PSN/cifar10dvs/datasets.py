import torch
from torch.utils.data import Subset
from typing import List

class TransformedSubset(Subset):
    def __init__(self, dataset, indices, transform=None):
        super().__init__(dataset, indices)
        self.transform = transform

    def __getitems__(self, indices: List[int]):
        # add batched sampling support when parent dataset supports it.
        # see torch.utils.data._utils.fetch._MapDatasetFetcher
        if callable(getattr(self.dataset, "__getitems__", None)):
            if self.transform is not None:
                data = self.dataset.__getitems__([self.indices[idx] for idx in indices])
                for d in data:
                    d[0] = self.transform(d[0])
                return data  # type: ignore[attr-defined]
            else:
                return self.dataset.__getitems__([self.indices[idx] for idx in indices])  # type: ignore[attr-defined]
        else:
            return [(self.transform(self.dataset[self.indices[idx]][0]), self.dataset[self.indices[idx]][1]) for idx in indices]

    def __len__(self):
        return len(self.indices)