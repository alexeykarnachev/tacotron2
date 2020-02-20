from torch.utils.data import Sampler
import numpy as np


def get_chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def flatten(l):
    return [item for sublist in l for item in sublist]


class LengthSortSampler(Sampler):
    def __init__(self, data_source, bs):
        super().__init__(data_source)
        self.data_source = data_source
        self.bs = bs

        try:
            int(self.data_source[0])
            lengths = self.data_source
        except TypeError:
            lengths = [len(x) for x in self.data_source]

        inds = np.argsort(lengths)[::-1]
        chunks = list(get_chunks(inds, bs))
        chunk_inds = list(range(len(chunks) - 1))
        np.random.shuffle(chunk_inds)
        chunk_inds = list(chunk_inds) + [len(chunk_inds)]
        self.inds = flatten([chunks[i] for i in chunk_inds])

    def __len__(self):
        return len(self.data_source)

    def __iter__(self):
        return iter(self.inds)
