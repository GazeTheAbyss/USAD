import math
import numpy as np

class SlidingWindowDataset():

    def __init__(self, values, window_size):
        self._values = values
        self._window_size = window_size
        self._strided_values = self._to_windows(self._values)

    def _to_windows(self, values):
        sliding_windows = []
        for i in range(values.shape[0] - self._window_size + 1):
            sliding_windows.append(self._values[i:i+self._window_size])
        return np.array(sliding_windows)
    
    def __getitem__(self, index):
        return np.copy(self._strided_values[index]).astype(np.float32)

    def __len__(self):
        return np.size(self._strided_values, 0)


class SlidingWindowDataLoader():

    def __init__(self, dataset, batch_size, shuffle=False, drop_last=False):
        self._dataset = dataset
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._drop_last = drop_last

        if self._shuffle:
            self._idxs = np.random.permutation(self._dataset.shape[0])
        else:
            self._idxs = np.arange(self._dataset.shape[0])

        if self._drop_last:
            self._total = len(self._dataset) // self._batch_size
        else:
            self._total = (len(self._dataset) + self._batch_size - 1) // self._batch_size

    def get_item(self, idx):

        if (idx + 1) * self._batch_size > self._dataset.shape[0]:
            batch_idx = self._idxs[idx * self._batch_size:]
            batch =  self._dataset[batch_idx]
        else:
            batch_idx = self._idxs[idx * self._batch_size:(idx + 1) * self._batch_size]
            batch =  self._dataset[batch_idx]

        return batch
