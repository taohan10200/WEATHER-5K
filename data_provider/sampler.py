import torch
from torch.utils.data.sampler import BatchSampler
from torch import Tensor
from torch.utils.data import Sampler
from typing import Iterator, Iterable, Optional, Sequence, List, TypeVar, Generic, Sized, Union
class MyBatchSampler(BatchSampler):
    def __init__(self, sampler, batch_size, drop_last):
        super(MyBatchSampler, self).__init__(sampler, batch_size, drop_last)
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            import pdb
            pdb.set_trace()
            if len(batch) == self.batch_size:
                yield batch
                batch = []

        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size

class MyRandomSampler(Sampler[int]):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.

    If with replacement, then user can specify :attr:`num_samples` to draw.

    Args:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`.
        generator (Generator): Generator used in sampling.
    """

    data_source: Sized
    replacement: bool

    def __init__(self, 
                data_source: Sized, 
                batch_size: Optional[int] = None, 
                shuffle =  True,
                drop_last: bool = False, 
                warm_batch_size = False,
                infinite = False,
                replacement: bool = False,
                num_samples: Optional[int] = None,
                generator=None) -> None:
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.warm_batch_size = warm_batch_size
        self.infinite = infinite
        if not isinstance(self.replacement, bool):
            raise TypeError(f"replacement should be a boolean value, but got replacement={self.replacement}")

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(f"num_samples should be a positive integer value, but got num_samples={self.num_samples}")
        
    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        if self.batch_size is not None:
            if self.shuffle: 
                indices = torch.randperm(n, generator=generator).tolist()
            else:
                indices = torch.arange(n, dtype=torch.int64).tolist()
            # warm up batch_size
            st = 0
            if self.warm_batch_size:
                count = 1
                while count < self.batch_size:
                    yield indices[st: st+count]
                    st += count
                    count += 1
            
            for i in range(st, n, self.batch_size):
                yield indices[i : i + self.batch_size]   
            
            if self.infinite: 
                for i in range(100):
                    for i in range(0, n, self.batch_size):
                        yield indices[i : i + self.batch_size] 

        if self.replacement:
            for _ in range(self.num_samples // 32):
                yield from torch.randint(high=n, size=(32,), dtype=torch.int64, generator=generator).tolist()
            yield from torch.randint(high=n, size=(self.num_samples % 32,), dtype=torch.int64, generator=generator).tolist()
        else:
            for _ in range(self.num_samples // n):
                yield from torch.randperm(n, generator=generator).tolist()
            yield from torch.randperm(n, generator=generator).tolist()[:self.num_samples % n]

    def __len__(self) -> int:
        if self.drop_last:
            return self.num_samples // self.batch_size
        else:
            return self.num_samples // self.batch_size + int((self.num_samples % self.batch_size >0))