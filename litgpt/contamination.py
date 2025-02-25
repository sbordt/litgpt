# Contaminate the training data. We monkey-patch the read method of the BinaryReader to modify samples on-the-fly.
#
# This is how litdata.streaming is organized:
#
# StreamingDataset:  Has a member cache of type litdata.streaming.Cache.
# -----------------  Has a function self._create_cache that creates the cache.
#                    Uses cache.__getitem__ to get data.
# 
# litdata.streaming.Cache: Has a member _reader of type litdata.streaming.BinaryReader.
# ------------------------ The reader is created during __init__.
#                          __getitem__ calls _reader.read.
#
# litdata.streaming.BinaryReader: Has a method read(self, index: ChunkedIndex) -> Any 
# -------------------------------
#
# litdata.streaming.ChunkedIndex: Index object that describes a position in the pre-training data.
# -------------------------------
#
#
#

import functools
import types

from typing import Any, Callable, Dict, List, Optional, Set, Union, TypeVar

from abc import ABC, abstractmethod

from litdata.streaming.reader import BinaryReader
from litdata.streaming.sampler import ChunkedIndex
from litdata.streaming import Cache
from litdata import StreamingDataset


import torch

# Import necessary litdata components
from litdata.streaming.sampler import ChunkedIndex



class ContaminationScheduler(ABC):
    """
    A class to contaminate the training data.
    """

    @abstractmethod
    def contaminate(self, global_start_idx: int, data : torch.Tensor) -> torch.Tensor:
        pass


class DictContaminationScheduler(ContaminationScheduler):
    """
    A contamination schedule maps global indices in the training data
    to the sequences that should be inserted at those indices.
    """
    def __init__(self, schedule: Dict[int, str]):
        self.schedule = schedule

    def contaminate(self, global_start_idx: int, data : torch.Tensor) -> torch.Tensor:
        pass


def patch_binary_reader(reader :BinaryReader, 
                        contamination_scheduler: ContaminationScheduler,
                        global_offset: int = 0) -> BinaryReader:
    """
    Monkey-patch a BinaryReader's read method to modify samples.
    
    Args:
        reader: The BinaryReader instance to patch
        contamination_scheduler: The contamination schedule to use
        global_offset: The global offset of the data that is being read. To be used with CombinedStreamingDataset.
    
    Returns:
        The same reader instance with its read method patched
    """
    # Store the original read method
    original_method = reader.read
    
    # Define the new read method
    @functools.wraps(original_method)
    def read(self, index: ChunkedIndex):
        # Call the original method
        sample = original_method(index)

        # check that the sample is a flat tensor.
        if not torch.is_tensor(sample):
            raise ValueError("The sample is not a tensor. The contamination scheduler can only contaminate tensors.")
        if len(sample.shape) != 1:
            raise ValueError("The sample is not a flat tensor. The contamination scheduler can only contaminate flat tensors.")
        
        # calculate the position of the current sample in the overall training data
        global_start_idx =  global_offset + index.index * sample.shape[0]

        # Contaminate
        sample = contamination_scheduler.contaminate(global_start_idx, sample)

        return sample
    
    # Replace the original method
    reader.read = types.MethodType(read, reader)



def add_contamination(dataset: StreamingDataset, 
                      contamination_scheduler: ContaminationScheduler,
                      global_offset: int = 0) -> StreamingDataset:
    """
    Apply contamination to a StreamingDataset.

    Internally, we monkey-patch the StreamingDataset's _create_cache method to patch the underlying reader.

    Args:
        dataset: The StreamingDataset instance to patch
        contamination_scheduler: The contamination schedule to use
        global_offset: The global offset of the data that is being read. To be used with CombinedStreamingDataset.

    Returns:
        The same dataset instance with its _create_cache method patched
    """
    from litdata.utilities.env import _WorkerEnv

    # check that the cache is not already created
    if dataset.cache is not None:
        raise ValueError("The cache of the dataset has already been created. Please apply contamination before iterating over the dataset.")

    # Store the original _create_cache method
    original_method = dataset._create_cache

    # Define the new _create_cache method
    @functools.wraps(original_method)
    def _create_cache(self, worker_env: _WorkerEnv) -> Cache:
        # Call the original method
        cache = original_method(worker_env)

        # patch the reader of the newly created cache
        patch_binary_reader(cache._reader, contamination_scheduler, global_offset)
        
        return cache
    
    # Replace the original method
    dataset._create_cache = types.MethodType(_create_cache, dataset)