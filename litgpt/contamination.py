# Add data contamination to litdata's StreamingDataset.
# 
# We decorate (monkey-patch) the read method of the BinaryReader to modify samples on-the-fly.
#
# This is how the relevant part of litdata works (tested with litdata version 0.28.0).
#
# litdata.StreamingDataset:  Has a member cache of type litdata.streaming.Cache.
# -------------------------  Has a function self._create_cache that creates the cache.
#                            Uses cache.__getitem__ to load training data.
# 
# litdata.streaming.Cache: Has a member _reader of type litdata.streaming.BinaryReader.
# ------------------------ The reader is created during __init__.
#                          __getitem__ defers to _reader.read.
#
# litdata.streaming.BinaryReader: Has a method read(self, index: ChunkedIndex) -> Any 
# -------------------------------
#
# litdata.streaming.ChunkedIndex: Index object that describes a position in the training data.
# -------------------------------
#

import functools
import types

from typing import Dict

from abc import ABC, abstractmethod

# Import necessary litdata components
from litdata import StreamingDataset
from litdata.streaming import Cache
from litdata.streaming.sampler import ChunkedIndex
from litdata.streaming.reader import BinaryReader

import bisect
import torch
from typing import Dict





class ContaminationScheduler(ABC):
    """
    A class to contaminate the training data.
    """

    @abstractmethod
    def contaminate(self, global_start_idx: int, data : torch.Tensor) -> torch.Tensor:
        pass


class DictContaminationScheduler(ContaminationScheduler):
    """
    A class that holds a dictionary mapping integer indices to flat torch.Tensors.
    When contaminate(...) is called, it replaces the corresponding slices in the
    provided data with these stored tensors, if they overlap in any way with the data.
    """
    
    def __init__(self, contamination_dict: Dict[int, torch.Tensor]):
        """
        Args:
            contamination_dict: A dictionary where
                - key: an integer index (global position)
                - value: a 1D torch.Tensor to be placed starting at that index
        """
        self.contamination_dict = contamination_dict
        # Keep the keys in a sorted list for efficient range queries
        self._sorted_keys = sorted(contamination_dict.keys())

        # Compute the maximum length of any contamination tensor for overlap checks
        self._max_contamination_length = 0
        if contamination_dict:
            self._max_contamination_length = max(t.shape[0] for t in contamination_dict.values())

    def contaminate(self, global_start_idx: int, data: torch.Tensor) -> torch.Tensor:
        """
        Replaces parts of `data` with the relevant contamination tensors whose keys
        fall into or partially overlap with [global_start_idx, global_start_idx + len(data) - 1].
        
        Args:
            global_start_idx: The "global" index where the provided data starts.
                              e.g., data[0] corresponds to index global_start_idx,
                              data[1] corresponds to global_start_idx + 1, etc.
            data: A 1D torch.Tensor of shape [N] whose values may be replaced by 
                  contamination data.

        Returns:
            The same `data` tensor (mutated in-place) with all applicable 
            contamination applied (including partial overlaps).
        """
        # copy the data tensor (litdata crashes if we don't do this - my guess is that the original tensor offers a view into some memory that we should not modify)
        data = data.clone()

        data_len = data.shape[0]
        data_start = global_start_idx
        data_end   = global_start_idx + data_len - 1
        
        # We want to include dictionary keys that might overlap with the range
        # [data_start, data_end]. If a contamination starts at key < data_start
        # but has length enough to reach data_start, it still overlaps.
        # e.g. key = data_start - (self._max_contamination_length - 1) could overlap
        # as far as data_start if the contamination length is self._max_contamination_length.
        search_start = data_start - (self._max_contamination_length - 1)
        # We don't want to search any index < 0 if it doesn't make sense in your context,
        # but if it can be negative, we leave it as is. For safety, one could do:
        #   search_start = max(0, data_start - (self._max_contamination_length - 1))

        # Find relevant keys with bisect
        left_idx = bisect.bisect_left(self._sorted_keys, search_start)
        right_idx = bisect.bisect_right(self._sorted_keys, data_end)
        relevant_keys = self._sorted_keys[left_idx:right_idx]

        for key in relevant_keys:
            contamination_tensor = self.contamination_dict[key]
            contamination_len = contamination_tensor.shape[0]

            # local_offset is where the contamination would start inside `data`
            # (can be negative if contamination starts before data_start)
            local_offset = key - global_start_idx

            # The contamination covers [local_offset, local_offset + contamination_len) in data
            # We'll compute the portion that actually overlaps with [0, data_len)
            overlap_start_data = max(0, local_offset)
            overlap_end_data   = min(data_len, local_offset + contamination_len)
            overlap_len = overlap_end_data - overlap_start_data

            if overlap_len > 0:
                # The slice in the contamination tensor that overlaps with data
                # starts at contamination_offset_start = overlap_start_data - local_offset
                contamination_offset_start = overlap_start_data - local_offset
                data[overlap_start_data:overlap_end_data] = contamination_tensor[
                    contamination_offset_start : contamination_offset_start + overlap_len
                ]

        return data


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