# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import json
import pickle
import time
from typing import Iterable, Tuple

from deepspeed.inference.v2.inference_utils import PrefixCacheStrategy
from deepspeed.inference.v2.ragged.prefix_cache_manager import PrefixCacheManager
import torch

import deepspeed.comm as dist

from deepspeed.accelerator import get_accelerator
from deepspeed.comm.comm import init_distributed

from .model_implementations import InferenceV2Policy
from .logging import inference_logger
from .ragged import DSStateManager, RaggedBatchWrapper, PlaceholderSequenceDescriptor
from .scheduling_utils import SchedulingError, SchedulingResult
from .model_implementations.flat_model_helpers import make_param_filename, make_metadata_filename
from .model_implementations.inference_model_base import DSInferenceModelBase

from .config_v2 import RaggedInferenceEngineConfig

INFERENCE_MODEL_TIMER = "model-forward-inference"


class InferenceEngineV2:

    _config: RaggedInferenceEngineConfig
    """
    Configuration of the inference engine.
    """

    _model: DSInferenceModelBase
    """
    Inference model supporting ragged inference.
    """

    _state_manager: DSStateManager
    """
    Persistent state manager for sequences and KV-cache.
    """

    _prefix_cache_manager: PrefixCacheManager
    """
    Prefix cache manager for managing prefix cache.
    """

    @property
    def free_blocks(self) -> torch.Tensor:
        """
        Number of free KV blocks. This is a tensor of shape [n_kv_cache_groups] where each
        element is the number of free blocks in the corresponding KV cache group.
        """
        return self._state_manager.free_blocks

    @property
    def n_kv_cache_groups(self) -> int:
        """
        Number of KV cache groups.
        """
        return self._state_manager.n_kv_cache_groups

    def model(self) -> DSInferenceModelBase:
        """
        The model implementation.
        """
        return self._model

    def __init__(self, policy: InferenceV2Policy, engine_config: RaggedInferenceEngineConfig) -> None:
        """
        Create the Inference V2 engine.

        Arguments:
            policy (InferenceV2Policy): Policy for the model implementation. This policy object
                will be used to build the model and load the checkpoint associated with it.
            engine_config (RaggedInferenceEngineConfig): Configuration for the inference engine.
        """
        self._config = engine_config
        self._policy = policy
        self._base_mp_group = self._initialize_tp_group()

        # Build model from policy
        inference_logger().info("Building model...")
        self._model = self._policy.build_model(self._config, self._base_mp_group)
        inference_logger().info("Model built.")

        # Create state manager
        self._batch = RaggedBatchWrapper(self._config.state_manager)
        self._state_manager = DSStateManager(self._config.state_manager,
                                             self._model.kv_cache_config(),
                                             base_mp_group=self._base_mp_group)
        self._model.set_state_manager(self._state_manager)

        # Create prefix cache manager
        self._prefix_cache_manager = PrefixCacheManager(
            self._config.state_manager.prefix_cache_strategy,
            self._config.state_manager.prefix_cache_strategy_alt,
            self._config.state_manager.h_cache_layer,
            self._state_manager,
        )
        self._model.set_prefix_cache_manager(self._prefix_cache_manager)

    def _initialize_tp_group(self):
        """
        Implementation of our TP group initialization.
        """
        init_distributed()
        local_rank = int(os.getenv("LOCAL_RANK", 0))
        get_accelerator().set_device(local_rank)

        if local_rank >= self._config.tensor_parallel.tp_size:
            raise RuntimeError("Local rank is greater than TP size, ensure that the TP config is correct.")

        ranks = list(range(self._config.tensor_parallel.tp_size))
        return dist.new_group(ranks=ranks)

    def put(self,
            batch_sids: Iterable[str],
            batch_uids: Iterable[int],
            batch_tokens: Iterable[torch.Tensor],
            prefix_lengths: Iterable[int],
            do_checks: bool = True) -> torch.Tensor:
        """
        Put a ragged batch onto the inference engine. This will perform one forward and return
        a Tensor of the shape [len(batch_uids), *output_shape]. Logits for the non-final tokens
        are not calculated.

        Arguments:
            batch_sids: Iterable of sids for the batch on the host
            batch_uids: Iterable of uids for the batch on the host
            batch_tokens: Iterable of token tensors for the batch on the host
            do_checks: Check schedulability when it is set to True. You can skip this check for better performance when it has already been completed.
        """

        if do_checks:
            token_lens = [len(tokens) for tokens in batch_tokens]
            schedule_check = self.can_schedule(batch_uids, token_lens)
            if schedule_check != SchedulingResult.Success:
                raise SchedulingError(schedule_check)

        self._batch.clear()
        for sid, uid, tokens, prefix_length in zip(batch_sids, batch_uids, batch_tokens, prefix_lengths):

            host_seq_desc = self._state_manager.get_or_create_sequence(uid, sid)
            self._model.maybe_allocate_kv(host_seq_desc, tokens.numel())

            # before start forward, process the prefix kv first
            load_tokens = 0
            if tokens.numel() > 1:
                start_time = time.perf_counter()
                if self._config.state_manager.prefix_cache_strategy == PrefixCacheStrategy.RECOMP:
                    load_tokens = self._prefix_cache_manager.recompute_kv_cache(self._model, tokens, host_seq_desc, prefix_length)
                elif self._config.state_manager.prefix_cache_strategy == PrefixCacheStrategy.KV_OFFLOAD:
                    load_tokens = self._prefix_cache_manager.load_kv_cache(self._model.num_layers, host_seq_desc, tokens)
                elif self._config.state_manager.prefix_cache_strategy == PrefixCacheStrategy.H_CACHE:
                    load_tokens = self._prefix_cache_manager.load_h_cache(self._model, tokens, host_seq_desc)
                end_time = time.perf_counter()

            if load_tokens > 0:
                print(f"load {load_tokens}/{tokens.numel()} prefix tokens for {sid=} {uid=}, cost {(end_time - start_time) * 1000:.2f} ms")
            
            host_seq_desc.pre_forward(tokens.numel() - load_tokens)

            # We can disable checks since we already validated schedulability.
            self._batch.insert_sequence(host_seq_desc, tokens[load_tokens:], do_checks=do_checks)

        # Send all metadata to the device
        self._batch.finalize()

        # ensure all prefix cache is loaded
        torch.cuda.synchronize()

        # Prep all data structures for the actual forward (in anticipation of CG in the future)
        # and also to amortize some of the costs in a more straightforward way.
        self._model.prepare_batch(self._batch)

        # Model implementation will pick up in the forward.
        logits = self._model.forward(self._batch)

        # We return one set of logits per sequence in the batch (saves cost on unembedding)
        assert logits.shape[0] == self._batch.current_sequences

        for uid in batch_uids:
            host_seq_desc = self._state_manager.get_sequence(uid)
            host_seq_desc.post_forward()  # Updates sequence metadata.
            self._model.maybe_free_kv(host_seq_desc)

        return logits

    def query(self, uid: int, max_request_tokens: int, max_request_blocks) -> Tuple[int, torch.Tensor]:
        """
        Determine the number of tokens and KV blocks to reserve for a given request. Given a UID
        (this UID may not be recognized by the model yet), this will return the number of tokens
        and blocks to reserve for the request.

        Arguments:
            uid (int): The UID of the sequence (as tracked by the scheduling entity). If
                this is a new sequence (with a UID unknown to the inference engine), then
                an empty placeholder is created to pass to the occupancy logic.
            n_tokens (int): The number of tokens to hypothetically send.

        Returns:
            Tuple[int, Optional[int]]: Tuple of free kv blocks and the number of blocks
                required to schedule the sequence.
        """
        seq_desc = self._state_manager.get_sequence(uid)
        if seq_desc is None:
            if (self._state_manager.n_tracked_sequences == self._config.state_manager.max_tracked_sequences):
                return (0, 0)
            seq_desc = PlaceholderSequenceDescriptor()

        req_tokens, req_blocks = self._model.get_kv_requirements(seq_desc, max_request_tokens, max_request_blocks)

        return (req_tokens, req_blocks)

    def can_schedule(self, uids: Iterable[int], lengths: Iterable[int]) -> SchedulingResult:
        """
        Dry run a batch to determine if it can be scheduled. Placeholder sequences will be
        created for any UIDs that are unknown to the inference engine.

        Arguments:
            uids (Iterable[int]): Iterable of UIDs for the batch
            lengths (Iterable[int]): Iterable of lengths for each sequence of the batch. This lengths
                corresponds to the number of tokens to send in the hypothetical forward; history
                tokens will be determined via UID lookup and future tokens are disregarded.

        Returns:
            bool: True if the batch can be scheduled, False otherwise.
        """

        cur_seqs = self._state_manager.n_tracked_sequences
        free_blocks = self._state_manager.free_blocks
        req_blocks = 0
        batch_len = 0

        if len(uids) > self._config.state_manager.max_ragged_sequence_count:
            # Can only compose a batch from a limited number of sequences
            return SchedulingResult.BatchSequenceLimitExceeded

        for uid, length in zip(uids, lengths):
            seq_desc = self._state_manager.get_sequence(uid)
            if seq_desc is None:
                cur_seqs += 1
                seq_desc = PlaceholderSequenceDescriptor()

            sched_len, sched_blocks = self._model.get_kv_requirements(seq_desc, length, free_blocks)

            if sched_len != length:
                # We ran out of KV cache
                return SchedulingResult.KVCacheLimitExceeded

            batch_len += length
            free_blocks -= sched_blocks

        if cur_seqs > self._config.state_manager.max_tracked_sequences:
            # Would run out of tracking metadata
            return SchedulingResult.EngineSequenceLimitExceeded

        if batch_len > self._config.state_manager.max_ragged_batch_size:
            # Would exceed the maximum batch size
            return SchedulingResult.BatchTokenLimitExceeded

        return SchedulingResult.Success

    def get_remaining_block_capacity(self, uid: int) -> int:
        """
        Get the remaining capacity of the last block already allocated.
        """
        seq_desc = self._state_manager.get_sequence(uid)
        if seq_desc is None:
            return 0
        return self._model.get_remaining_block_capacity(seq_desc)

    def flush(self, uid: int) -> None:
        """
        Remove all state associated with a sequence from the inference engine.

        Arguments:
            uid (int): The UID of the sequence to flush.
        """
        self._state_manager.flush_sequence(uid)

    def serialize(self, save_path: str) -> None:
        """
        Serialize the model to a file.

        Arguments:
            path (str): Path to the file to serialize to.
        """
        param_file_name = make_param_filename(save_path, self._model.tp_rank, self._model.tp_size)
        metadata_file_name = make_metadata_filename(save_path, self._model.tp_rank, self._model.tp_size)

        # Save the flattened parameters

        torch.save(self._model.flattened_params, param_file_name)

        json.dump(self._model.flattened_param_metadata.json(), open(metadata_file_name, "w"))

        if self._model.tp_rank == 0:
            pickle.dump(self._model._config, open(os.path.join(save_path, "ds_model_config.pkl"), "wb"))
