from typing import Any, Dict, List, Tuple
from deepspeed.inference.v2.inference_utils import PrefixCacheStrategy
from deepspeed.inference.v2.ragged.manager_configs import KVCacheConfig
from deepspeed.inference.v2.ragged.ragged_manager import DSStateManager
from deepspeed.inference.v2.ragged.ragged_wrapper import RaggedBatchWrapper
from deepspeed.inference.v2.ragged.sequence_descriptor import DSSequenceDescriptor
import torch


class PrefixCacheManager:
    def __init__(
        self,
        cache_strategy: PrefixCacheStrategy,
        state_manager: DSStateManager,
    ):
        self.cache_strategy = cache_strategy
        self.state_manager = state_manager

        # session_id => cache
        self.cahce_map: Dict[str, List[List[torch.Tensor]]] = {}

    # TODO: make it asynchronous
    def offload_kv_cache(
        self,
        layer_idx: int,
        kv_cache: torch.Tensor,
        ragged_batch_info: RaggedBatchWrapper,
        kv_cache_configs: Tuple[KVCacheConfig, ...]
    ):
        kv_cache_config = kv_cache_configs[0]
        num_layers = kv_cache_config.cache_shape[0]
        num_heads = kv_cache_config.cache_shape[1]
        head_size = kv_cache_config.cache_shape[2]

        kv_block_ids = ragged_batch_info.kv_block_ids()
        inflight_seq_descriptors = ragged_batch_info.inflight_seq_descriptors(on_device=False)
        session_ids = ragged_batch_info.session_ids()

        for block_ids, inflight_seq_descriptor, session_id in zip(kv_block_ids, inflight_seq_descriptors, session_ids):
            token_start, inflight_tokens, seen_tokens, _ = inflight_seq_descriptor
            for i in range(inflight_tokens):
                token_idx = seen_tokens + i
                block_idx = block_ids[token_idx // kv_cache_config.block_size]
                block_offset = token_idx % kv_cache_config.block_size
                # print(f"{block_idx=} {block_offset=} {kv_cache.shape=} {kv_cache[block_idx, block_offset].shape=}")
                host_kv = torch.zeros(
                    (2, num_heads, head_size),
                    dtype=kv_cache.dtype,
                    device="cpu",
                ).copy_(kv_cache[block_idx, block_offset], non_blocking=True)

                if session_id not in self.cahce_map:
                    self.cahce_map[session_id] = []
                    for _ in range(num_layers):
                        self.cahce_map[session_id].append([])
                self.cahce_map[session_id][layer_idx].append(host_kv)
    
    def load_kv_cache(
        self,
        num_layers: int,
        host_seq_desc: DSSequenceDescriptor,
    ) -> int:
        session_id = host_seq_desc.session_id

        # no prefix cache to load
        if session_id not in self.cahce_map:
            return 0
        
        assert num_layers == len(self.cahce_map[session_id]), f"num_layers {num_layers} != {len(self.cahce_map[session_id])}"

        block_size = self.state_manager._kv_configs[0].block_size
        prefix_tokens = len(self.cahce_map[session_id][0])
        block_ids = host_seq_desc.all_block_ids()

        assert all(len(prefix_cache) == prefix_tokens for prefix_cache in self.cahce_map[session_id]), \
            f"prefix cache size mismatch, {prefix_tokens=}"

        for layer_idx in range(num_layers):
            # print(f"load kv cache for layer {layer_idx} of session {session_id}")
            kv_cache = self.state_manager.get_cache(layer_idx)
            prefix_cache = self.cahce_map[session_id][layer_idx]

            for token_idx, kv in enumerate(prefix_cache):
                block_idx = block_ids[token_idx // block_size]
                block_offset = token_idx % block_size
                kv_cache[block_idx, block_offset].copy_(kv, non_blocking=True)

        host_seq_desc._seen_tokens = prefix_tokens

        return prefix_tokens