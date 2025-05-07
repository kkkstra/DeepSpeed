from typing import Any, Dict, List, Tuple
from deepspeed.accelerator.real_accelerator import get_accelerator
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
        self.cache_type_map: Dict[str, List[PrefixCacheStrategy]] = {}

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
            if session_id not in self.cahce_map:
                self.cahce_map[session_id] = []
                for _ in range(num_layers):
                    self.cahce_map[session_id].append([])
                self.cache_type_map[session_id] = [PrefixCacheStrategy.RECOMP] * num_layers
            self.cache_type_map[session_id][layer_idx] = PrefixCacheStrategy.KV_OFFLOAD

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

                self.cahce_map[session_id][layer_idx].append(host_kv)
    
    def offload_h_cache(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        ragged_batch_info: RaggedBatchWrapper,
        kv_cache_configs: Tuple[KVCacheConfig, ...]
    ):
        kv_cache_config = kv_cache_configs[0]
        num_layers = kv_cache_config.cache_shape[0]
        num_heads = kv_cache_config.cache_shape[1]
        head_size = kv_cache_config.cache_shape[2]

        inflight_seq_descriptors = ragged_batch_info.inflight_seq_descriptors(on_device=False)
        session_ids = ragged_batch_info.session_ids()

        for inflight_seq_descriptor, session_id in zip(inflight_seq_descriptors, session_ids):
            if session_id not in self.cahce_map:
                self.cahce_map[session_id] = []
                for _ in range(num_layers):
                    self.cahce_map[session_id].append([])
                self.cache_type_map[session_id] = [PrefixCacheStrategy.RECOMP] * num_layers
            self.cache_type_map[session_id][layer_idx] = PrefixCacheStrategy.H_CACHE

            token_start, inflight_tokens, seen_tokens, _ = inflight_seq_descriptor

            for i in range(inflight_tokens):
                host_hidden_states = torch.zeros(
                    (num_heads * head_size),
                    dtype=hidden_states.dtype,
                    device="cpu",
                ).copy_(hidden_states[i], non_blocking=True)

                self.cahce_map[session_id][layer_idx].append(host_hidden_states)

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
    
    def load_h_cache(
        self,
        model,
        tokens: torch.Tensor,
        host_seq_desc: DSSequenceDescriptor,
    ) -> int:
        from deepspeed.inference.v2.model_implementations.inference_model_base import DSInferenceModelBase
        model: DSInferenceModelBase = model

        session_id = host_seq_desc.session_id

        # no prefix cache to load
        if session_id not in self.cahce_map:
            return 0
        
        kv_cache_config = self.state_manager._kv_configs[0]
        num_layers = kv_cache_config.cache_shape[0]
        num_heads = kv_cache_config.cache_shape[1]
        head_size = kv_cache_config.cache_shape[2]

        num_tokens = len(self.cahce_map[session_id][0])
        host_seq_desc.pre_forward(num_tokens)

        batch = RaggedBatchWrapper(self.state_manager._config)
        batch.insert_sequence(host_seq_desc, tokens[:num_tokens])
        batch.finalize()

        for layer_idx in range(num_layers):
            assert num_tokens == len(self.cahce_map[session_id][layer_idx]), \
                f"num_tokens {num_tokens} != {len(self.cahce_map[session_id][layer_idx])}"
            host_hidden_states = self.cahce_map[session_id][layer_idx]
            kv_cache = self.state_manager.get_cache(layer_idx)

            hidden_states = torch.zeros(
                (num_tokens, num_heads * head_size),
                dtype=host_hidden_states[0].dtype,
                device=get_accelerator().current_device(),
            )

            # step 1: copy all hidden states cache to device
            hidden_states.copy_(torch.stack(host_hidden_states), non_blocking=True)

            # step 2: qkv projection
            qkv_w = model._transformer[layer_idx].qkv_w
            hidden_states = model.qkv(hidden_states, qkv_w, b=None)

            # step 3: copy to kv cache
            model.kv_copy(kv_cache, hidden_states, batch)
        
        host_seq_desc.post_forward()
        return num_tokens