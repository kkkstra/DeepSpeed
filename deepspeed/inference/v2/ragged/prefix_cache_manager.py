import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple
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
        alternate_cache_strategy: PrefixCacheStrategy,
        layer: int,
        state_manager: DSStateManager,
    ):
        self.cache_strategy = cache_strategy
        self.alternate_cache_strategy = alternate_cache_strategy
        self.h_cache_layer = layer
        self.num_layers = state_manager._kv_configs[0].cache_shape[0]

        self.chunk_size = state_manager._kv_configs[0].block_size

        print(f"cache strategy: {self.cache_strategy}, alternate cache strategy: {self.alternate_cache_strategy}, h_cache layer: {self.h_cache_layer}")

        self.state_manager = state_manager

        # session_id => cache
        self.cahce_map: Dict[str, List[List[torch.Tensor]]] = {}
        self.cache_type_map: Dict[str, List[PrefixCacheStrategy]] = {}
        self.cache_tokens: Dict[str, List[int]] = {}

    def select_cache_strategy(
        self,
        layer_idx: int,
    ):
        if self.cache_strategy == PrefixCacheStrategy.H_CACHE:
            if self.alternate_cache_strategy == PrefixCacheStrategy.KV_OFFLOAD:
                return PrefixCacheStrategy.KV_OFFLOAD if layer_idx >= self.h_cache_layer else PrefixCacheStrategy.H_CACHE
            elif self.alternate_cache_strategy == PrefixCacheStrategy.RECOMP:
                return PrefixCacheStrategy.RECOMP if layer_idx < self.num_layers - self.h_cache_layer else PrefixCacheStrategy.H_CACHE
            else:
                return PrefixCacheStrategy.H_CACHE
        
        return self.cache_strategy

    async def offload_kv_cache(
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
                self.cache_tokens[session_id] = [0] * num_layers
            self.cache_type_map[session_id][layer_idx] = PrefixCacheStrategy.KV_OFFLOAD

            token_start, inflight_tokens, seen_tokens, _ = inflight_seq_descriptor

            for i in range(inflight_tokens):
                token_idx = seen_tokens + i
                block_idx = block_ids[token_idx // kv_cache_config.block_size]
                block_offset = token_idx % kv_cache_config.block_size
                
                chunk_idx = self.cache_tokens[session_id][layer_idx] // self.chunk_size
                chunk_offset = self.cache_tokens[session_id][layer_idx] % self.chunk_size
                self.cache_tokens[session_id][layer_idx] += 1

                if chunk_offset == 0:
                    host_kv_chunk = torch.zeros(
                        (self.chunk_size, 2, num_heads, head_size),
                        dtype=kv_cache.dtype,
                        device="cpu",
                        pin_memory=True,
                    )
                    self.cahce_map[session_id][layer_idx].append(host_kv_chunk)

                host_kv = self.cahce_map[session_id][layer_idx][chunk_idx][chunk_offset]
                host_kv.copy_(kv_cache[block_idx, block_offset],
                            #   non_blocking=True,
                )
    
    def offload_h_cache(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        ragged_batch_info: RaggedBatchWrapper,
        kv_cache_configs: Tuple[KVCacheConfig, ...]
    ):
        # print(f"start offload h cache for layer {layer_idx}")
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
                self.cache_tokens[session_id] = [0] * num_layers
            self.cache_type_map[session_id][layer_idx] = PrefixCacheStrategy.H_CACHE

            token_start, inflight_tokens, seen_tokens, _ = inflight_seq_descriptor
            for i in range(inflight_tokens):

                chunk_idx = self.cache_tokens[session_id][layer_idx] // self.chunk_size
                chunk_offset = self.cache_tokens[session_id][layer_idx] % self.chunk_size
                self.cache_tokens[session_id][layer_idx] += 1

                if chunk_offset == 0:
                    host_h_chunk = torch.zeros(
                        (self.chunk_size, num_heads * head_size),
                        dtype=hidden_states.dtype,
                        device="cpu",
                        pin_memory=True,
                    )
                    self.cahce_map[session_id][layer_idx].append(host_h_chunk)

                host_h_cache = self.cahce_map[session_id][layer_idx][chunk_idx][chunk_offset]
                host_h_cache.copy_(hidden_states[token_start + i],
                                #    non_blocking=True,
                )
        
        # torch.cuda.synchronize()
        # print(f"finish offload h cache for layer {layer_idx}")

    def load_kv_cache(
        self,
        num_layers: int,
        host_seq_desc: DSSequenceDescriptor,
        tokens: torch.Tensor,
    ) -> int:
        session_id = host_seq_desc.session_id

        # no prefix cache to load
        if session_id not in self.cahce_map:
            return 0
        
        # assert num_layers == len(self.cahce_map[session_id]), f"num_layers {num_layers} != {len(self.cahce_map[session_id])}"

        prefix_tokens = min(self.cache_tokens[session_id][0] - host_seq_desc.seen_tokens, tokens.numel()-1)
        if prefix_tokens <= 0:
            return 0

        host_seq_desc.pre_forward(prefix_tokens)

        block_size = self.state_manager._kv_configs[0].block_size
        block_ids = host_seq_desc.all_block_ids()

        # assert all(len(prefix_cache) == prefix_tokens for prefix_cache in self.cahce_map[session_id]), \
        #     f"prefix cache size mismatch, {prefix_tokens=}"

        for layer_idx in range(num_layers):
            # print(f"load kv cache for layer {layer_idx} of session {session_id}")
            kv_cache = self.state_manager.get_cache(layer_idx)
            prefix_cache = self.cahce_map[session_id][layer_idx]

            for idx in range(prefix_tokens):
                token_idx = host_seq_desc.seen_tokens + idx
                block_idx = block_ids[token_idx // block_size]
                block_offset = token_idx % block_size

                chunk_idx = token_idx // self.chunk_size
                chunk_offset = token_idx % self.chunk_size

                kv = prefix_cache[chunk_idx][chunk_offset]
                kv_cache[block_idx, block_offset].copy_(kv,
                                          non_blocking=True,
                )

        host_seq_desc.post_forward()

        # torch.cuda.synchronize()
        return prefix_tokens

    async def recompute_h_cache_layer(
        self,
        layer_idx: int,
        model,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        batch: RaggedBatchWrapper,
    ):
        from deepspeed.inference.v2.model_implementations.inference_model_base import DSInferenceModelBase
        model: DSInferenceModelBase = model

        # qkv projection
        qkv_w = model._transformer[layer_idx].qkv_w
        hidden_states = model.qkv(hidden_states, qkv_w, b=None)

        # copy to kv cache
        model.kv_copy(kv_cache, hidden_states, batch)

    async def recompute_layer(
        self,
        layers: int,
        model,
        batch: RaggedBatchWrapper,
    ):
        from deepspeed.inference.v2.model_implementations.inference_model_base import DSInferenceModelBase
        model: DSInferenceModelBase = model

        residual = model._forward_embed(batch)

        residual, hidden_states = model.norm(residual, None, model._transformer[0].attn_norm_gamma, beta=None)

        for layer_idx in range(layers):
            residual, hidden_states = model._forward_transformer_layer(layer_idx, residual, hidden_states,
                                                                       batch)

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

        total_num_tokens = min(self.cache_tokens[session_id][-1] - host_seq_desc.seen_tokens, tokens.numel()-1)
        if total_num_tokens <= 0:
            return 0

        block_size = self.state_manager._kv_configs[0].block_size
        block_ids = host_seq_desc.all_block_ids()

        load_tokens = 0

        recompute_tasks = []
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        while load_tokens < total_num_tokens:
            num_tokens = min(total_num_tokens - load_tokens, 512)

            host_seq_desc.pre_forward(num_tokens)

            batch = RaggedBatchWrapper(self.state_manager._config)
            batch.insert_sequence(host_seq_desc, tokens[load_tokens:load_tokens + num_tokens])
            batch.finalize()


            # start_time = time.perf_counter()
            for layer_idx in range(num_layers):
                cache_type = self.cache_type_map[session_id][layer_idx]
                # assert cache_type == PrefixCacheStrategy.RECOMP or num_tokens == len(self.cahce_map[session_id][layer_idx]), \
                #     f"{layer_idx=}, {cache_type=}, num_tokens {num_tokens} != {len(self.cahce_map[session_id][layer_idx])}"
                
                if cache_type == PrefixCacheStrategy.RECOMP:
                    # print(f"recompute layer {layer_idx} of session {session_id}")
                    if layer_idx == 0:
                        recompute_tasks.append(
                            loop.create_task(
                                self.recompute_layer(num_layers - self.h_cache_layer, model, batch)
                            )
                        )
                
                elif cache_type == PrefixCacheStrategy.KV_OFFLOAD:
                    # print(f"kv load layer {layer_idx} of session {session_id}")
                    kv_cache = self.state_manager.get_cache(layer_idx)
                    prefix_cache = self.cahce_map[session_id][layer_idx]

                    for idx in range(num_tokens):
                        token_idx = host_seq_desc.seen_tokens + idx
                        block_idx = block_ids[token_idx // block_size]
                        block_offset = token_idx % block_size

                        chunk_idx = token_idx // self.chunk_size
                        chunk_offset = token_idx % self.chunk_size

                        kv = prefix_cache[chunk_idx][chunk_offset]
                        kv_cache[block_idx, block_offset].copy_(kv,
                                                                non_blocking=True,
                        )
                    
                elif cache_type == PrefixCacheStrategy.H_CACHE:
                    # print(f"load hcache layer {layer_idx} of session {session_id}")
                    host_hidden_states = self.cahce_map[session_id][layer_idx]
                    kv_cache = self.state_manager.get_cache(layer_idx)

                    chunk_idx_start = host_seq_desc.seen_tokens // self.chunk_size
                    chunk_start_offset = host_seq_desc.seen_tokens % self.chunk_size
                    chunk_idx_end = (host_seq_desc.seen_tokens + num_tokens) // self.chunk_size
                    chunk_length = chunk_idx_end - chunk_idx_start 

                    hidden_states = torch.zeros(
                        (num_tokens, num_heads * head_size),
                        dtype=host_hidden_states[0].dtype,
                        device=get_accelerator().current_device(),
                    )

                    # copy all hidden states cache to device
                    if chunk_length == 0:
                        hidden_states.copy_(
                                    host_hidden_states[chunk_idx_start][chunk_start_offset:chunk_start_offset + num_tokens],
                                    non_blocking=True,
                                )
                    else:
                        for chunk_idx in range(chunk_length):
                            # print(f"{hidden_states[chunk_idx * self.chunk_size: (chunk_idx + 1) * self.chunk_size].shape=}, {host_hidden_states[chunk_idx].shape=}")
                            if chunk_idx == 0:
                                hidden_states[chunk_idx * self.chunk_size: (chunk_idx + 1) * self.chunk_size - chunk_start_offset].copy_(
                                    host_hidden_states[chunk_idx_start + chunk_idx][chunk_start_offset:],
                                    non_blocking=True,
                                )
                            else:
                                hidden_states[chunk_idx * self.chunk_size - chunk_start_offset: (chunk_idx + 1) * self.chunk_size - chunk_start_offset].copy_(
                                    host_hidden_states[chunk_idx_start + chunk_idx],
                                    non_blocking=True,
                                )
                        if chunk_length * self.chunk_size - chunk_start_offset < num_tokens:
                            # print(f"{chunk_length=}, {self.chunk_size=}, {host_seq_desc.seen_tokens=}, {num_tokens=}, {chunk_start_offset=}, {chunk_idx_start=}, {chunk_idx_end=}")
                            hidden_states[chunk_length * self.chunk_size - chunk_start_offset: num_tokens].copy_(
                                host_hidden_states[chunk_idx_end][0: num_tokens - chunk_length * self.chunk_size + chunk_start_offset],
                                non_blocking=True,
                            )

                    # recompute hidden states
                    recompute_tasks.append(
                        loop.create_task(self.recompute_h_cache_layer(layer_idx, model, hidden_states, kv_cache, batch))
                    )
            # end_time = time.perf_counter()
            # print(f"load {num_tokens} tokens hidden states took {end_time - start_time:.4f} seconds")

            # torch.cuda.synchronize()
            
            host_seq_desc.post_forward()
            load_tokens += num_tokens
        try:
            loop.run_until_complete(asyncio.gather(*recompute_tasks))
        finally:
            loop.close()

        return total_num_tokens
    
    def recompute_kv_cache(
        self,
        model,
        tokens: torch.Tensor,
        host_seq_desc: DSSequenceDescriptor,
        prefix_length: int,
    ) -> int:
        from deepspeed.inference.v2.model_implementations.inference_model_base import DSInferenceModelBase
        model: DSInferenceModelBase = model

        total_num_tokens = min(prefix_length - host_seq_desc.seen_tokens, tokens.numel()-1)
        if total_num_tokens <= 0:
            return 0

        batch = RaggedBatchWrapper(self.state_manager._config)

        load_tokens = 0
        while load_tokens < total_num_tokens:
            num_tokens = min(total_num_tokens - load_tokens, 512)

            host_seq_desc.pre_forward(num_tokens)

            batch.clear()
            batch.insert_sequence(host_seq_desc, tokens[load_tokens:load_tokens + num_tokens])
            batch.finalize()

            model.prepare_batch(batch)
            model.forward(batch)
            
            host_seq_desc.post_forward()
            load_tokens += num_tokens

        return total_num_tokens