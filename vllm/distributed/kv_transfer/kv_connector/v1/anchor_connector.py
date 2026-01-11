# SPDX-License-Identifier: Apache-2.0
"""
Anchor Connector for Kimi-Linear KDA + MLA State Save/Restore

This connector enables semantic anchor-based context compression for
Kimi-Linear hybrid models (KDA + MLA layers).

Key features:
- Save/restore KDA recurrent state (compressed context)
- Save/restore MLA K,V cache
- Use semantic anchors as cache keys instead of full token hashes

Usage:
    vllm serve model --kv-connector anchor --kv-connector-config '{"storage_path": "/tmp/anchors"}'
"""

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import safetensors
import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class AnchorMeta:
    """Metadata for a single anchor save/load operation."""
    anchor_id: str  # Semantic anchor like "<alice-software-tokyo/>"
    slot_mapping: torch.Tensor
    is_store: bool
    num_tokens: int


@dataclass
class AnchorConnectorMetadata(KVConnectorMetadata):
    """Connector metadata passed from scheduler to worker."""
    anchors: list[AnchorMeta] = field(default_factory=list)

    def add_anchor(
        self,
        anchor_id: str,
        block_ids: list[int],
        block_size: int,
        num_tokens: int,
        is_store: bool,
    ) -> None:
        # Create slot mapping from block IDs
        block_ids_tensor = torch.tensor(block_ids)
        num_blocks = block_ids_tensor.shape[0]
        block_offsets = torch.arange(0, block_size)
        slot_mapping = (
            block_offsets.reshape((1, block_size))
            + block_ids_tensor.reshape((num_blocks, 1)) * block_size
        )
        slot_mapping = slot_mapping.flatten()[:num_tokens]

        self.anchors.append(AnchorMeta(
            anchor_id=anchor_id,
            slot_mapping=slot_mapping,
            is_store=is_store,
            num_tokens=num_tokens,
        ))


class AnchorConnector(KVConnectorBase_V1):
    """
    Anchor-based KV connector for Kimi-Linear models.

    Saves and restores both:
    - KDA recurrent state (linear attention compressed context)
    - MLA K,V cache (standard attention)

    Uses semantic anchors as cache keys for efficient lookup.
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: Optional["KVCacheConfig"] = None,
    ):
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )
        self._block_size = vllm_config.cache_config.block_size
        self._requests_need_load: dict[str, "Request"] = {}
        self._anchor_registry: dict[str, str] = {}  # request_id -> anchor_id

        # Get storage path from config
        self._storage_path = self._kv_transfer_config.get_from_extra_config(
            "storage_path", "/tmp/anchor_cache"
        )
        os.makedirs(self._storage_path, exist_ok=True)
        logger.info(f"AnchorConnector initialized with storage: {self._storage_path}")

    def register_anchor(self, request_id: str, anchor_id: str) -> None:
        """Register a semantic anchor for a request."""
        self._anchor_registry[request_id] = anchor_id
        logger.info(f"Registered anchor {anchor_id} for request {request_id}")

    def _get_anchor_path(self, anchor_id: str) -> str:
        """Get storage path for an anchor."""
        # Sanitize anchor_id for filesystem
        safe_id = anchor_id.replace("<", "").replace(">", "").replace("/", "_")
        return os.path.join(self._storage_path, safe_id)

    def _get_layer_file(self, anchor_id: str, layer_name: str) -> str:
        """Get file path for a layer's cache."""
        anchor_path = self._get_anchor_path(anchor_id)
        os.makedirs(anchor_path, exist_ok=True)
        safe_layer = layer_name.replace(".", "_")
        return os.path.join(anchor_path, f"{safe_layer}.safetensors")

    def _is_kda_layer(self, layer) -> bool:
        """Check if layer is KDA (has recurrent state) vs MLA."""
        # KDA layers have kv_cache as tuple of 4 tensors
        # MLA layers have kv_cache as single tensor
        kv_cache = getattr(layer, "kv_cache", None)
        if kv_cache is None:
            return False
        if isinstance(kv_cache, (list, tuple)) and len(kv_cache) > 0:
            first_cache = kv_cache[0]
            # KDA: (conv_state_q, conv_state_k, conv_state_v, recurrent_state)
            return isinstance(first_cache, (list, tuple)) and len(first_cache) == 4
        return False

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        """Load KV cache from anchor storage."""
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, AnchorConnectorMetadata):
            return

        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            return

        for anchor_meta in metadata.anchors:
            if anchor_meta.is_store:
                continue

            anchor_id = anchor_meta.anchor_id
            logger.info(f"Loading anchor {anchor_id} ({anchor_meta.num_tokens} tokens)")

            for layer_name in forward_context.no_compile_layers:
                layer = forward_context.no_compile_layers[layer_name]

                kv_cache_attr = getattr(layer, "kv_cache", None)
                if kv_cache_attr is None:
                    continue

                file_path = self._get_layer_file(anchor_id, layer_name)
                if not os.path.exists(file_path):
                    logger.warning(f"Anchor file not found: {file_path}")
                    continue

                saved = safetensors.torch.load_file(file_path)

                if self._is_kda_layer(layer):
                    self._load_kda_state(layer, saved, forward_context)
                else:
                    self._load_mla_cache(
                        layer, saved, anchor_meta.slot_mapping, forward_context
                    )

    def _load_kda_state(
        self,
        layer,
        saved: dict[str, torch.Tensor],
        forward_context: "ForwardContext"
    ) -> None:
        """Load KDA recurrent state."""
        kv_cache = layer.kv_cache[forward_context.virtual_engine]
        conv_state_q, conv_state_k, conv_state_v, recurrent_state = kv_cache

        if "recurrent_state" in saved:
            recurrent_state.copy_(saved["recurrent_state"].cuda())
        if "conv_state_q" in saved:
            conv_state_q.copy_(saved["conv_state_q"].cuda())
        if "conv_state_k" in saved:
            conv_state_k.copy_(saved["conv_state_k"].cuda())
        if "conv_state_v" in saved:
            conv_state_v.copy_(saved["conv_state_v"].cuda())

    def _load_mla_cache(
        self,
        layer,
        saved: dict[str, torch.Tensor],
        slot_mapping: torch.Tensor,
        forward_context: "ForwardContext"
    ) -> None:
        """Load MLA K,V cache."""
        kv_cache = layer.kv_cache[forward_context.virtual_engine]

        if "kv_cache" in saved:
            src_cache = saved["kv_cache"].cuda()
            # Inject into paged memory
            kv_shape = kv_cache.shape
            if len(kv_shape) == 4:  # [2, num_pages, page_size, head_dim]
                num_pages, page_size = kv_shape[1], kv_shape[2]
                flat_cache = kv_cache.reshape(2, num_pages * page_size, -1)
                flat_cache[:, slot_mapping.cuda(), ...] = src_cache
            else:  # MLA: [num_pages, page_size, latent_dim]
                num_pages, page_size = kv_shape[0], kv_shape[1]
                flat_cache = kv_cache.reshape(num_pages * page_size, -1)
                flat_cache[slot_mapping.cuda(), ...] = src_cache

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Wait for layer load (sync implementation)."""
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: Any,  # AttentionMetadata - using Any for compatibility
        **kwargs: Any,
    ) -> None:
        """Save KV cache to anchor storage."""
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, AnchorConnectorMetadata):
            return

        for anchor_meta in metadata.anchors:
            if not anchor_meta.is_store:
                continue

            anchor_id = anchor_meta.anchor_id
            file_path = self._get_layer_file(anchor_id, layer_name)

            # Get the layer from forward context
            from vllm.forward_context import get_forward_context
            forward_context = get_forward_context()
            layer = forward_context.no_compile_layers.get(layer_name)

            if layer is None:
                # Direct save of kv_layer
                tensors = self._extract_mla_cache(kv_layer, anchor_meta.slot_mapping)
            elif self._is_kda_layer(layer):
                tensors = self._extract_kda_state(layer, forward_context)
            else:
                tensors = self._extract_mla_cache(kv_layer, anchor_meta.slot_mapping)

            safetensors.torch.save_file(tensors, file_path)
            logger.debug(f"Saved {layer_name} to {file_path}")

    def _extract_kda_state(
        self,
        layer,
        forward_context: "ForwardContext"
    ) -> dict[str, torch.Tensor]:
        """Extract KDA state tensors for saving."""
        kv_cache = layer.kv_cache[forward_context.virtual_engine]
        conv_state_q, conv_state_k, conv_state_v, recurrent_state = kv_cache

        return {
            "recurrent_state": recurrent_state.detach().cpu(),
            "conv_state_q": conv_state_q.detach().cpu(),
            "conv_state_k": conv_state_k.detach().cpu(),
            "conv_state_v": conv_state_v.detach().cpu(),
        }

    def _extract_mla_cache(
        self,
        kv_layer: torch.Tensor,
        slot_mapping: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Extract MLA K,V cache for saving."""
        kv_shape = kv_layer.shape

        if len(kv_shape) == 4:  # [2, num_pages, page_size, head_dim]
            num_pages, page_size = kv_shape[1], kv_shape[2]
            flat_cache = kv_layer.reshape(2, num_pages * page_size, -1)
            extracted = flat_cache[:, slot_mapping, ...].detach().cpu()
        else:  # MLA: [num_pages, page_size, latent_dim]
            num_pages, page_size = kv_shape[0], kv_shape[1]
            flat_cache = kv_layer.reshape(num_pages * page_size, -1)
            extracted = flat_cache[slot_mapping, ...].detach().cpu()

        return {"kv_cache": extracted}

    def wait_for_save(self) -> None:
        """Wait for saves to complete (sync implementation)."""
        return

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        """Check if anchor exists for request."""
        request_id = request.request_id

        # Check if we have an anchor registered for this request
        if request_id not in self._anchor_registry:
            return 0, False

        anchor_id = self._anchor_registry[request_id]
        anchor_path = self._get_anchor_path(anchor_id)

        if not os.path.exists(anchor_path):
            return 0, False

        # TODO: Determine how many tokens the anchor covers
        # For now, return 0 (no cached tokens) until we implement anchor metadata
        logger.info(f"Anchor hit for {anchor_id}")
        return 0, False

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int
    ) -> None:
        """Update state after block allocation."""
        if num_external_tokens > 0:
            self._requests_need_load[request.request_id] = request

    def build_connector_meta(
        self,
        scheduler_output: Any  # SchedulerOutput - using Any for compatibility
    ) -> KVConnectorMetadata:
        """Build connector metadata for this step."""
        meta = AnchorConnectorMetadata()

        for new_req in scheduler_output.scheduled_new_reqs:
            request_id = new_req.req_id

            if request_id in self._anchor_registry:
                anchor_id = self._anchor_registry[request_id]

                if request_id in self._requests_need_load:
                    # Load from anchor
                    meta.add_anchor(
                        anchor_id=anchor_id,
                        block_ids=new_req.block_ids[0],
                        block_size=self._block_size,
                        num_tokens=len(new_req.prompt_token_ids or []),
                        is_store=False,
                    )
                else:
                    # Store to anchor
                    meta.add_anchor(
                        anchor_id=anchor_id,
                        block_ids=new_req.block_ids[0],
                        block_size=self._block_size,
                        num_tokens=len(new_req.prompt_token_ids or []),
                        is_store=True,
                    )

        self._requests_need_load.clear()
        return meta

    def anchor_exists(self, anchor_id: str) -> bool:
        """Check if an anchor exists in storage."""
        anchor_path = self._get_anchor_path(anchor_id)
        return os.path.exists(anchor_path)

    def list_anchors(self) -> list[str]:
        """List all stored anchors."""
        if not os.path.exists(self._storage_path):
            return []
        return os.listdir(self._storage_path)

    def delete_anchor(self, anchor_id: str) -> bool:
        """Delete an anchor from storage."""
        import shutil
        anchor_path = self._get_anchor_path(anchor_id)
        if os.path.exists(anchor_path):
            shutil.rmtree(anchor_path)
            return True
        return False
