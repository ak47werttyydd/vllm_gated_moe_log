import json
import os
import threading
from typing import Optional
from vllm import envs
import torch
#read req_id from thread local storage
from vllm.utils.moe_log_context import get_current_req_ids

def _env_bool(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).lower() in ("1", "true", "yes", "y", "on")


class MoETopKLogger:
    """
    Flag-gated MoE top-k routing logger.

    Enabled when VLLM_LOG_MOE is set to a file path.

    JSONL schema:
      - First line: {"type":"meta", ...}
      - Then: one {"type":"route", ...} per token (for the chosen layer)
    """

    def __init__(self) -> None:
        self.path: str = envs.VLLM_LOG_MOE
        self.enabled: bool = bool(self.path)

        # Select exactly one layer to log
        self.layer_idx: int = envs.LOG_MOE_LAYER_IDX

        # seed for reproduction of MoE choice
        self.seed: Optional[int] = envs.LOG_MOE_SEED

        self.void_req_id: str = "void_req"

        #default model
        self.default_model_id: str = "qwen2_moe"

        #default vllm version
        self.default_vllm_version: str = "v0.10.2"

        #defualt torch version
        self.default_torch_version: str = torch.__version__ if torch.__version__ is not None else "v2.8.0"

        self._lock = threading.Lock()
        self._fh = None
        self._meta_written = False

    def enabled_for(
        self,
        layer_idx: Optional[int],
    ) -> bool:
        if not self.enabled:
            return False
        # only log one indicated layer by layer_idx
        if self.layer_idx is not None and layer_idx is not None:
            return int(layer_idx) == int(self.layer_idx)
        #enbale log, but no designated layer, don't log.
        return False

    def _ensure_open_and_meta(
        self,
        *,
        layer_idx: int,
        top_k: int,
        device: str
    ) -> None:
        with self._lock:
            if self._fh is None:
                os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
                self._fh = open(self.path, "a", buffering=1)

            if not self._meta_written:
                meta = {
                    "type": "meta",
                    "model_id": self.default_model_id,
                    "vllm_version": self.default_vllm_version,
                    "torch_version": self.default_torch_version,
                    "device": device,
                    "seed": (self.seed if self.seed is not None else None),
                    "layers_logged": [int(layer_idx)],
                    "top_k": top_k,
                }
                # Remove Nones to keep the header clean (optional)
                meta = {k: v for k, v in meta.items() if v is not None and v != ""}
                self._fh.write(json.dumps(meta, separators=(",", ":")) + "\n")
                self._meta_written = True

    def log_topk(
        self,
        *,
        layer_idx: int,
        topk_ids: torch.Tensor,
        topk_weights: Optional[torch.Tensor],
        top_k: Optional[int] = None
    ) -> None:
        """
        Writes:
          - meta (once)
          - route lines: {"type":"route","req_id":...,"token_idx":...,"layer":...,"topk_ids":[...],"topk_weights":[...]}
        """
        if not self.enabled:
            return

        # Decide top_k for meta
        if top_k is None:
            # Try infer from tensor shape
            top_k = int(topk_ids.size(1))

        
        device = str(topk_ids.device)

        # Ensure meta line exists
        self._ensure_open_and_meta(
            layer_idx = layer_idx,
            top_k = top_k,
            device = device
        )

        # Snapshot to CPU (only when enabled + layer matched by caller)
        ids_cpu = topk_ids.detach().to("cpu")
        w_cpu = None
        if topk_weights is not None:
            w_cpu = topk_weights.detach().to("cpu")

        # Number of tokens
        M = int(ids_cpu.size(0))

        #read req_id from thread local storage
        req_ids = get_current_req_ids()

        with self._lock:
            for t in range(M):
                
                # get req_id for this token, use void_req if not available
                rid = (
                    req_ids[t]
                    if req_ids is not None and t < len(req_ids)
                    else self.void_req_id
                )
                rec = {
                    "type": "route",
                    "req_id": rid,
                    "token_idx": int(t),
                    "layer": layer_idx,
                    "topk_ids": ids_cpu[t].tolist(),
                }
                if w_cpu is not None:
                    rec["topk_weights"] = w_cpu[t].tolist()
                self._fh.write(json.dumps(rec, separators=(",", ":")) + "\n")


MOE_TOPK_LOGGER = MoETopKLogger()
