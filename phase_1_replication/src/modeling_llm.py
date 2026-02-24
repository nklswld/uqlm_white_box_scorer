"""
Model and tokenizer wrapper for deterministic question–answer encoding and forward passes.

Key inputs: HuggingFace model_name; question/answer strings for encoding; tokenized tensors for forward().
Key outputs: BatchEncodedQA (batched token IDs/masks + answer span indices) and ForwardOut (logits + optional hidden states).
Design contract: encode_qa_batch produces padding-agnostic answer_start_idx in padded space, suitable for left-padded batching.
Reproducibility: deterministic given fixed model weights/config and identical tokenizer settings; no stochastic sampling here.
"""

# phase_1_replication/src/modeling_llm.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class ForwardOut:
    """Model forward outputs: logits plus optional hidden_states (when requested)."""
    logits: torch.Tensor
    hidden_states: Optional[Tuple[torch.Tensor, ...]]


# -----------------------------
# Central contract
# -----------------------------
@dataclass(frozen=True)
class BatchEncodedQA:
    """
    Batched QA encoding contract used by all scorers (single source of truth).

    Exposes padded tensors plus answer span metadata in *padded* coordinates so downstream
    scorers can index without depending on padding side.
    """
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    answer_start_idx: torch.Tensor
    prompt_len: torch.Tensor
    seq_len: torch.Tensor
    first_real_idx: torch.Tensor
    answer_len: torch.Tensor


def _first_real_idx_from_attention_mask(attn: torch.Tensor) -> torch.Tensor:
    """
    Return index of the first non-pad token per row (attention_mask==1) in padded space.

    Works for both left- and right-padding; rows with no real tokens map to T (should not occur).
    """
    if attn.dim() != 2:
        raise ValueError(f"attention_mask must be 2D (B,T), got shape={tuple(attn.shape)}")
    B, T = attn.shape
    out = torch.full((B,), T, dtype=torch.long, device=attn.device)
    for b in range(B):
        nz = torch.nonzero(attn[b], as_tuple=False).flatten()
        if nz.numel() > 0:
            out[b] = nz[0]
    return out


class LLMWrapper:
    """
    Minimal HuggingFace CausalLM wrapper with a frozen prompt and encoding utilities.

    NOTE: potential issue: prompt format must exactly match the dataset's frozen-answer generation,
    otherwise answer span alignment and all derived scores will be inconsistent.
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        max_input_tokens: Optional[int] = None,
        *,
        # If set, enables device_map="auto" with max_memory for VRAM headroom.
        # For full-quality on large GPUs, keep this as None.
        max_cuda_mem: Optional[str] = None,
        # Enable only when you explicitly need to reduce activation memory.
        enable_gradient_checkpointing: bool = False,
    ) -> None:
        self.model_name = model_name

        requested = (device or "").strip().lower() if device else None
        use_cuda = torch.cuda.is_available() and requested not in {"cpu"}

        # Decide target device string for explicit placement (device_map="auto" handled separately below)
        if device is None:
            device = "cuda:0" if use_cuda else "cpu"
        self._requested_device = device

        # Default dtype (favor bf16 on supported CUDA; otherwise fp16; CPU defaults fp32)
        if dtype is None:
            if use_cuda and ("cuda" in device):
                try:
                    bf16_ok = torch.cuda.is_bf16_supported()
                except Exception:
                    bf16_ok = False
                dtype = torch.bfloat16 if bf16_ok else torch.float16
            else:
                dtype = torch.float32
        self.dtype = dtype
        self.max_input_tokens = max_input_tokens

        # Tokenizer (NO vocab modifications in Plan A)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        # Ensure a pad token exists for batched padding; prefer eos_token to avoid vocab mutation
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        # Convention: decoder-only LMs typically score more robustly with left-padding in batches
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"

        # Model load
        if use_cuda and ("cuda" in device):
            if max_cuda_mem is not None:
                # VRAM-headroom mode: allow HF to shard/offload; input_device inferred from hf_device_map
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=self.dtype,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    max_memory={"cuda:0": max_cuda_mem, "cpu": "999GiB"},
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=self.dtype,
                    low_cpu_mem_usage=True,
                ).to(device)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
            ).to(device)

        self.model.eval()
        # Disable KV caching to keep forward behavior consistent across call sites (esp. when returning hidden states)
        self.model.config.use_cache = False

        if enable_gradient_checkpointing:
            # Trade-off: lower activation memory at the cost of extra compute; keep explicit/opt-in
            self.model.gradient_checkpointing_enable()

        # Device where tokenized inputs must live (may differ from requested device under sharding/offload)
        self.input_device = self._infer_input_device(use_cuda=use_cuda)

    def _infer_input_device(self, *, use_cuda: bool) -> torch.device:
        # Prefer the model's device map if sharded/offloaded (inputs must be placed on a CUDA shard device)
        hf_map = getattr(self.model, "hf_device_map", None)
        if isinstance(hf_map, dict) and len(hf_map) > 0:
            for _, dev in hf_map.items():
                if isinstance(dev, str) and dev.startswith("cuda"):
                    return torch.device(dev)

        # Otherwise, mirror requested device
        if use_cuda and torch.cuda.is_available() and "cuda" in self._requested_device:
            return torch.device(self._requested_device)
        return torch.device("cpu")

    # -------------------------------------------------------------------------
    # Prompting / encoding (SINGLE SOURCE OF TRUTH)
    # -------------------------------------------------------------------------

    def build_prompt(self, question: str) -> str:
        """Return the frozen prompt template used for all encodings."""
        return f"Question: {question}\nAnswer:"

    def _tokenize_text(self, text: str) -> Dict[str, Any]:
        # Centralize tokenizer flags to avoid drift across callers (no special tokens; no truncation here)
        return self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            return_tensors="pt",
        )

    def encode_qa(self, question: str, answer: str):
        """Encode a single QA pair and return tensors plus answer span (unbatched, unpadded)."""
        tok = self.tokenizer

        prompt = self.build_prompt(question)
        full_text = prompt + answer

        enc_full = tok(
            full_text,
            add_special_tokens=False,
            return_tensors="pt",
        )
        enc_prompt = tok(
            prompt,
            add_special_tokens=False,
            return_tensors="pt",
        )

        input_ids = enc_full["input_ids"].to(self.input_device)
        attention_mask = enc_full["attention_mask"].to(self.input_device)

        prompt_len = enc_prompt["input_ids"].size(1)
        seq_len = attention_mask.sum().item()

        # Invariant: answer must contribute at least one real token beyond the prompt
        if prompt_len >= seq_len:
            raise RuntimeError(
                f"Prompt length >= sequence length in encode_qa "
                f"(prompt_len={prompt_len}, seq_len={seq_len})"
            )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "answer_start_idx": prompt_len,
            "answer_len": seq_len - prompt_len,
        }

    # -----------------------------
    # Tokenizer-based batching, padding-agnostic answer_start_idx in PADDED space.
    # -----------------------------
    def encode_qa_batch(self, questions, answers) -> BatchEncodedQA:
        """Batch-encode QA pairs, returning padded tensors plus padding-agnostic answer indices."""
        if len(questions) != len(answers):
            raise ValueError("encode_qa_batch: questions and answers must have same length")
        if len(questions) == 0:
            raise ValueError("encode_qa_batch: empty batch")

        tok = self.tokenizer

        prompts = [self.build_prompt(q) for q in questions]
        full_texts = [p + a for p, a in zip(prompts, answers)]

        max_len = self.max_input_tokens  # None => no truncation

        # 1) Full texts: padded for batching; optionally truncated from the left (truncation_side="left")
        enc_full = tok(
            full_texts,
            add_special_tokens=False,
            padding=True,
            truncation=(max_len is not None),
            max_length=max_len,
            return_tensors="pt",
        )

        input_ids = enc_full["input_ids"].to(self.input_device)
        attention_mask = enc_full["attention_mask"].to(self.input_device).to(torch.long)

        # seq_len counts real tokens only (padding-agnostic); first_real_idx locates the first real token in padded space
        seq_len = attention_mask.sum(dim=1).to(torch.long)  # (B,)
        first_real_idx = _first_real_idx_from_attention_mask(attention_mask)  # (B,)

        if max_len is None:
            # No truncation: prompt is a true prefix of the full sequence by construction
            enc_prompt = tok(
                prompts,
                add_special_tokens=False,
                padding=True,
                truncation=False,
                return_tensors="pt",
            )
            prompt_mask = enc_prompt["attention_mask"].to(self.input_device).to(torch.long)
            prompt_len = prompt_mask.sum(dim=1).to(torch.long)

        else:
            # Truncation: prompt may be clipped; recover the surviving prompt length via suffix(prompt)==prefix(full_real)
            # NOTE: potential issue: this relies on exact tokenization stability and prompt being contiguous at sequence start.
            prompt_ids_list = tok(
                prompts,
                add_special_tokens=False,
                padding=False,
                truncation=False,
            )["input_ids"]

            prompt_len_list = []
            for i in range(len(prompts)):
                # Slice the real (non-pad) portion; indices are in padded space derived from attention_mask
                fr = int(first_real_idx[i].item())
                sl = int(seq_len[i].item())
                full_real = input_ids[i, fr:fr + sl].tolist()

                pr = prompt_ids_list[i]
                m = min(len(pr), len(full_real))

                L_found = 0
                # Heuristic: longest suffix of (possibly truncated) prompt that matches the prefix of full_real
                for L in range(m, -1, -1):
                    if pr[-L:] == full_real[:L]:
                        L_found = L
                        break

                # NOTE: potential issue: repeated substrings in the prompt can make the suffix/prefix match non-unique (ambiguous prompt_len).
                prompt_len_list.append(L_found)

            prompt_len = torch.tensor(prompt_len_list, device=self.input_device, dtype=torch.long)

        # Span convention: answer starts immediately after the surviving prompt tokens (in real-token coordinates)
        answer_len = (seq_len - prompt_len).to(torch.long)
        answer_start_idx = (first_real_idx + prompt_len).to(torch.long)

        # Guardrails: detect degenerate spans early (otherwise downstream scorers may silently index pads/overflow)
        if bool((answer_len <= 0).any().item()):
            bad = (answer_len <= 0).nonzero(as_tuple=True)[0][:10].tolist()
            raise RuntimeError(f"encode_qa_batch: empty/invalid answer span for indices {bad}.")

        end_real = first_real_idx + seq_len
        if bool((answer_start_idx >= end_real).any().item()):
            bad = (answer_start_idx >= end_real).nonzero(as_tuple=True)[0][:10].tolist()
            raise RuntimeError(f"encode_qa_batch: answer_start_idx out of bounds for indices {bad}.")
        
        return BatchEncodedQA(
            input_ids=input_ids,
            attention_mask=attention_mask,
            answer_start_idx=answer_start_idx,
            prompt_len=prompt_len,
            seq_len=seq_len,
            first_real_idx=first_real_idx,
            answer_len=answer_len,
        )

    # -------------------------------------------------------------------------
    # Forward passes
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden_states: bool = False,
    ) -> ForwardOut:
        """Forward pass returning logits and (optionally) hidden states; gradients disabled."""
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=False,
            use_cache=False,
        )

        hidden_states = None
        if output_hidden_states:
            hidden_states = out.hidden_states

        return ForwardOut(logits=out.logits, hidden_states=hidden_states)

    def forward_embeds(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        output_hidden_states: bool = False,
    ):
        """
        Forward pass using inputs_embeds (enables gradient flow through embeddings).
        Intended for gradient-based scoring methods.
        """
        return self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            use_cache=False,
        )