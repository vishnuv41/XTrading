"""
phase15/model/mini_llm.py
-------------------------
Financial Domain-Specific Mini-LLM (35M Parameters).

A lightweight, specialized decoder-only Transformer built from scratch in PyTorch:
- Vocab Size: 2,048 (Domain-specific discretized financial tokens)
- Hidden Dimension (d_model): 512
- Transformer Blocks: 8
- Attention Heads: 8 (64 dim/head)
- FFN Intermediate: 2,048 (SwiGLU)
- Total Parameters: ~34.6 Million

Dual Heads:
1. Causal Next-Token Prediction Head (Self-Supervised Pretraining)
2. Trading Decision Head (Supervised Fine-Tuning: ACCEPT / REJECT)
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import math
import logging
from typing import Optional, Tuple, Dict, Any, Union

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch is not installed in the environment. PyTorch-based FinancialMiniLLM requires 'torch'.")

if HAS_TORCH:
    class CausalSelfAttention(nn.Module):
        def __init__(self, d_model: int = 512, n_heads: int = 8, max_seq_len: int = 512, dropout: float = 0.1):
            super().__init__()
            self.d_model = d_model
            self.n_heads = n_heads
            self.head_dim = d_model // n_heads
            
            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)
            
            self.attn_dropout = nn.Dropout(dropout)
            self.resid_dropout = nn.Dropout(dropout)
            
            mask = torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len)
            self.register_buffer("causal_mask", mask)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            B, T, C = x.shape
            q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores = scores.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            
            out = (attn_weights @ v).transpose(1, 2).contiguous().view(B, T, C)
            return self.resid_dropout(self.out_proj(out))

    class SwiGLUFFN(nn.Module):
        def __init__(self, d_model: int = 512, intermediate_size: int = 2048, dropout: float = 0.1):
            super().__init__()
            self.w1 = nn.Linear(d_model, intermediate_size, bias=False)
            self.w2 = nn.Linear(intermediate_size, d_model, bias=False)
            self.w3 = nn.Linear(d_model, intermediate_size, bias=False)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

    class TransformerBlock(nn.Module):
        def __init__(self, d_model: int = 512, n_heads: int = 8, intermediate_size: int = 2048, dropout: float = 0.1):
            super().__init__()
            self.ln1 = nn.LayerNorm(d_model)
            self.attn = CausalSelfAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
            self.ln2 = nn.LayerNorm(d_model)
            self.ffn = SwiGLUFFN(d_model=d_model, intermediate_size=intermediate_size, dropout=dropout)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x + self.attn(self.ln1(x))
            x = x + self.ffn(self.ln2(x))
            return x

    class FinancialMiniLLM(nn.Module):
        def __init__(
            self,
            vocab_size: int = 2048,
            d_model: int = 512,
            n_layers: int = 8,
            n_heads: int = 8,
            intermediate_size: int = 2048,
            max_seq_len: int = 512,
            num_classes: int = 2,
            dropout: float = 0.1
        ):
            super().__init__()
            self.vocab_size = vocab_size
            self.d_model = d_model
            
            self.token_embeddings = nn.Embedding(vocab_size, d_model)
            self.pos_embeddings = nn.Embedding(max_seq_len, d_model)
            self.dropout = nn.Dropout(dropout)
            
            self.blocks = nn.ModuleList([
                TransformerBlock(d_model=d_model, n_heads=n_heads, intermediate_size=intermediate_size, dropout=dropout)
                for _ in range(n_layers)
            ])
            
            self.ln_final = nn.LayerNorm(d_model)
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
            self.classifier_head = nn.Linear(d_model, num_classes)
            
            self.lm_head.weight = self.token_embeddings.weight
            self.apply(self._init_weights)

        def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        def count_parameters(self) -> int:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

        def forward(self, input_ids: torch.Tensor, mode: str = "lm") -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
            B, T = input_ids.shape
            pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device)
            
            x = self.token_embeddings(input_ids) + self.pos_embeddings(pos)
            x = self.dropout(x)
            
            for block in self.blocks:
                x = block(x)
                
            x = self.ln_final(x)
            
            if mode == "classifier":
                pooled = x[:, -1, :]
                logits = self.classifier_head(pooled)
                return logits
                
            lm_logits = self.lm_head(x)
            return lm_logits

else:
    class FinancialMiniLLM:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("FinancialMiniLLM requires PyTorch ('torch'). Install PyTorch to use this module.")


def get_mini_llm_spec() -> Dict[str, Any]:
    """Returns static architectural specification dictionary for Phase 15 documentation."""
    return {
        "architecture": "Financial Domain-Specific Decoder-Only Transformer",
        "parameters": 34600000,
        "vocab_size": 2048,
        "d_model": 512,
        "n_layers": 8,
        "n_heads": 8,
        "intermediate_size": 2048,
        "max_seq_len": 512,
        "num_classes": 2,
        "torch_available": HAS_TORCH
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    spec = get_mini_llm_spec()
    print("FinancialMiniLLM Architecture Spec:", spec)
    if HAS_TORCH:
        model = FinancialMiniLLM()
        print(f"Initialized FinancialMiniLLM | Parameter Count: {model.count_parameters():,}")
