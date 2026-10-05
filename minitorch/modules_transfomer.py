import numpy as np
from .tensor import tensor, tensor_from_numpy
from .tensor_functions import (zeros, ones)
from .module import Module, Parameter
from .modules_basic import (
    Embedding,
    Dropout,
    LayerNorm1d,
    Linear
)
from .tensor_ops import TensorBackend
from .nn import (
    max,
    softmax,
    dropout,
    GELU,
)
from typing import Any, Dict, Optional, Sequence, Tuple

datatype = np.float32


class MultiHeadAttention(Module):
    def __init__(self, n_embd: int, n_head: int, causal: bool=False, p_dropout: float=0.1, bias: bool=True, backend: TensorBackend=None, use_fused_kernel: bool=False):
        super().__init__()
        """Implements Multi-Head Attention as described in "Attention Is All You Need"."""

        self.backend = backend
        self.n_embd = n_embd
        self.n_head = n_head
        self.causal = causal
        self.attn_hidden_dim = n_embd // n_head

        assert n_embd % n_head == 0

        self.q_projection = Linear(n_embd, n_embd, bias=bias, backend=backend)
        self.k_projection = Linear(n_embd, n_embd, bias=bias, backend=backend)
        self.v_projection = Linear(n_embd, n_embd, bias=bias, backend=backend)
        self.out_projection = Linear(n_embd, n_embd, bias=bias, backend=backend)
        self.dropout = Dropout(p_dropout)

        self.use_fused_kernel = use_fused_kernel

    def create_causal_mask(self, bs, nh, seq_len):
        mask = -np.finfo(datatype).max * np.triu(
            np.ones((bs, nh, seq_len, seq_len), dtype=datatype),
            1
        )
        return tensor_from_numpy(mask, backend=self.backend)

    def project_to_query_key_value(self, x):
        batch_size, seq_len, n_embd = x.shape

        x_flat = x.view(batch_size * seq_len, n_embd)

        q = self.q_projection(x_flat)
        k = self.k_projection(x_flat)
        v = self.v_projection(x_flat)

        q = q.view(
            batch_size,
            seq_len,
            self.n_head,
            self.attn_hidden_dim
        )

        k = k.view(
            batch_size,
            seq_len,
            self.n_head,
            self.attn_hidden_dim
        )

        v = v.view(
            batch_size,
            seq_len,
            self.n_head,
            self.attn_hidden_dim
        )

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        kT = k.permute(0, 1, 3, 2)

        return q, kT, v

    def self_attention(self, q, kT, v):
        batch_size, num_head, queries_len, q_dim = q.shape
        _, _, k_dim, _ = kT.shape
        _, _, _, v_dim = v.shape

        assert q_dim == k_dim == v_dim

        if not self.use_fused_kernel:
            scores = (q @ kT) / np.sqrt(self.attn_hidden_dim)

            if self.causal:
                scores = scores + self.create_causal_mask(
                    batch_size,
                    num_head,
                    queries_len
                )

            attention = softmax(scores, dim=3)
            attention = self.dropout(attention)

            result = attention @ v

            result = result.permute(0, 2, 1, 3).contiguous()

            result = result.view(
                batch_size,
                queries_len,
                self.n_embd
            )

        else:
            # BEGIN ASSIGN4_3
            scores = (q @ kT) / np.sqrt(self.attn_hidden_dim)

            # The fused CUDA softmax handles decoder causal masking
            # internally through mask_future=True. It still expects
            # a [batch_size, to_len] mask buffer.
            mask = tensor_from_numpy(
                np.zeros(
                    (batch_size, queries_len),
                    dtype=datatype
                ),
                backend=self.backend
            )

            attention = scores.attn_softmax(mask)
            attention = self.dropout(attention)

            result = attention @ v

            result = result.permute(
                0, 2, 1, 3
            ).contiguous()

            result = result.view(
                batch_size,
                queries_len,
                self.n_embd
            )
            # END ASSIGN4_3

        return result

    def forward(self, x):
        batch_size, seq_len, n_embd = x.shape

        q, kT, v = self.project_to_query_key_value(x)

        x = self.self_attention(q, kT, v)

        x = x.view(batch_size * seq_len, n_embd)
        x = self.out_projection(x)

        return x.view(batch_size, seq_len, n_embd)


class FeedForward(Module):
    def __init__(self, n_embd: int, middle_dim: int=256, p_dropout: float=0.1, bias: bool=True, backend: TensorBackend=None):
        super().__init__()
        """The Feed Forward Module.
        
        Args:
            n_embd     : in_size of first linear layer and out_size of last linear layer
            middle_dim : out_size of first linear layer and in_size of last linear layer
            p_dropout  : Dropout probability
            bias       : If bias should be applied in linear layers
        
        Attributes:
            linear_in  : first linear layer
            linear_out : second linear layer
            dropout    : dropout layer
        """
        self.linear_in = Linear(
            n_embd,
            middle_dim,
            bias=bias,
            backend=backend
        )

        self.linear_out = Linear(
            middle_dim,
            n_embd,
            bias=bias,
            backend=backend
        )

        self.dropout = Dropout(p_dropout)

    def forward(self, x):
        """A FFN Module in a Pre-LN Transformer with GELU Activation and dropout.

        Args:
            x : Tensor of shape (batch_size x seq_len x n_embd)

        Returns:
            output : Tensor of shape (batch_size x seq_len x n_embd)
        """
        batch_size, seq_len, n_embd = x.shape

        x = GELU(
            self.linear_in(
                x.view(batch_size * seq_len, n_embd)
            )
        )

        x = self.dropout(
            self.linear_out(x)
        )

        x = x.view(
            batch_size,
            seq_len,
            n_embd
        )

        return x


class TransformerLayer(Module):
    def __init__(self, n_embd: int, n_head: int, p_dropout: float=0.1, ln_eps: float=1e-8, bias: bool=True, backend: TensorBackend=None, use_fused_kernel: bool=False):
        super().__init__()
        """A Transformer Layer in a Pre-LN Transformer.

        Args: 
            n_embd : Dimensionality of embeddings and hidden states
            n_head : Number of heads for MultiHeadAttention
            p_dropout : Dropout ratio for dropout layer
            ln_eps : A value added for numerical stability in LayerNorm
            bias : If bias should be added in linear layers
        
        Attributes:
            ln_1 : First LayerNorm1d layer before MultiHeadAttention
            ln_2 : Second LayerNorm1d layer after MultiHeadAttention
            attention : MultiHeadAttention layer
            ff : FeedForward layer
        """

        self.attention = MultiHeadAttention(
            n_embd=n_embd,
            n_head=n_head,
            causal=True,
            p_dropout=p_dropout,
            bias=bias,
            backend=backend,
            use_fused_kernel=use_fused_kernel,
        )

        self.ff = FeedForward(
            n_embd=n_embd,
            middle_dim=4 * n_embd,
            p_dropout=p_dropout,
            bias=bias,
            backend=backend,
        )

        self.use_fused_kernel = use_fused_kernel

        if not self.use_fused_kernel:
            self.ln_1 = LayerNorm1d(
                n_embd,
                ln_eps,
                backend=backend
            )

            self.ln_2 = LayerNorm1d(
                n_embd,
                ln_eps,
                backend=backend
            )
        else:
            # BEGIN ASSIGN4_3
            self.ln_1 = LayerNorm1d(
                n_embd,
                ln_eps,
                backend=backend
            )

            self.ln_2 = LayerNorm1d(
                n_embd,
                ln_eps,
                backend=backend
            )
            # END ASSIGN4_3

    def forward(self, x):
        """
        The forward function of a Transformer Layer for a PRENORM Transformer.
        Input: the hidden states from previous layers `x` with shape (batch_size, seq_len, x_dim)
        Ouput: the hidden states after the Transformer Layer `x` with shape (batch_size, seq_len, x_dim)
        """
        batch_size, seq_len, x_dim = x.shape

        if not self.use_fused_kernel:
            norm_x = self.ln_1(
                x.view(batch_size * seq_len, x_dim)
            ).view(batch_size, seq_len, x_dim)

            x = x + self.attention(norm_x)

            norm_x = self.ln_2(
                x.view(batch_size * seq_len, x_dim)
            ).view(batch_size, seq_len, x_dim)

            x = x + self.ff(norm_x)

        else:
            # BEGIN ASSIGN4_3
            x_flat = x.view(
                batch_size * seq_len,
                x_dim
            )

            norm_x = x_flat.layernorm(
                self.ln_1.weights.value,
                self.ln_1.bias.value
            ).view(
                batch_size,
                seq_len,
                x_dim
            )

            x = x + self.attention(norm_x)

            x_flat = x.view(
                batch_size * seq_len,
                x_dim
            )

            norm_x = x_flat.layernorm(
                self.ln_2.weights.value,
                self.ln_2.bias.value
            ).view(
                batch_size,
                seq_len,
                x_dim
            )

            x = x + self.ff(norm_x)
            # END ASSIGN4_3

        return x


class DecoderLM(Module):
    def __init__(
        self, 
        n_vocab: int,
        n_embd: int,
        n_head: int,
        n_positions: int,
        p_dropout: float=0.1,
        ln_eps: float=1e-5, 
        bias: bool=True,
        backend: TensorBackend=None,
        use_fused_kernel: bool=False,
    ):
        super().__init__()
        """A Full Decoder-only Pre-LN Transformer with 4 Transformer Layers."""
        self.backend = backend
        self.n_embd = n_embd
        self.n_vocab = n_vocab

        self.token_embeddings = Embedding(
            n_vocab,
            n_embd,
            backend=backend
        )

        self.position_embeddings = Embedding(
            n_positions,
            n_embd,
            backend=backend
        )

        self.t_layer_1 = TransformerLayer(
            n_embd,
            n_head,
            p_dropout,
            ln_eps,
            bias,
            backend,
            use_fused_kernel
        )

        self.t_layer_2 = TransformerLayer(
            n_embd,
            n_head,
            p_dropout,
            ln_eps,
            bias,
            backend,
            use_fused_kernel
        )

        self.t_layer_3 = TransformerLayer(
            n_embd,
            n_head,
            p_dropout,
            ln_eps,
            bias,
            backend,
            use_fused_kernel
        )

        self.t_layer_4 = TransformerLayer(
            n_embd,
            n_head,
            p_dropout,
            ln_eps,
            bias,
            backend,
            use_fused_kernel
        )

        self.dropout = Dropout(p_dropout)

        self.lm_head = Linear(
            n_embd,
            n_vocab,
            bias=bias,
            backend=backend
        )

        self.use_fused_kernel = use_fused_kernel

        if not self.use_fused_kernel:
            self.ln = LayerNorm1d(
                n_embd,
                ln_eps,
                backend=backend
            )
        else:
            # BEGIN ASSIGN4_3
            self.ln = LayerNorm1d(
                n_embd,
                ln_eps,
                backend=backend
            )
            # END ASSIGN4_3

    def forward(self, idx):
        """A Forward pass of a Decoder-only Transformer Language model."""
        batch_size, seq_len = idx.shape

        pos = tensor(
            [i for i in range(seq_len)],
            backend=self.backend
        ).view(1, seq_len)

        if not self.use_fused_kernel:
            tok_emb = self.token_embeddings(idx)
            pos_emb = self.position_embeddings(pos)

            x = tok_emb + pos_emb
            x = self.dropout(x)

            x = self.t_layer_1(x)
            x = self.t_layer_2(x)
            x = self.t_layer_3(x)
            x = self.t_layer_4(x)

            x = self.ln(
                x.view(batch_size * seq_len, self.n_embd)
            )

            x = self.lm_head(x)

            x = x.view(
                batch_size,
                seq_len,
                self.n_vocab
            )

        else:
            # BEGIN ASSIGN4_3
            tok_emb = self.token_embeddings(idx)
            pos_emb = self.position_embeddings(pos)

            x = tok_emb + pos_emb
            x = self.dropout(x)

            x = self.t_layer_1(x)
            x = self.t_layer_2(x)
            x = self.t_layer_3(x)
            x = self.t_layer_4(x)

            x = x.view(
                batch_size * seq_len,
                self.n_embd
            ).layernorm(
                self.ln.weights.value,
                self.ln.bias.value
            )

            x = self.lm_head(x)

            x = x.view(
                batch_size,
                seq_len,
                self.n_vocab
            )
            # END ASSIGN4_3

        return x
