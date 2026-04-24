import mlx.core as mx
from .basics import linear, silu
from .attention import scaled_dot_product_attention_grouped
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from typing import Any
from .embedding import Embedding
from .quantize import dequantize_linear


class Qwen2MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        assert(self.hidden_size % self.num_heads == 0, f"hidden_size mod num_heads is not zero")
        self.head_size = self.hidden_size // self.num_heads
        self.kv_heads = num_kv_heads
        self.wq = wq
        self.wk = wk
        self.wv = wv
        # the output dimmenssion of wk and wv is different
        self.wo = wo
        self.bq = bq
        self.bk = bk
        self.bv = bv
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.rope = RoPE(self.head_size, self.max_seq_len, theta)
        self.scale = mx.rsqrt(self.head_size)


    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        B, L, _ = x.shape

        projection_q = linear(x, self.wq, self.bq).reshape(B, L, self.num_heads, self.head_size)
        projection_k = linear(x, self.wk, self.bk).reshape(B, L, self.kv_heads, self.head_size)
        projection_v = linear(x, self.wv, self.bv).reshape(B, L, self.kv_heads, self.head_size)

        projection_q = self.rope(projection_q, offset=slice(0, L))
        projection_k = self.rope(projection_k, offset=slice(0, L))
        projection_q = projection_q.transpose(0, 2, 1, 3)
        projection_k = projection_k.transpose(0, 2, 1, 3)
        projection_v = projection_v.transpose(0, 2, 1, 3)

        output = scaled_dot_product_attention_grouped(
            projection_q.astype(mx.float32),
            projection_k.astype(mx.float32), 
            projection_v.astype(mx.float32), 
            self.scale, mask
            ).astype(x.dtype)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, self.hidden_size)
        return linear(output, self.wo)


class Qwen2MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
    ):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down

    def __call__(self, x: mx.array) -> mx.array:
        return linear(silu(linear(x, self.w_gate)) * linear(x, self.w_up),self.w_down)


class Qwen2TransformerBlock:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        intermediate_size: int,
        rms_norm_eps: float,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        pass

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        pass


class Qwen2ModelWeek1:
    def __init__(self, mlx_model: Any):
        pass

    def __call__(
        self,
        inputs: mx.array,
    ) -> mx.array:
        pass
