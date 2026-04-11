import mlx.core as mx


class RoPE:
    def __init__(
        self,
        dims: int,
        seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        self.dims = dims
        self.half_dim = dims // 2
        self.seq_len = seq_len
        self.base = base
        self.traditional = traditional
        inner = mx.arange(0, self.half_dim, dtype=mx.float32) / self.half_dim
        freqs = mx.power(base, -inner)
        t = mx.arange(self.seq_len)
        freqs = mx.outer(t, freqs)
        self.cos_freqs = mx.cos(freqs)
        self.sin_freqs = mx.sin(freqs)
        

    def __call__(
        self, x: mx.array, offset: list[slice] | slice | None = None
    ) -> mx.array:
        N, S, H, D = x.shape
        if offset is not None:
            if isinstance(offset, slice):
                assert(offset.stop - offset.start == S, f"offset length must equal to sequence length {S}")
            elif isinstance(offset, list):
                assert(len(offset) == N, f"offset size must qual to batch size {N}")
                for o in offset:
                    assert(o.stop - o.start == S, f"offset length must qual to sequnce length {S}")
                offset = mx.array([list(range(o.start, o.stop)) for o in offset])
        cos_basis = (self.cos_freqs[:S, :] if offset is None else self.cos_freqs[offset, :])
        sin_basis = (self.sin_freqs[:S, :] if offset is None else self.sin_freqs[offset, :])
        if self.traditional:
            x = x.reshape(N, S, H, D // 2, 2)
            x1 = x[..., 0]
            x2 = x[..., 1]
        else:
            x1 = x[..., : self.half_dim]
            x2 = x[..., self.half_dim : self.dims]
        cos_basis = cos_basis.reshape(-1, S, 1, self.half_dim)
        sin_basis = sin_basis.reshape(-1, S, 1, self.half_dim)
        real = mx.multiply(x1, cos_basis) - mx.multiply(x2, sin_basis)
        imag = mx.multiply(x1, sin_basis) + mx.multiply(x2, cos_basis)
        if self.traditional:
            y = mx.stack([real, imag], axis=-1)
            y = y.reshape(N, S, H, D)
        else:
            y = mx.concat([real, imag], axis=-1)
            y = y.reshape(N, S, H, D)
        return y.astype(x.dtype)

