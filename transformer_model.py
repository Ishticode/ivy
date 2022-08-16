import ivy
from ivy import Container, Linear, MultiHeadAttention, Module, Sequential, SoftMax, Embedding, Dropout, LayerNorm


# class TransformerConfig(Container):
#     def __init__(self, stacks, heads, dim, dropout, device=None, v=None):
#         super().__init__()


class TransformerBlock(Module):

    def __init__(self, embed_dim, num_heads, dropout, device=None, v=None):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.device = device
        self.v = v
        self.attention = MultiHeadAttention(self.embed_dim,
                                            self.num_heads,
                                            self.head_dim,
                                            self.dropout,
                                            self.device,
                                            self.v)
        self.ffn = Sequential(
            Linear(self.embed_dim, 4 * self.head_dim),
            Linear(4 * self.head_dim, self.embed_size),
        )
        self.dropout = Dropout(dropout)
        self.Norm = LayerNorm(self.embed_dim)
        ivy.Module.__init__(
            self,
            device,
            v if v_exists else None,
            dtype=dtype,
        )

    def _forward(self, input, mask=None):
        attn = self.attention(input, mask)
        normalised = self.dropout(self.Norm(attn.to_q.v+input))
        fwd = self.ffn(normalised)
        y = self.dropout(self.Norm(fwd + normalised))
        return y


class TransformerEncoder(Module):
    def __init__(self, vocab_size, embed_size, num_stacks, num_heads, drop_out, maxlength, device=None, v=None):
        self.embedding = Embedding(vocab_size, embed_size)
        self.pos_embedding = Embedding(maxlength, embed_size)
        self.blocks = Sequential(
            *[TransformerBlock(embed_size, num_heads, drop_out, device, v) for _ in range(num_stacks)])
        

    def _create_variables(self, device, dtype):
        pass

    def _forward(self, x, mask=None):
        batch, seq_len = x.shape
        pass