import ivy
from ivy import Container, Linear, MultiHeadAttention, Module, Sequential, Embedding, Dropout, LayerNorm


# class TransformerConfig(Container):
#     def __init__(self, stacks, heads, dim, dropout, device=None, v=None):
#         super().__init__()


class Encoder(Module):

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
            v)

    def _forward(self, input, mask=None):
        attn = self.attention(input, mask)
        normalised = self.dropout(self.Norm(attn.to_q.v+input))
        fwd = self.ffn(normalised)
        y = self.dropout(self.Norm(fwd + normalised))
        return y


class TransformerEncoder(Module):
    def __init__(self, vocab_size, embed_size, num_stacks, num_heads, drop_out, maxlength, device=None, v=None):
        self.device = device
        self.embedding = Embedding(vocab_size, embed_size)
        self.pos_embedding = Embedding(maxlength, embed_size)
        self.blocks = Sequential(
            *[Encoder(embed_size, num_heads, drop_out, device, v) for _ in range(num_stacks)])
        ivy.Module.__init__(
            self,
            device,
            v)

    def _forward(self, x, mask=None):
        batch, seq_len = x.shape
        positions = ivy.arange(seq_len, device=self.device).broadcast_to((batch, seq_len))
        x = self.dropout(self.embedding(x) + self.pos_embedding(positions))
        output = self.blocks(x, mask)
        return output


class DecoderBlock(Module):

    def __init__(self, embed_size, mask, num_heads = 8, drop_out=0.1, device=None, v=None):
        self.mask = mask
        self.dropout = Dropout(drop_out)
        self.norm = LayerNorm(embed_size)
        self.attn = MultiHeadAttention(embed_size, num_heads, embed_size // num_heads, drop_out, device, v)

    def _forward(self, x, mask, opt_mask=None):
        batch, seq_len = x.shape
        positions = ivy.arange(seq_len).broadcast_to((batch, seq_len))
        x = self.dropout(self.embedding(x) + self.pos_embedding(positions))
        attn = self.attn(x, self.mask)
        output = self.dropout(self.norm(attn + x))
        attn2 = self.attn(output, opt_mask)
        output = self.dropout(self.norm(attn2 + output))
        ff = self.ffn(output)
        output = self.dropout(self.norm(ff + output))
        return output


class TransformerDecoder(Module):
    def __init__(self, tg_vocab_size, embed_size, mask,maxlength, num_stacks=6, num_heads=8, drop_out=0.1, device=None, v=None):
        self.device = device
        self.embedding = Embedding(tg_vocab_size, embed_size)
        self.pos_embedding = Embedding(maxlength, embed_size)
        self.blocks = Sequential(
            *[DecoderBlock(embed_size, mask, num_heads, drop_out, device, v) for _ in range(num_stacks)])
        ivy.Module.__init__(
            self,
            device,
            v)

    def _forward(self, x, mask, opt_mask=None):
        batch, seq_len = x.shape
        positions = ivy.arange(seq_len).broadcast_to((batch, seq_len))
        x = self.dropout(self.embedding(x) + self.pos_embedding(positions))
        output = self.blocks(x, mask, opt_mask)
        return output


class Transformer(Module):
    def __init__(self, src_vocab_size, tg_vocab_size, embed_size, mask, maxlength, num_stacks=6, num_heads=8, drop_out=0.1, device=None, v=None):
        self.device = device
        self.encoder = TransformerEncoder(src_vocab_size, embed_size, num_stacks, num_heads, drop_out, maxlength, device, v)
        self.decoder = TransformerDecoder(tg_vocab_size, embed_size, mask, maxlength, num_stacks, num_heads, drop_out, device, v)
        ivy.Module.__init__(
            self,
            device,
            v)

    def _create_masks(self, src, tg):
        pass

    def _forward(self, x, y, mask, opt_mask=None):
        pass
        # TODO: implement masks


