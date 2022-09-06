import ivy
ivy.set_backend('torch')
# class TransformerConfig(ivy.Container):
#     def __init__(self, stacks, heads, dim, dropout, device=None, v=None):
#         super().__init__()


class Encoder(ivy.Module):

    def __init__(self, embed_dim, num_heads, dropout, device=None, v=None):
        head_dim = embed_dim // num_heads
        self.attention = ivy.MultiHeadAttention(embed_dim,
                                                num_heads,
                                                head_dim,
                                                dropout,
                                                device=device,
                                                v=v)
        self.ffn = ivy.Sequential(
            ivy.Linear(embed_dim, 4 * head_dim),
            ivy.Linear(4 * head_dim, embed_dim),
        )
        self.dropout = ivy.Dropout(dropout)
        self.Norm = ivy.LayerNorm((embed_dim,))
        ivy.Module.__init__(
            self,
            device,
            v)

    def _forward(self, x, mask=None):
        attn = self.attention(x, mask=mask)
        normalised = self.dropout(self.Norm(attn + x))
        fwd = self.ffn(normalised)
        y = self.dropout(self.Norm(fwd + normalised))
        return y

encoder = Encoder(16, 8, 0.1)
x = ivy.random_uniform(shape=(3, 16))
print(encoder(x))


class TransformerEncoder(ivy.Module):
    def __init__(self, vocab_size, embed_size, num_stacks, num_heads, drop_out, maxlength, device=None, v=None):
        self.device = device
        self.embedding = ivy.Embedding(vocab_size, embed_size)
        self.pos_embedding = ivy.Embedding(maxlength, embed_size)
        self.blocks = ivy.Sequential(
            *[Encoder(embed_size, num_heads, drop_out, device, v) for _ in range(num_stacks)])
        ivy.ivy.Module.__init__(
            self,
            device,
            v)

    def _forward(self, x, mask=None):
        batch, seq_len = x.shape
        positions = ivy.arange(seq_len, device=self.device).broadcast_to((batch, seq_len))
        x = self.dropout(self.embedding(x) + self.pos_embedding(positions))
        output = self.blocks(x, mask)
        return output


class DecoderBlock(ivy.Module):

    def __init__(self, embed_size, mask, num_heads = 8, drop_out=0.1, device=None, v=None):
        self.mask = mask
        self.dropout = ivy.Dropout(drop_out)
        self.norm = ivy.LayerNorm(embed_size)
        self.attn = ivy.MultiHeadAttention(embed_size, num_heads, embed_size // num_heads, drop_out, device, v)

    def _forward(self, x, enc_output, opt_mask=None):
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


class TransformerDecoder(ivy.Module):
    def __init__(self, tg_vocab_size, embed_size, trg_mask,maxlength, num_stacks=6, num_heads=8, drop_out=0.1, device=None, v=None):
        self.device = device
        self.embedding = ivy.Embedding(tg_vocab_size, embed_size)
        self.pos_embedding = ivy.Embedding(maxlength, embed_size)
        self.blocks = ivy.Sequential(
            *[DecoderBlock(embed_size, trg_mask, num_heads, drop_out, device, v) for _ in range(num_stacks)])
        ivy.ivy.Module.__init__(
            self,
            device,
            v)

    def _forward(self, x, enc_output,trg_mask, opt_mask=None):
        batch, seq_len = x.shape
        positions = ivy.arange(seq_len).broadcast_to((batch, seq_len))
        x = self.dropout(self.embedding(x) + self.pos_embedding(positions))
        output = self.blocks(x, enc_output, trg_mask, opt_mask)
        return output


class Transformer(ivy.Module):
    def __init__(self, src_vocab_size, tg_vocab_size, embed_size, mask, maxlength, num_stacks=6, num_heads=8, drop_out=0.1, device=None, v=None):
        self.device = device
        self.encoder = TransformerEncoder(src_vocab_size, embed_size, num_stacks, num_heads, drop_out, maxlength, device, v)
        self.decoder = TransformerDecoder(tg_vocab_size, embed_size, mask, maxlength, num_stacks, num_heads, drop_out,
                                          device, v)
        ivy.ivy.Module.__init__(
            self,
            device,
            v)

    def _create_masks(self, src, tg):
        pass

    def _forward(self, x, y, mask, opt_mask=None):
        pass
        # TODO: implement masks


