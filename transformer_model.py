import ivy
from ivy import Container, Linear, MultiHeadAttention, Module, Sequential, softmax, Embedding, Dropout


# class TransformerConfig(Container):
#     def __init__(self, stacks, heads, dim, dropout, device=None, v=None):
#         super().__init__()


class TransformerBlock(Module):

    def __init__(self, embed_size, num_heads, head_dim, dropout, device=None, v=None):
        super().__init__()
        self.attention = MultiHeadAttention(embed_size, num_heads, head_dim, dropout, device, v)
        self.ffn = Sequential(
            Linear(embed_size, 4 * head_dim),
            Linear(4 * head_dim, embed_size),
            softmax,
        )
        #add normlizer
        self.dropout = dropout
        self.device = device
        self.v = v

    def _forward(self, x, mask=None):
        y = self.attention(x, x, x, mask)
        y = (1 - self.dropout) * y + self.dropout * x
        y = self.ffn(y)
        y = (1 - self.dropout) * y + self.dropout * x
        return y


class TransformerEncoder(Module):
    def __init__(self, vocab, embed_size, num_stacks, num_heads, drop_out, maxlength, device=None, v=None):
        super().__init__()
        self.vocab = vocab
        self.embed_size = embed_size
        self.num_layers = num_stacks
        self.heads = num_heads
        self.head_dim = embed_size // num_heads
        self.drop_out = Dropout(drop_out)
        self.maxlength = maxlength
        self.device = device
        self.v = v
        self.embedding = Embedding(vocab, embed_size)
        self.pos_embedding = Embedding(maxlength, embed_size)
        self.layers = Sequential(
            *[TransformerBlock(embed_size, num_heads, self.head_dim, drop_out, device, v) for _ in range(num_stacks)]
        )

    def _forward(self, x, mask=None):
        batch, seq_len = x.shape
        pass