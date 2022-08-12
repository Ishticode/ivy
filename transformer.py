import ivy
from ivy import Container, Linear, MultiHeadAttention, Module, Sequential, Softmax


class TransformerConfig(Container):
    def __init__(self, stacks, heads, dim, dropout, device=None, v=None):
        super().__init__()


class TransformerEncoder(Module):
    def __init__(self, config: TransformerConfig):
        self.config = config

    def _forward(self, x, **kwargs):
        pass


class Embedding():
    def __init__(self):
        pass
