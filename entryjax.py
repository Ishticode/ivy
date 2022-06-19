import jax.numpy as jnp
from jax import vmap
from jax import random
import tensorflow as tf
import ivy
ivy.set_backend("jax")
key = random.PRNGKey(0)
mat = jnp.reshape(jnp.arange(50), (10, 5))
vec = jnp.reshape(jnp.arange(10), (5, 2))
batch = jnp.reshape(jnp.arange(5), (5))


def vv(x, y):
    return jnp.dot(mat, y)


def vmap_batched_apply_matrix(v_batched):
    return ivy.vmap(vv(batch))


print('Auto-vectorized with vmap')
print((vmap_batched_apply_matrix(vec)))
