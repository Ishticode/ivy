import jax.numpy as jnp
from jax import vmap
from jax import random
import tensorflow as tf
import ivy
ivy.set_backend("tensorflow")
key = random.PRNGKey(0)
mat = tf.random.uniform(shape=(10, 5))
batched_x = tf.random.uniform(shape=(2, 5))


def apply_matrix(v):
    return ivy.vecdot(mat, v)


def vmap_batched_apply_matrix(v_batched):
    return ivy.vmap(apply_matrix)(v_batched)


print('Auto-vectorized with vmap')
print((vmap_batched_apply_matrix(batched_x)))
