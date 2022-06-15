import jax.numpy as jnp
from jax import vmap
from jax import random
import ivy
ivy.set_backend("jax")
key = random.PRNGKey(0)
mat = random.normal(key, (10, 5))
batched_x = random.normal(key, (2, 5))


def apply_matrix(v):
    return jnp.dot(mat, v)


def vmap_batched_apply_matrix(v_batched):
    return ivy.vmap(apply_matrix)(v_batched)


print('Auto-vectorized with vmap')
print(type(vmap_batched_apply_matrix(batched_x)))
