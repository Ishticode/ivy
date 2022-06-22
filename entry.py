import jax.numpy as jnp
from jax import vmap
from jax import random
from functools import reduce
import tensorflow as tf
import ivy
ivy.set_array_significant_figures(4)
ivy.set_backend("tensorflow")
mat = ivy.reshape(ivy.arange(50), (10, 5))
batch = ivy.reshape(ivy.arange(10), (5, 2))
batch2 = ivy.reshape(ivy.arange(16), (2, 8))

mat1 = tf.random.normal((3, 5))
batch1 = tf.random.normal((5, 1, 6))


def vv(x):
    # try:
    return ivy.matmul(mat1, x)
    # except:
    #     print("failed matmul")
    #     print("shapes: ", mat1.shape, x.shape)

#print(ivy.vmap(vv, 1, 1)(batch1).shape)


print('Auto-vectorized with vmap')
for i in range(3):
    for j in range(3):
        try:
            print(ivy.vmap(vv, i, j)(batch1).shape)
        except:
            print("ignored")



# def vmap_batched_apply_matrix(v_batched):
#     return ivy.vmap(vv)(v_batched)

# print("reduce")
# print(ivy.matmul(ivy.matmul(mat, batch), batch2).shape)
# listof = [mat, batch, batch2]
# print(reduce(ivy.matmul, listof).shape)