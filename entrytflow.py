import tensorflow as tf
import numpy as np
import ivy
ivy.set_array_significant_figures(4)
ivy.set_backend("tensorflow")
# mat = ivy.reshape(ivy.arange(50), (10, 5))
# batch = ivy.reshape(ivy.arange(10), (2, 5))


mat1 = tf.constant(np.random.randint(2, 4, size=(3, 5)))
batched_x1 = tf.constant(np.random.randint(2, 3, size=(5, 3, 2)))

def vv(x):
    # try:
    return ivy.matmul(mat1, x)
    # except:
    #     print("failed matmul")
    #     print("shapes: ", mat1.shape, x.shape)


print(ivy.vmap(vv, [0, 1], 0)(batched_x1).shape)


print('Auto-vectorized with vmap')
for i in range(3):
    for j in range(3):
        try:
            print(ivy.vmap(vv, (i, 2), j)(batched_x1).shape)
        except:
            print("ignored")
        #print(ivy.vmap(vv, (i, j), 0)(mat1, batched_x1).shape)

# def vmap_batched_apply_matrix(v_batched):
#     return ivy.vmap(vv)(v_batched)
