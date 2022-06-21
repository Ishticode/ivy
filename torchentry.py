import ivy
import numpy as np
ivy.set_backend("torch")
arr = ivy.random_normal(0, 1, (3, 4, 5))
# print(ivy.unstack(arr, 0).shape)
from functorch import vmap
import torch

mat1 = torch.tensor(np.random.randint(2, 4, size=(3, 5)))
batched_x1 = torch.tensor(np.random.randint(2, 3, size=(5, 3, 2)))


# batched_x1 = jnp.expand_dims(batched_x1, 0)
# print(mat1,batched_x1.T)


def vdot(mat, batch):
    return torch.matmul(mat, batch)


for i in range(3):
    for j in range(3):
        try:
            print(ivy.vmap(vdot, (i, j), 0)(mat1, batched_x1).shape)
        except:
            print("ignored")