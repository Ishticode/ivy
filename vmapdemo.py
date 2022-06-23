import ivy

backends = ["tensorflow", "jax", "numpy", "torch"]
for backend in backends:
    ivy.set_backend(backend)
    mat1 = ivy.random_normal(shape=(3, 5, 3, 2))
    batched_x1 = ivy.random_normal(shape=(5, 3, 2, 5))


    def mat(mat, b):
        ret = ivy.matmul(mat, b)
        ret = ivy.to_native(ret)
        return ret

    print()
    print(backend+"-----------------")
    print()

    for i in range(3):
        for j in range(3):
            try:
                print(ivy.vmap(mat, (i, j), 0)(mat1, batched_x1).shape)
            except Exception as error:
                print("ignored: ")

    print("finished")
    ivy.unset_backend()