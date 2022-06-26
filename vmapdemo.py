import ivy

backends = ["tensorflow", "jax", "numpy", "torch"]
for backend in backends:
    ivy.set_backend(backend)
    mat1 = ivy.random_normal(shape=(3, 5, 3))
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
                if backend in ['torch', 'numpy']:
                    print("ignored: ", error)
                else:
                    print('ignored: ')

    print("finished")
    ivy.unset_backend()

    # if backend == 'numpy':
    #     print("split_func")
    #     ivy.set_backend(backend)
    #     for i in range(3):
    #         for j in range(3):
    #             try:
    #                 print(ivy.split_func_call(mat,
    #                                           mode="concat",
    #                                           inputs=[mat1, batched_x1],
    #                                           input_axes=(i, j),
    #                                           output_axes=0).shape)
    #             except Exception as error:
    #                 print("ignored: ", error)