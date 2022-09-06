import ivy
ivy.set_backend('torch')
mha = ivy.MultiHeadAttention(4, 8, 32, 0.1)

#
# x= ivy.random_uniform(shape=(3,4))
# print(x)
# print(mha._to_out(mha._to_q(x)))
# print(mha(x))

x = ivy.array([[1.,2.,3.],[4.,5.,6]])
print(ivy.layer_norm(x,[0]))