import torch
import sten
import scipy

# 密集实现
a = torch.randn(5, 10, requires_grad=True)
b = torch.randn(5, 10, requires_grad=True)
c = torch.randn(10, 15, requires_grad=True)
grad_d = torch.randn(5, 15)

print("a: ", a)
print("b: ", b)
print("c: ", c)
print("grad_d: ", grad_d)

d = torch.mm(torch.add(a, b), c)

d.backward(grad_d)

print("d: ", d)


# 稀疏化实现
class MyRandomFractionSparsifier:
    def __init__(self, fraction):
        self.fraction = fraction

class MyCscTensor:
    def __init__(self, data):
        self.data = data

    def to_dense(self):
        return torch.from_numpy(self.data.todense())
    

# 自定义稀疏运算符，在不要求梯度的情况下使用比较方便
sparse_add = sten.sparsified_op(
    orig_op=torch.add,
    out_fmt=(
        (sten.KeepAll(), torch.Tensor,
         MyRandomFractionSparsifier(0.5), MyCscTensor),
    ),
    grad_out_fmt=(
        (sten.KeepAll(), torch.Tensor,
         MyRandomFractionSparsifier(0.5), MyCscTensor),
    ),
)

@sten.register_sparsifier_implementation(
    sparsifier=MyRandomFractionSparsifier, inp=torch.Tensor, out=MyCscTensor
)
def torch_tensor_to_csc_random_fraction(sparsifier, tensor, grad_fmt=None):
    return sten.SparseTensorWrapper.wrapped_from_dense(
        MyCscTensor(scipy.sparse.csc_matrix(sten.random_mask_sparsify(tensor, sparsifier.fraction))),
        tensor,
        grad_fmt,
    )


# 自定义稀疏运算符，整合到了torch原有的自动求导框架中，可以求梯度
@sten.register_fwd_op_impl(
    operator=torch.mm,
    inp=(MyCscTensor, torch.Tensor),
    out=[(sten.KeepAll, torch.Tensor)],
)
def torch_mm_fwd_impl(ctx, inputs, output_sparsifiers):
    input1, input2 = inputs
    ctx.save_for_backward(input1, input2)
    output = torch.from_numpy(input1.wrapped_tensor.data @ input2.numpy())
    return output

@sten.register_bwd_op_impl(
    operator=torch.mm,
    grad_out=[torch.Tensor],
    grad_inp=(
        (sten.KeepAll, torch.Tensor),
        (sten.KeepAll, torch.Tensor),
    ),
    inp=(MyCscTensor, torch.Tensor),
)
def torch_mm_bwd_impl(ctx, grad_outputs, input_sparsifiers):
    input1, input2 = ctx.saved_tensors
    [grad_output] = grad_outputs
    grad_input1 = torch.mm(grad_output, input2.T)
    grad_input2 = torch.from_numpy(
        input1.wrapped_tensor.data.transpose() @ grad_output)
    return grad_input1, grad_input2

sparse_d = torch.mm(sparse_add(a, b), c)
sparse_d.backward(grad_d)

print("sparse_d: ", sparse_d)

print("test sten2 success")


#  试一试自己定义一个稀疏操作符sparse_MM
# x = torch.randn(5, 10, requires_grad=True)
# y = torch.randn(10, 5, requires_grad=True)
# grad_result = torch.randn(5, 5, requires_grad=True)

# sparse_mm = sten.sparsified_op(
#     orig_op=torch.mm,
#     out_fmt=(
#         (sten.KeepAll(), torch.Tensor,
#          MyRandomFractionSparsifier(0.5), MyCscTensor),
#     ),
#     grad_out_fmt=(
#         (sten.KeepAll(), torch.Tensor,
#          MyRandomFractionSparsifier(0.5), MyCscTensor),
#     ),
# )

# MM_d = torch.mm(x,y)
# sparseMM_d = sparse_mm(x,y)

# print("MM_d: ", MM_d)
# print("sparseMM_d: ", sparseMM_d.data.to_dense())