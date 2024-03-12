import torch
import sten
import scipy

from native_scripting import compile
import functools
import ctypes
import time
import math
from heapq import nlargest
from grouped_nmv_tensor import SrNMTensor, nm_vector_mask_sparsify

# 普通正常密集矩阵操作的测试
a = torch.randn(768, 768, requires_grad=True)
b = torch.randn(768, 768, requires_grad=True)
c = torch.ones(768, 768, requires_grad=True)
    
class NMVectorSparsifier:
    def __init__(self, n, m, tileM):
        self.n = n
        self.m = m
        self.tileM = tileM

@sten.register_sparsifier_implementation(
    sparsifier=NMVectorSparsifier, inp=torch.Tensor, out=SrNMTensor
)
def torch_tensor_to_srnm_random_fraction(sparsifier, tensor, grad_fmt=None):
    print("inside NMVectorSparsifier sparsifier")
    masks, columns = nm_vector_mask_sparsify(tensor, sparsifier.n, sparsifier.m, sparsifier.tileM)
    return sten.SparseTensorWrapper.wrapped_from_dense(
        SrNMTensor(sparsifier.n, sparsifier.m, sparsifier.tileM, tensor, masks, columns),
        tensor,
        grad_fmt,
    )

n=2; m=4; tileM=128
sparse_add = sten.sparsified_op(
    orig_op=torch.add,
    out_fmt=(
        (sten.KeepAll(), torch.Tensor,
         NMVectorSparsifier(n,m,tileM), SrNMTensor),
    ),
    grad_out_fmt=(
        (sten.KeepAll(), torch.Tensor,
         NMVectorSparsifier(n,m,tileM), SrNMTensor),
    ),
)

import spatha
@sten.register_fwd_op_impl(
    operator=torch.mm,
    inp=(SrNMTensor, torch.Tensor),
    out=[(sten.KeepAll, torch.Tensor)],
)
def sparse_torch_add_fwd_impl(ctx, inputs, output_sparsifiers):
    """ input, other = inputs
    [out_sp] = output_sparsifiers
    dense_out = torch.add(
        input.wrapped_tensor.to_dense(),
        other.wrapped_tensor.to_dense(),
    )
    return torch_tensor_to_srnm_random_fraction(
        KeepAll(), nm_vector_mask_sparsify(dense_out, out_sp.n, out_sp.m, out_sp.tileM)
    ) """

    input1, input2 = inputs
    ctx.save_for_backward(input1, input2)

    bias = torch.zeros(input1.wrapped_tensor.nrows)

    output = spatha.spmm(input1.wrapped_tensor.metadata.cuda(), # metadata
                          input1.wrapped_tensor.columns.cuda(),  # indices
                          input1.wrapped_tensor.values.to(dtype=torch.half).cuda(),                                    # values
                          input2.to(dtype=torch.half).cuda(),    # rhs_matrix
                          bias.to(dtype=torch.half).cuda(),
                          input1.wrapped_tensor.nrows,           # A_num_rows
                          input1.wrapped_tensor.ncols,           # A_num_cols
                          input2.shape[1],                       # B_num_cols
                          input1.wrapped_tensor.tileM,           # vec_length
                          input1.wrapped_tensor.n,               # n
                          input1.wrapped_tensor.m,               # m
                          input1.wrapped_tensor.nnz,             # nnz
                          0,                                     # seed
                          32,                                    # mbrow
                          4                                     # brow
                          )

    return output


srnm = sparse_add(a, b)
d = torch.mm(srnm, c)

e = torch.from_numpy(srnm.wrapped_tensor.to_dense().to(dtype=torch.half).detach().numpy() @ c.detach().numpy()).to(device="cuda:0").to(dtype=torch.half)

print("d: ", d)
print("e: ", e)
print("d.T equal e: ", torch.equal(d.T,e) )
print("d.T allclose e: ", torch.allclose(d.T,e) )