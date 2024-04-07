import numpy as np
import torch
import sten
import scipy
import spatha

from native_scripting import compile
import functools
import ctypes
import time
import math
from heapq import nlargest
from grouped_nmv_tensor import SrNMTensor, nm_vector_mask_sparsify
from concurrent.futures import ThreadPoolExecutor

class MyCscTensor:
    def __init__(self, data):
        self.data = data

    def to_dense(self):
        return torch.from_numpy(self.data.todense())
    
def calculate_sparse_ratio(matrix):
    """
    计算矩阵的稀疏比率。

    参数:
    matrix: torch.Tensor, 需要计算稀疏比率的矩阵。

    返回:
    float, 矩阵的稀疏比率（零元素所占的百分比）。
    """
    # 计算矩阵中总的元素数量
    total_elements = matrix.numel()
    
    # 计算矩阵中零元素的数量
    # 注意：PyTorch中，使用 matrix == 0 会返回一个布尔张量，
    # 需要使用 torch.count_nonzero 来统计非零元素，然后从总元素中减去非零元素得到零元素数量
    zero_elements = total_elements - torch.count_nonzero(matrix)
    
    # 计算稀疏比率
    sparse_ratio = (zero_elements.float() / total_elements) * 100
    
    return sparse_ratio

def convert_to_csc(tensor):
    """
    将密集矩阵转换为CSC格式的函数。
    """
    return sten.SparseTensorWrapper.wrapped_from_dense(
        MyCscTensor(scipy.sparse.csc_matrix(tensor)),
        tensor,
        None
    )

def convert_to_nm(tensor):
    """
    将密集矩阵转换为N:M格式的函数。
    """
    n=2; m=4; tileM=128
    masks, columns = nm_vector_mask_sparsify(tensor, n, m, tileM)
    print("tensor.device: ", tensor.device)
    return sten.SparseTensorWrapper.wrapped_from_dense(
        SrNMTensor(n, m, tileM, tensor, masks, columns, tensor.device),
        tensor,
        None,
    )

# 自定义稀疏运算符，整合到了torch原有的自动求导框架中，可以求梯度
@sten.register_fwd_op_impl(
    operator=torch.mm,
    inp=(MyCscTensor, torch.Tensor),
    out=[(sten.KeepAll, torch.Tensor)],
)
def torch_mm_fwd_impl(ctx, inputs, output_sparsifiers):
    print("csc-dense mul")
    input1, input2 = inputs
    ctx.save_for_backward(input1, input2)
    output = torch.from_numpy(input1.wrapped_tensor.data @ input2.numpy())
    return output

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
    # print("nm-dense mul")
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

def process_matrix(matrix):
    sparse_ratio = calculate_sparse_ratio(matrix)
    print(f"稀疏比率: {sparse_ratio}%")
    if sparse_ratio > 80:
        print("转换为CSC格式")
        return convert_to_csc(matrix)
    else:
        print("转换为N:M格式")
        return convert_to_nm(matrix)
    


####################################################################################
#################                   正确性测试代码                    #################
####################################################################################
def generate_sparse_matrix(rows, cols, sparsity):
    # 计算非零元素的数量
    num_nonzeros = int(rows * cols * (1 - sparsity))

    # 生成非零元素的随机索引
    nonzero_indices = np.random.choice(rows * cols, num_nonzeros, replace=False)

    # 创建一个全零张量
    matrix = torch.zeros((rows, cols))

    # 在随机索引位置填充非零元素
    for idx in nonzero_indices:
        row_idx = idx // cols
        col_idx = idx % cols
        # 生成非零元素的值
        value = torch.rand(1)
        # 填充非零元素
        matrix[row_idx, col_idx] = value

    return matrix

def compare_tensors_with_tolerance(tensor_a, tensor_b, tolerance=1e-03):
    """
    Compare two PyTorch tensors with a tolerance for differences and calculate the number and percentage
    of elements that differ beyond this tolerance among the non-zero elements of the first tensor.

    Args:
    - tensor_a (torch.Tensor): The first tensor to compare.
    - tensor_b (torch.Tensor): The second tensor to compare.
    - tolerance (float): The tolerance within which elements are considered equal.

    Returns:
    - tuple: A tuple containing the number of elements differing beyond the tolerance among the non-zero elements of tensor_a, and their percentage.
    """
    # Calculate the absolute difference and check against the tolerance
    difference = torch.abs(tensor_a - tensor_b)
    differing_elements_mask = difference > tolerance
    
    # Calculate the number of non-zero elements in tensor_a
    non_zero_elements_mask = tensor_a != 0
    
    # Calculate the number of differing elements among the non-zero elements of tensor_a
    num_differing = torch.sum(differing_elements_mask & non_zero_elements_mask).item()
    
    # Calculate the total number of non-zero elements in tensor_a
    total_non_zero_elements = torch.sum(non_zero_elements_mask).item()
    
    # Calculate the percentage of differing elements among the non-zero elements
    if total_non_zero_elements > 0:
        percentage_differing = (num_differing / total_non_zero_elements) * 100
    else:
        percentage_differing = 0  # Avoid division by zero if tensor_a has no non-zero elements
    
    return num_differing, percentage_differing

nums = 1024
sparsity = 0.4
a = generate_sparse_matrix(nums, nums, sparsity)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_a = a.to(device)
srnm = convert_to_nm(gpu_a)
dense_srnm = srnm.wrapped_tensor.to_dense()

num_differing, percentage_differing  = compare_tensors_with_tolerance(gpu_a, dense_srnm)
print(gpu_a)
print(dense_srnm)
print(num_differing, percentage_differing)

def calculate_sparsity(tensor):
    total_elements = tensor.numel()  # 获取张量中的元素总数
    non_zero_elements = torch.count_nonzero(tensor)  # 获取张量中非零元素的数量
    sparsity = (1 - non_zero_elements / total_elements) * 100  # 计算稀疏度
    return sparsity.item()  # 返回稀疏度的数值

# 示例
sparsity = calculate_sparsity(dense_srnm)
print(f"Tensor sparsity: {sparsity}%")

