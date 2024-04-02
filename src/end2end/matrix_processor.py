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

####################################################################################
#################                     功能函数                      #################
####################################################################################
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
    of elements that differ beyond this tolerance.

    Args:
    - tensor_a (torch.Tensor): The first tensor to compare.
    - tensor_b (torch.Tensor): The second tensor to compare.
    - tolerance (float): The tolerance within which elements are considered equal.

    Returns:
    - tuple: A tuple containing the number of elements differing beyond the tolerance and their percentage.
    """
    # Calculate the absolute difference and check against the tolerance
    num_differing = torch.sum(torch.abs(tensor_a - tensor_b) > tolerance).item()
    
    # Calculate the total number of elements in the tensor
    total_elements = tensor_a.numel()
    
    # Calculate the percentage of differing elements
    percentage_differing = (num_differing / total_elements) * 100
    
    return num_differing, percentage_differing


####################################################################################
#################                   稀疏格式类                      #################
####################################################################################
class MyCooTensor:
    def __init__(self, data):
        self.data = data

    def to_dense(self):
        return self.data.to_dense()
    
class MyCscTensor:
    def __init__(self, data):
        self.data = data

    def to_dense(self):
        return self.data.to_dense()
    
class MyCsrTensor:
    def __init__(self, data):
        self.data = data

    def to_dense(self):
        return self.data.to_dense()

####################################################################################
#################                   格式转化函数                    #################
####################################################################################
def convert_to_coo(tensor):
    """
    将密集矩阵转换为COO格式的函数。
    """  
    return sten.SparseTensorWrapper.wrapped_from_dense(
        MyCooTensor(tensor.to_sparse_coo()),
        tensor,
        None
    )

def convert_to_csc(tensor):
    """
    将密集矩阵转换为CSC格式的函数。
    """
    return sten.SparseTensorWrapper.wrapped_from_dense(
        MyCscTensor(tensor.to_sparse_csc()),
        tensor,
        None
    )

def convert_to_csr(tensor):
    """
    将密集矩阵转换为CSR格式的函数。
    """
    return sten.SparseTensorWrapper.wrapped_from_dense(
        MyCsrTensor(tensor.to_sparse_csr()),
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

####################################################################################
#################                    自定义算子                     #################
####################################################################################
@sten.register_fwd_op_impl(
    operator=torch.mm,
    inp=(MyCooTensor, torch.Tensor),
    out=[(sten.KeepAll, torch.Tensor)],
)
def torch_mm_fwd_impl(ctx, inputs, output_sparsifiers):
    # print("coo-dense mul")
    input1, input2 = inputs
    ctx.save_for_backward(input1, input2)
    output = torch.mm(input1, input2)
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

####################################################################################
#################                    矩阵处理函数                    ################
####################################################################################
def process_matrix(matrix):
    sparse_ratio = calculate_sparse_ratio(matrix)
    print(f"稀疏比率: {sparse_ratio}%")
    if sparse_ratio > 80:
        print("转换为CSC格式")
        return convert_to_coo(matrix)
    else:
        print("转换为N:M格式")
        return convert_to_nm(matrix)


rows = 1024
cols = 1024
sparsity = 0.5
a = generate_sparse_matrix(rows, cols, sparsity)
b = torch.randn(cols, rows)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gpu_a = a.to(device)
gpu_b = b.to(device)
coo_tensor = gpu_a.to_sparse_coo()
csr_tensor = gpu_a.to_sparse_csr()
csc_tensor = gpu_a.to_sparse_csc()

# 正式运行并测量时间
warm_iterations = 100
num_iterations = 4000

# 热身运行
total_execution_time = 0
for _ in range(warm_iterations):
    torch.mm(gpu_a, gpu_b)

for i in range(num_iterations):
    start_time = time.time()
    torch.mm(gpu_a, gpu_b)
    end_time = time.time()
    execution_time = end_time - start_time
    total_execution_time += execution_time

average_execution_time = total_execution_time / num_iterations
print("torch.mm程序执行时间: ", average_execution_time, "秒")

# 热身运行
total_execution_time = 0
for _ in range(warm_iterations):
    torch.mm(coo_tensor, gpu_b)

for i in range(num_iterations):
    start_time = time.time()
    torch.mm(coo_tensor, gpu_b)
    end_time = time.time()
    execution_time = end_time - start_time
    total_execution_time += execution_time

average_execution_time = total_execution_time / num_iterations
print("coo torch.mm程序执行时间: ", average_execution_time, "秒")

# 热身运行
total_execution_time = 0
for _ in range(warm_iterations):
    torch.sparse.mm(coo_tensor, gpu_b)

for i in range(num_iterations):
    start_time = time.time()
    torch.sparse.mm(coo_tensor, gpu_b)
    end_time = time.time()
    execution_time = end_time - start_time
    total_execution_time += execution_time

average_execution_time = total_execution_time / num_iterations
print("coo torch.sparse.mm程序执行时间: ", average_execution_time, "秒")


# 热身运行
total_execution_time = 0
for _ in range(warm_iterations):
    torch.mm(csr_tensor, gpu_b)

for i in range(num_iterations):
    start_time = time.time()
    torch.mm(csr_tensor, gpu_b)
    end_time = time.time()
    execution_time = end_time - start_time
    total_execution_time += execution_time

average_execution_time = total_execution_time / num_iterations
print("csr torch.mm程序执行时间: ", average_execution_time, "秒")

# 热身运行
total_execution_time = 0
for _ in range(warm_iterations):
    torch.sparse.mm(csr_tensor, gpu_b)

for i in range(num_iterations):
    start_time = time.time()
    torch.sparse.mm(csr_tensor, gpu_b)
    end_time = time.time()
    execution_time = end_time - start_time
    total_execution_time += execution_time

average_execution_time = total_execution_time / num_iterations
print("csr torch.sparse.mm程序执行时间: ", average_execution_time, "秒")

# 热身运行
total_execution_time = 0
for _ in range(warm_iterations):
    torch.mm(csc_tensor, gpu_b)

for i in range(num_iterations):
    start_time = time.time()
    torch.mm(csc_tensor, gpu_b)
    end_time = time.time()
    execution_time = end_time - start_time
    total_execution_time += execution_time

average_execution_time = total_execution_time / num_iterations
print("csc torch.mm程序执行时间: ", average_execution_time, "秒")

# 热身运行
total_execution_time = 0
for _ in range(warm_iterations):
    torch.sparse.mm(csc_tensor, gpu_b)

for i in range(num_iterations):
    start_time = time.time()
    torch.sparse.mm(csc_tensor, gpu_b)
    end_time = time.time()
    execution_time = end_time - start_time
    total_execution_time += execution_time

average_execution_time = total_execution_time / num_iterations
print("csc torch.sparse.mm程序执行时间: ", average_execution_time, "秒")