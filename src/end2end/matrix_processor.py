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
#################                   功能测试代码                    #################
####################################################################################
# # 测试将矩阵转为CSC
# matrix = torch.tensor([[0, 0, 3, 0, 0], 
#                        [0, 0, 0, 4, 0], 
#                        [0, 2, 0, 0, 0], 
#                        [0, 0, 0, 0, 0], 
#                        [0, 0, 0, 0, 0]], dtype=torch.float32)
# sparse_matrix = process_matrix(matrix)
# a = torch.rand(5,5)
# print(sparse_matrix)
# spase_result = torch.mm(sparse_matrix, a)
# dense_result = torch.mm(matrix, a)
# print("sparse result: ", spase_result)
# print("dense result: ", dense_result)


# # 测试将矩阵转为NM
# a = torch.randn(768, 768, requires_grad=True)
# b = torch.randn(768, 768, requires_grad=True)
# sparse_matrix1 = process_matrix(a)
# spase_result1 = torch.mm(sparse_matrix1, b)
# dense_result1 = torch.mm(a, b)
# print("sparse result1: ", spase_result1)
# print("dense result1: ", dense_result1)


####################################################################################
#################                   性能测试代码                    #################
####################################################################################
def fill_nonzero_elements(args):
    matrix, nonzero_indices, rows, cols = args
    for idx in nonzero_indices:
        row_idx = idx // cols
        col_idx = idx % cols
        # 生成非零元素的值
        value = torch.rand(1)
        # 填充非零元素
        matrix[row_idx, col_idx] = value
    return matrix

def pral_generate_sparse_matrix(rows, cols, sparsity, num_threads=4):
    # 计算非零元素的数量
    num_nonzeros = int(rows * cols * (1 - sparsity))

    # 生成非零元素的随机索引
    nonzero_indices = np.random.choice(rows * cols, num_nonzeros, replace=False)

    # 创建一个全零张量
    matrix = torch.zeros((rows, cols))

    # 准备参数列表
    args_list = [(matrix, nonzero_indices[i::num_threads], rows, cols) for i in range(num_threads)]

    # 使用线程池并行填充非零元素
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(fill_nonzero_elements, args_list))

    # 合并结果
    final_matrix = sum(results)

    return final_matrix

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

# 测试CSC矩阵乘法，scipy,np,torch.mm哪种实现方式更快
nums = 1024
sparsity = 0.5
a = generate_sparse_matrix(nums, nums, sparsity)
b = torch.randn(nums, nums)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # 转化为CSC格式
# csc_tensor = scipy.sparse.csc_matrix(a)

# # 转化为CSR格式
# csr_tensor = scipy.sparse.csr_matrix(a)

# # 转化为COO格式
# nonzero_indices = a.nonzero().t()
# values = a[nonzero_indices[0], nonzero_indices[1]]
# coo_tensor = torch.sparse_coo_tensor(nonzero_indices, values, a.size())

# 转移至gpu上
gpu_a = a.to(device)
gpu_b = b.to(device)
# nm_tensor = convert_to_nm(a)
# gpu_coo_tensor = coo_tensor.to(device)

# # 转化为NM格式
# start_time = time.time()
print(1)
nm_tensor = convert_to_nm(gpu_a)
print(1)
# end_time = time.time()
# execution_time = end_time - start_time
# print("convert time: ", execution_time, "秒")

# 正式运行并测量时间
total_execution_time = 0
warm_iterations = 100
num_iterations = 10000

# # 热身运行
# for _ in range(warm_iterations):
#     torch.mm(a, b)

# for i in range(num_iterations):
#     start_time = time.time()
#     torch.mm(a, b)
#     end_time = time.time()
#     execution_time = end_time - start_time
#     total_execution_time += execution_time

# average_execution_time = total_execution_time / num_iterations
# print("cpu 普通torch.mm程序执行时间: ", average_execution_time, "秒")


# # 热身运行
# total_execution_time = 0
# for _ in range(warm_iterations):
#     torch.mm(gpu_a,gpu_b)

# for i in range(num_iterations):
#     start_time = time.time()
#     torch.mm(gpu_a,gpu_b)
#     torch.cuda.synchronize()
#     end_time = time.time()
#     execution_time = end_time - start_time
#     total_execution_time += execution_time

# average_execution_time = total_execution_time / num_iterations
# print("gpu 普通torch.mm程序执行时间: ", average_execution_time, "秒")


# # 热身运行
# total_execution_time = 0
# for _ in range(warm_iterations):
#     torch.mm(coo_tensor, b)

# for i in range(num_iterations):
#     start_time = time.time()
#     torch.mm(coo_tensor, b)
#     end_time = time.time()
#     execution_time = end_time - start_time
#     total_execution_time += execution_time

# average_execution_time = total_execution_time / num_iterations
# print("cpu COO pytorch程序执行时间: ", average_execution_time, "秒")

# # 热身运行
# total_execution_time = 0
# for _ in range(warm_iterations):
#     torch.mm(gpu_coo_tensor, gpu_b)

# for i in range(num_iterations):
#     start_time = time.time()
#     torch.mm(gpu_coo_tensor, gpu_b)
#     end_time = time.time()
#     execution_time = end_time - start_time
#     total_execution_time += execution_time

# average_execution_time = total_execution_time / num_iterations
# print("gpu COO pytorch程序执行时间: ", average_execution_time, "秒")

# # 热身运行
# total_execution_time = 0
# for _ in range(warm_iterations):
#     torch.mm(nm_tensor, b)

# for i in range(num_iterations):
#     start_time = time.time()
#     torch.mm(nm_tensor, gpu_b)
#     end_time = time.time()
#     execution_time = end_time - start_time
#     total_execution_time += execution_time

# average_execution_time = total_execution_time / num_iterations
# print("spatha程序执行时间: ", average_execution_time, "秒")


result = torch.mm(nm_tensor, gpu_b)
print(1)
print("nm_tensor: ", nm_tensor.wrapped_tensor.to_dense())


# start_time = time.time()
# torch.mm(a, b)
# end_time = time.time()
# execution_time = end_time - start_time
# print("cpu 普通torch.mm程序执行时间: ", execution_time, "秒")

# start_time = time.time()
# torch.mm(gpu_a,gpu_b)
# torch.cuda.synchronize()
# end_time = time.time()
# execution_time = end_time - start_time
# print("gpu 普通torch.mm程序执行时间: ", execution_time, "秒")

# start_time = time.time()
# torch.mm(coo_tensor, b)
# end_time = time.time()
# execution_time = end_time - start_time
# print("cpu COO pytorch程序执行时间: ", execution_time, "秒")

# start_time = time.time()
# torch.mm(gpu_coo_tensor, gpu_b)
# torch.cuda.synchronize()
# end_time = time.time()
# execution_time = end_time - start_time
# print("gpu COO pytorch程序执行时间: ", execution_time, "秒")

# start_time = time.time()
# csc_tensor.dot(b)
# end_time = time.time()
# execution_time = end_time - start_time
# print("CSC scipy程序执行时间: ", execution_time, "秒")

# start_time = time.time()
# csr_tensor.dot(b)
# end_time = time.time()
# execution_time = end_time - start_time
# print("CSR scipy程序执行时间: ", execution_time, "秒")

# start_time = time.time()
# torch.mm(nm_tensor, gpu_b)
# torch.cuda.synchronize()
# end_time = time.time()
# execution_time = end_time - start_time
# print("spatha程序执行时间: ", execution_time, "秒")

# #测试spatha和普通torch.mm哪种实现方式更快
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
# a = generate_sparse_matrix(4096, 4096, 0.5)
# b = torch.randn(4096, 4096)
# a.to(device)
# b.to(device)
# test_tensor = process_matrix(a)

# start_time = time.time()
# torch.mm(a,b)
# end_time = time.time()
# execution_time = end_time - start_time
# print("普通torch.mm程序执行时间: ", execution_time, "秒")

# start_time = time.time()
# torch.mm(test_tensor, b)
# end_time = time.time()
# execution_time = end_time - start_time
# print("spatha程序执行时间: ", execution_time, "秒")

