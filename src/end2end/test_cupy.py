from concurrent.futures import ThreadPoolExecutor
import time
import cupy
import cupyx
import numpy as np
import torch

# 得在base下跑

# # 创建一个稀疏矩阵
# # 例如，我们使用一个随机稀疏矩阵
# # 这里我们创建一个大小为 1000x1000 的稀疏矩阵，大约有1%的元素是非零的
# data = cp.random.rand(10000)  # 假设有10000个非零元素
# rows = cp.random.randint(0, 1000, size=10000)  # 非零元素的行索引
# cols = cp.random.randint(0, 1000, size=10000)  # 非零元素的列索引
# sparse_matrix = cupyx.scipy.sparse.coo_matrix((data, (rows, cols)), shape=(1000, 1000))

# # 创建一个稠密矩阵
# dense_matrix = cp.random.rand(1000, 1000)

# # 执行稀疏-稠密矩阵乘法
# # 注意：稀疏矩阵乘法通常返回稠密结果
# for i in range(1000):
#     result_matrix = sparse_matrix.dot(dense_matrix)

# # 查看结果的一小部分
# print(result_matrix[:5, :5])

# result = cupy.dot(gpu_a_cupy, gpu_b_cupy)
# cupy_result = torch.from_dlpack(result.toDlpack())
# torch_result = torch.mm(gpu_a, gpu_b)
# print(torch.equal(cupy_result, torch_result))

# # 注意：对于稀疏矩阵 gpu_a 的处理会更复杂，因为直接的DLPack转换可能不支持
# # 你可能需要将稀疏矩阵转换为稠密形式，或者找到一种方法将其直接转换为CuPy支持的稀疏矩阵格式

# # 假设你已经有了 gpu_a 和 gpu_b 的CuPy版本，名为 gpu_a_cupy 和 gpu_b_cupy
# # 你可以直接使用 CuPy 进行矩阵乘法操作
# result = cupy.dot(gpu_a_cupy, gpu_b_cupy)
# print(type(result))
# print(result.get())  # 将result转换为NumPy数组并打印

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

nums = 4096
sparsity = 0.75
a = pral_generate_sparse_matrix(nums, nums, sparsity)
b = torch.randn(nums, nums)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gpu_a = a.to(device)
gpu_b = b.to(device)
gpu_a_cupy = cupy.fromDlpack(torch.to_dlpack(gpu_a))
gpu_b_cupy = cupy.fromDlpack(torch.to_dlpack(gpu_b))

sparse_a = cupyx.scipy.sparse.coo_matrix(gpu_a_cupy)


# 正式运行并测量时间
total_execution_time = 0
warm_iterations = 50
num_iterations = 500

print("test begin")

# 热身运行
total_execution_time = 0
for _ in range(warm_iterations):
    sparse_a.dot(gpu_b_cupy)

for i in range(num_iterations):
    # 在开始测量前同步
    cupy.cuda.Stream.null.synchronize()
    start_time = time.time()
    
    # 执行矩阵乘法
    sparse_a.dot(gpu_b_cupy)
    
    # 在结束测量前同步，确保矩阵乘法已经完成
    cupy.cuda.Stream.null.synchronize()
    end_time = time.time()

    execution_time = end_time - start_time
    total_execution_time += execution_time

average_execution_time = total_execution_time / num_iterations
print("csr cupy普通执行时间: ", average_execution_time, "秒")

