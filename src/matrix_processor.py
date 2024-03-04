import torch

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

def convert_to_nm(matrix):
    """
    将密集矩阵转换为N:M格式的示例函数。
    注意：这是一个占位函数，实际转换需要特定的实现。
    """
    # 这里仅返回原始矩阵作为演示，实际上N:M转换需要专门的逻辑和可能的硬件支持
    return matrix

def process_matrix(matrix):
    sparse_ratio = calculate_sparse_ratio(matrix)
    print(f"稀疏比率: {sparse_ratio}%")
    if sparse_ratio > 75:
        print("转换为CSR格式")
        return matrix.to_sparse_csr()
    else:
        print("转换为N:M格式")
        return convert_to_nm(matrix)


# 示例矩阵
matrix = torch.tensor([[0, 0, 3, 0, 0], [0, 0, 0, 4, 0], [0, 2, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=torch.float32)

# 处理矩阵
sparse_matrix = process_matrix(matrix)
print(sparse_matrix)