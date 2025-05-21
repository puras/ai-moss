import torch

tensor0d = torch.tensor(1) # 从整数创建一个零维张量，标量
tensor1d = torch.tensor([1, 2, 3]) # 从列表创建一个一维张量，向量
tensor2d = torch.tensor([[1, 2], [3, 4]]) # 从嵌套的列表创建一个二维张量，矩阵
tensor3d = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]) # 从嵌套的列表创建一个三维张量

print(tensor0d.dtype)
print(tensor1d.dtype)
print(tensor2d.dtype)
print(tensor3d.dtype)

# floatevc = torch.tensor([1.0, 2.0, 3.0])
# print(floatevc.dtype)
floatevc = tensor1d.to(torch.float32)
print(floatevc.dtype)

tensor2d = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(tensor2d)
print(tensor2d.shape)
print(tensor2d.reshape(3, 2)) # 重塑张量
print(tensor2d.view(3, 2)) # 重塑张量
print(tensor2d.T) # 转置张量，沿对角线翻转
print(tensor2d.matmul(tensor2d.T)) # 矩阵相乘方法,矩阵的 (i,j) 位置元素等于左矩阵的第 i 行与右矩阵的第 j 列对应元素乘积之和
print(tensor2d @ tensor2d.T) # 与matmul相同