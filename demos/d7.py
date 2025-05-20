import torch
from torch.utils.data import DataLoader

from demos.d6 import train_ds, test_ds

torch.manual_seed(123)

train_loader = DataLoader(
    dataset=train_ds,
    batch_size=2,
    shuffle=True, # 是否打乱顺序
    num_workers=0 # 设置 num_workers=4 通常会在许多真实世界数据集上获得最佳性能，但最佳设置取决于你的硬件和用于加载 Dataset 类中训练示例的代码
)

test_loader = DataLoader(
    dataset=test_ds,
    batch_size=2,
    shuffle=False,
    num_workers=0
)

for idx, (x, y) in enumerate(train_loader):
    print(f"Batch {idx+1}: ", x, y)

print('--')

for idx, (x, y) in enumerate(test_loader):
    print(f"Batch {idx+1}: ", x, y)