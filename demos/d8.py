from torch.utils.data import DataLoader

from demos.d6 import train_ds

train_loader = DataLoader(
    dataset=train_ds,
    batch_size=2,
    shuffle=True,
    num_workers=0,
    drop_last=True # 在每轮中丢弃最后一个批次
)

for idx, (x, y) in enumerate(train_loader):
    print(f"Batch {idx+1}: ", x, y)