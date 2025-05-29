from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# 1. CIFAR-100 불러오기 (Train만)
transform = transforms.ToTensor()
full_train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)

# 2. 데이터셋 크기 확인 및 분할 비율 설정
total_size = len(full_train_dataset)  # 50000
val_ratio = 0.2
val_size = int(total_size * val_ratio)
train_size = total_size - val_size

# 3. 고정된 seed로 데이터 분할 (재현성 보장)
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

# 4. DataLoader로 변환
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
