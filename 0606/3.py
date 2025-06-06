import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
from tqdm import tqdm
import timm
from torch.optim.swa_utils import AveragedModel, update_bn

# Device 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터 전처리
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761))
])

# CIFAR-100 데이터
full_train = datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
val_len = int(len(full_train) * 0.1)
train_len = len(full_train) - val_len
train_set, val_set = random_split(full_train, [train_len, val_len])
test_set_cifar = datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)

# 커스텀 테스트셋
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_list = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_list[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.img_list[idx]

test_dir = "/home/user/deep/dataset/CImages"
test_set = CustomImageDataset(test_dir, transform=test_transform)

# DataLoaders
batch_size = 256
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=4)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader_cifar = DataLoader(test_set_cifar, batch_size=batch_size, shuffle=False, num_workers=4)

# 모델 (EfficientNet-b0으로 변경)
model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=100).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
swa_model = AveragedModel(model)

# 학습
for epoch in range(50):
    model.train()
    correct = total = 0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    scheduler.step()
    train_acc = 100 * correct / total

    # Validation
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_acc = 100 * correct / total

    # Test (CIFAR-100)
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader_cifar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_acc = 100 * correct / total

    print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Test Acc: {test_acc:.2f}%")

# SWA 적용 및 저장
update_bn(train_loader, swa_model, device=device)
torch.save(swa_model.module.state_dict(), "weight_가반4조_0606_tta2.pth")

# TTA 함수
def tta_predict(model, image, device):
    model.eval()
    image = image.unsqueeze(0).to(device)  # (1, C, H, W)

    flips = [
        image,
        torch.flip(image, dims=[2]),      # H 방향 (세로)
        torch.flip(image, dims=[3]),      # W 방향 (가로)
        torch.flip(image, dims=[2, 3])    # 대각 뒤집기
    ]

    with torch.no_grad():
        preds = [model(f) for f in flips]  # 각 변형 이미지에 대해 예측
        avg_pred = torch.stack(preds).mean(0)  # 평균 내기

    return avg_pred.squeeze(0)  # (100,)

# 결과 저장
swa_model.eval()
results = []
with torch.no_grad():
    for images, filenames in tqdm(test_loader, desc="Testing"):
        for image, fname in zip(images, filenames):
            pred = tta_predict(swa_model, image.cpu(), device)
            number = int(os.path.splitext(fname)[0])
            results.append((number, int(pred.argmax())))

results.sort(key=lambda x: x[0])
with open("results_가반4조_0606_tta2.txt", "w") as f:
    f.write("number, label\n")
    for number, label in results:
        f.write(f"{str(number).zfill(4)}, {label}\n")

print("✅ TTA + AutoAug 결과 저장 완료")
