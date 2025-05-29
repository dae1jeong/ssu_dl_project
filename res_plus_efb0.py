# 필요한 라이브러리
import torch, torchvision, timm, random
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# Seed 고정
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

# Cutout 정의
class Cutout(object):
    def __init__(self, n_holes=1, length=8):
        self.n_holes = n_holes
        self.length = length
    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            mask[y1:y2, x1:x2] = 0.
        mask = torch.from_numpy(mask).expand_as(img)
        return img * mask

# CIFAR-100 전처리
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    Cutout(n_holes=1, length=8),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

# CIFAR-100 데이터셋
full_train = datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
test_set = datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)

# train/val 나누기
val_len = int(len(full_train) * 0.1)
train_len = len(full_train) - val_len
train_set, val_set = random_split(full_train, [train_len, val_len])

# DataLoader
batch_size = 256
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size)
test_loader = DataLoader(test_set, batch_size=batch_size)

# 모델 정의 (ResNet34 + EfficientNet-B0 병합)
class CombinedModel(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.resnet = models.resnet34(pretrained=True)
        self.resnet.fc = nn.Identity()
        self.efficient = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0)
        self.classifier = nn.Linear(self.resnet.fc.in_features + self.efficient.num_features, num_classes)
    def forward(self, x):
        r = self.resnet(x)
        e = self.efficient(x)
        out = torch.cat([r, e], dim=1)
        return self.classifier(out)

# 모델, 옵티마이저, 손실 함수
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CombinedModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# EarlyStopping 클래스
class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')
        self.stop = False
    def __call__(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        if self.counter >= self.patience:
            self.stop = True

early_stopping = EarlyStopping(patience=5)

# 학습 루프
epochs = 30
for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total * 100
    val_loss = 0
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}")
    early_stopping(val_loss)
    if early_stopping.stop:
        print("Early stopping triggered.")
        break
