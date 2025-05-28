import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, models
from torch.utils.data import DataLoader
import numpy as np
import random #seed고정해야되나?
from PIL import Image

class Cutout(object):
    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)

        for _ in range(self.n_holes):
            y = random.randint(0, h - 1)
            x = random.randint(0, w - 1)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.

        mask = torch.from_numpy(mask).expand_as(img)
        return img * mask

# 1. 데이터 전처리 (224로 리사이즈 필요 없음, 32x32 그대로 사용)
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    Cutout(n_holes=1, length=8)  # Cutout 적용 (마스킹할 영역 수, 한 변의 길이(정사각형))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)

# 2. Pretrained ResNet-18 불러오기
model = models.resnet18(weights='DEFAULT')

# 3. conv1(7x7)과 maxpool을 3x3 conv 3개 + Identity로 교체
model.conv1 = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
    nn.BatchNorm2d(32), #레이어 출력을 정규화(32:해당 레이어 채널수)
    #각 채널 별로 독립적으로 배치 내 평균 /표준편차를 사용해 정규화
    nn.ReLU(inplace=True),
    nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
    nn.BatchNorm2d(32),
    nn.ReLU(inplace=True),
    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(inplace=True),
)
model.maxpool = nn.Identity()  # maxpool 제거

# 4. 출력층 수정
model.fc = nn.Linear(512, 100)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)


for name, param in model.named_parameters():
    if name.startswith("conv1") or name.startswith("fc"):
        param.requires_grad = True
    else:
        param.requires_grad = False
# 5. 손실함수, 옵티마이저
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

# 6. 학습 및 테스트 함수
def train(epoch):
    model.train()
    total, correct, running_loss = 0, 0, 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    acc = 100. * correct / total
    print(f'[Train] Epoch {epoch} | Loss: {running_loss:.3f} | Acc: {acc:.2f}%')

def test(epoch):
    model.eval()
    total, correct, test_loss = 0, 0, 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 100. * correct / total
    print(f'[Test ] Epoch {epoch} | Loss: {test_loss:.3f} | Acc: {acc:.2f}%')
    return acc

# 7. 전체 학습 루프
best_acc = 0
for epoch in range(30):  # 실제론 50~100 epoch 이상 돌려야 함!
    train(epoch)
    acc = test(epoch)
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), 'best_resnet18_cifar100_customconv.pth')

print(f'최종 최고 정확도: {best_acc:.2f}%')