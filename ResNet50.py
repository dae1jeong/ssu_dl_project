import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, models
from torch.utils.data import DataLoader

# ✅ 1. CIFAR-100 데이터 전처리 (32x32 원본 유지)
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408),
                         (0.2675, 0.2565, 0.2761))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408),
                         (0.2675, 0.2565, 0.2761))
])

train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)

# ✅ 2. Pretrained ResNet-50 불러오기
model = models.resnet50(weights='DEFAULT')

# ✅ 3. conv1을 3×3 conv 3개로 교체, maxpool 제거
model.conv1 = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
    nn.BatchNorm2d(32),
    nn.ReLU(inplace=True),
    nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
    nn.BatchNorm2d(32),
    nn.ReLU(inplace=True),
    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(inplace=True),
)
model.maxpool = nn.Identity()

# ✅ 4. 출력층 수정
model.fc = nn.Linear(2048, 100)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# ✅ 5. 학습할 파라미터만 선택 (conv1, fc만 학습)
for name, param in model.named_parameters():
    if name.startswith("conv1") or name.startswith("fc"):
        param.requires_grad = True
    else:
        param.requires_grad = False

# ✅ 6. 손실 함수, 옵티마이저
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=0.01, momentum=0.9, weight_decay=5e-4)

# ✅ 7. 학습 및 테스트 함수
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

# ✅ 8. 전체 학습 루프
best_acc = 0
for epoch in range(30):  # 충분한 성능을 위해선 50~100 epoch 이상 추천
    train(epoch)
    acc = test(epoch)
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), 'best_resnet50_partial_finetune_cifar100.pth')

print(f'최종 최고 정확도: {best_acc:.2f}%')