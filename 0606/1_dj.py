import torch, torchvision, timm, random
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, random_split, Dataset
from tqdm import tqdm
import os
from PIL import Image
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torchvision.transforms import AutoAugment, AutoAugmentPolicy

def freeze_feature_extractors(model):
    for param in model.resnet.parameters():
        param.requires_grad = False
    for param in model.efficient.parameters():
        param.requires_grad = False

def unfreeze_feature_extractors(model):
    for param in model.resnet.parameters():
        param.requires_grad = True
    for param in model.efficient.parameters():
        param.requires_grad = True


class EarlyStopping:
    def __init__(self, patience=20):
        self.patience = patience
        self.counter = 0
        self.best_acc = 0.0
        self.early_stop = False

    def __call__(self, val_acc):
        if val_acc > self.best_acc:
            self.best_acc = val_acc
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True



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

# CutMix 함수
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x_cutmix = x.clone()
    x_cutmix[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    y_a, y_b = y, y[index]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x_cutmix, y_a, y_b, lam

# 전처리
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # ← 핵심
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    AutoAugment(policy=AutoAugmentPolicy.CIFAR10),
    transforms.ToTensor(),
    Cutout(n_holes=1, length=8),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
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

# 모델
class CombinedModel(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()

        self.resnet = models.resnet34(weights='DEFAULT')
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.bn1 = nn.BatchNorm2d(64)
        self.resnet.relu = nn.ReLU(inplace=True)
        self.resnet.maxpool = nn.Identity()
        self.resnet.fc = nn.Identity()

        self.efficient = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0)

        # ✅ 각 feature에 BatchNorm + ReLU (중요)
        self.bn_r = nn.Sequential(
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
        )
        self.bn_e = nn.Sequential(
            nn.BatchNorm1d(self.efficient.num_features),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 + self.efficient.num_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes))



    def forward(self, x):
        r = self.resnet(x)
        e = self.efficient(x)
        r = self.bn_r(r)
        e = self.bn_e(e)


        out = torch.cat([r, e], dim=1)
        return self.classifier(out)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CombinedModel().to(device)

for param in model.parameters():
    param.requires_grad = True

unfreeze_epoch = 10

for epoch in range(epochs):
    # 🔓 일정 시점에 전체 fine-tuning 시작
    if epoch == unfreeze_epoch:
        print(f"[INFO] Unfreezing feature extractors at epoch {epoch}")
        unfreeze_feature_extractors(model)

freeze_feature_extractors(model)

 # 🔁 optimizer 재정의
decay, no_decay = [], []
for name, param in model.named_parameters():
    if param.requires_grad:
        if "bias" in name or "bn" in name or "norm" in name:
            no_decay.append(param)
        else:
            decay.append(param)
optimizer = optim.AdamW([
    {'params': decay, 'weight_decay': 0.01},
    {'params': no_decay, 'weight_decay': 0.0}
], lr=0.003)

epochs = 200
scheduler = OneCycleLR(
    optimizer,
    max_lr=0.005,
    steps_per_epoch=len(train_loader),
    epochs=epochs,
    pct_start=0.2,
    anneal_strategy='cos'
)

criterion = nn.CrossEntropyLoss(label_smoothing=0.2)
early_stopper = EarlyStopping(patience=20)
swa_start = int(epochs * 0.75)
swa_model = AveragedModel(model)
swa_scheduler = SWALR(optimizer, swa_lr=0.0005)

# Accuracy 평가
def eval_acc(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in data_loader:
            images = batch[0].to(device)
            labels = batch[1]
            if torch.is_tensor(labels):
                labels = labels.to(device)
            else:
                continue
            outputs = model(images)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total * 100 if total > 0 else None


# 학습 루프
for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        images, labels = images.to(device), labels.to(device)
        r = np.random.rand(1)
        if r < 0.5:
            images, targets_a, targets_b, lam = cutmix_data(images, labels, alpha=1.0)
            outputs = model(images)
            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
            _, preds = outputs.max(1)
            correct += (lam * (preds == targets_a).sum().item() + (1 - lam) * (preds == targets_b).sum().item())
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch >= swa_start:
            swa_scheduler.step()
        else:
            scheduler.step()
        total_loss += loss.item()
        total += labels.size(0)
    train_acc = correct / total * 100
    val_acc = eval_acc(model, val_loader, device)
    test_acc = eval_acc(model, test_loader_cifar, device)
    print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Test Acc: {test_acc:.2f}%")

        # EarlyStopping 체크
    early_stopper(val_acc)
    if early_stopper.early_stop:
        print(f"⏹️ Early stopping at epoch {epoch+1} due to no improvement in validation accuracy.")
        break



    if epoch >= swa_start:
        swa_model.update_parameters(model)

# SWA 저장
update_bn(train_loader, swa_model, device=device)
torch.save(swa_model.module.state_dict(), "weight_가반4조_0606_tta2.pth")

# TTA 예측
# (1) tta_tf_identity: 이미 정규화 완료된 Tensor를 그대로 반환
def tta_tf_identity(x: torch.Tensor) -> torch.Tensor:
    # x는 이미 (3×32×32) 형태로 Normalize까지 된 Tensor
    return x

# (2) tta_tf_hflip: 이미 Tensor 상태를 수평뒤집기만 수행
def tta_tf_hflip(x: torch.Tensor) -> torch.Tensor:
    # torch.flip에서 dims=[2]는 width(32) 축을 뒤집는 의미
    return torch.flip(x, dims=[2])

tta_transforms = [tta_tf_identity, tta_tf_hflip]

def tta_predict(model: nn.Module, image_tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        outputs = []
        for tf in tta_transforms:
            # tf(image_tensor)은 (3×32×32) 형태의 Tensor를 반환
            img = tf(image_tensor).unsqueeze(0)  # (1×3×32×32)
            img = torch.nn.functional.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)
            img = img.to(device)
            out = model(img)                              # (1×num_classes)
            outputs.append(torch.softmax(out, dim=1))     # (1×num_classes)
        # 두 개 결과를 평균 냄 → 최종 (1×num_classes) 확률
        return torch.stack(outputs, dim=0).mean(0)        # (1×num_classes)

# 커스텀 테스트셋 결과 저장
swa_model.eval()
results = []

with torch.no_grad():
    for images, filenames in tqdm(test_loader, desc="Testing"):
        for image, fname in zip(images, filenames):
            pred = tta_predict(swa_model, image.cpu(), device)
            number = int(os.path.splitext(fname)[0])
            results.append((number, int(pred.argmax())))

# 정렬 및 저장
results.sort(key=lambda x: x[0])
with open("results_가반4조_0606_tta2.txt", "w") as f:
    f.write("number, label\n")
    for number, label in results:
        f.write(f"{str(number).zfill(4)}, {label}\n")

print("✅ TTA + AutoAug 결과 저장 완료")
