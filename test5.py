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

# CutMix
def rand_bbox(size, lam):
    W, H = size[2], size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    bbx1, bbx2 = np.clip(cx - cut_w // 2, 0, W), np.clip(cx + cut_w // 2, 0, W)
    bby1, bby2 = np.clip(cy - cut_h // 2, 0, H), np.clip(cy + cut_h // 2, 0, H)
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

# 데이터셋 전처리
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
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

# CIFAR-100 데이터셋
full_train = datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
val_len = int(len(full_train) * 0.1)
train_len = len(full_train) - val_len
train_set, val_set = random_split(full_train, [train_len, val_len])

test_set_cifar = datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)


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

# DataLoader 설정
batch_size = 256
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=4)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader_cifar = DataLoader(test_set_cifar, batch_size=batch_size, shuffle=False, num_workers=4)

        
# 모델 정의
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
        self.classifier = nn.Linear(512 + self.efficient.num_features, num_classes)

    def forward(self, x):
        r = self.resnet(x)
        e = self.efficient(x)
        out = torch.cat([r, e], dim=1)
        return self.classifier(out)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CombinedModel().to(device)

decay, no_decay = [], []
for name, param in model.named_parameters():
    if "bias" in name or "bn" in name or "norm" in name:
        no_decay.append(param)
    else:
        decay.append(param)

optimizer = optim.AdamW([
    {'params': decay, 'weight_decay': 0.001},
    {'params': no_decay, 'weight_decay': 0.0}
], lr=0.001)

epochs = 200
from torch.optim.lr_scheduler import OneCycleLR
scheduler = OneCycleLR(
    optimizer, 
    max_lr=0.005, 
    steps_per_epoch=len(train_loader), 
    epochs=epochs, 
    pct_start=0.2, 
    anneal_strategy='cos')

criterion = nn.CrossEntropyLoss(label_smoothing=0.0)

class EarlyStopping:
    def __init__(self, patience=7):
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

early_stopping = EarlyStopping(patience=7)

# 평가함수
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
    total_loss, correct, total = 0, 0, 0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        images, labels = images.to(device), labels.to(device)
        r = np.random.rand(1)
        if r < 0.2:
            images, targets_a, targets_b, lam = cutmix_data(images, labels)
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
        scheduler.step()
        total_loss += loss.item()
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


    val_acc = eval_acc(model, val_loader, device)
    test_acc = eval_acc(model, test_loader_cifar, device)
    print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Test Acc: {test_acc:.2f}%")
    early_stopping(val_loss)
    if early_stopping.stop:
        print("Early stopping triggered.")
        break

# 모델 저장
torch.save(model.state_dict(), "weight_가반4조_0604.pth")
print("✅ 모델 저장 완료.pth")

result =[]

with torch.no_grad():
    for images, filenames in tqdm(test_loader, desc="testing"):
        images = images.to(device)
        outputs = model(images)
        _, preds = outputs.max(1)
        for fname, pred in zip(filenames, preds):
            number = int(os.path.splitext(fname)[0])
            result.append((number, int(pred)))
            
result = sorted(result, key=lambda x: x[0])
with open("result_가반4조_0604.txt", "w") as f:
    f.write("number,label\n")
    for number, label in result:
        f.write(f"{str(number).zfill(4)},{label}\n")
        
print("✅ 결과 저장 완료: result_가반4조_0604.txt")