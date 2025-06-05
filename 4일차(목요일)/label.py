import os
from PIL import Image
import numpy as np
import torch
from torchvision import transforms, datasets, models
from tqdm import tqdm

# ==========================
# 1. 경로 설정
# ==========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CIFAR_PATH = "./dataset/cifar100"
CIMAGES_FOLDER = "./dataset/CImages"
SUBMISSION_PATH = "./submission.txt"

# ==========================
# 2. 이미지 전처리 및 모델 로딩
# ==========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 사전학습된 ResNet18 (마지막 분류기 제거)
resnet = models.resnet18(pretrained=True)
resnet.fc = torch.nn.Identity()
resnet = resnet.to(DEVICE)
resnet.eval()

# ==========================
# 3. CIFAR-100 train 불러오기 & 특징 추출
# ==========================
print("▶ CIFAR-100 특징 추출 중...")
cifar100 = datasets.CIFAR100(root=CIFAR_PATH, train=True, download=True, transform=transform)
cifar_loader = torch.utils.data.DataLoader(cifar100, batch_size=512, shuffle=False)

cifar_feats = []
cifar_labels = []

with torch.no_grad():
    for images, labels in tqdm(cifar_loader, desc="CIFAR-100"):
        images = images.to(DEVICE)
        feats = resnet(images).cpu().numpy()
        cifar_feats.append(feats)
        cifar_labels.extend(labels.cpu().numpy())

cifar_feats = np.vstack(cifar_feats)  # (50000, 512)
cifar_labels = np.array(cifar_labels)

# ==========================
# 4. CImages 특징 추출 및 유사도 기반 라벨 추정
# ==========================
print("▶ CImages 라벨 추정 중...")
cimage_files = sorted(os.listdir(CIMAGES_FOLDER), key=lambda x: int(x.split('.')[0]))

results = []

with torch.no_grad():
    for fname in tqdm(cimage_files, desc="CImages"):
        img_path = os.path.join(CIMAGES_FOLDER, fname)
        image = Image.open(img_path).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(DEVICE)
        feat = resnet(tensor).cpu().numpy()  # (1, 512)

        # cosine similarity 계산
        sim = np.dot(cifar_feats, feat.T).squeeze()  # (50000,)
        best_idx = np.argmax(sim)
        label = cifar_labels[best_idx]

        number = int(os.path.splitext(fname)[0])
        results.append((f"{number:04d}", label))

# ==========================
# 5. submission.txt 저장
# ==========================
print("▶ 결과 저장 중...")

with open(SUBMISSION_PATH, "w") as f:
    f.write("number, label\n")
    for number, label in results:
        f.write(f"{number}, {label}\n")

print(f"✅ 완료! 결과 파일: {SUBMISSION_PATH}")
