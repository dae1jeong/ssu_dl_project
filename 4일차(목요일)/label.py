import os
import zipfile
from PIL import Image
import numpy as np
import torch
from torchvision import transforms, datasets, models
from tqdm import tqdm
import shutil

# 1. 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CIFAR_PATH = "./cifar100"
CIMAGES_ZIP_PATH = "./CImages.zip"
CIMAGES_EXTRACT_PATH = "./CImages"
SUBMISSION_PATH = "submission.txt"

# 2. 전처리 정의 (ResNet에 맞춤)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 3. ResNet18 로딩
resnet = models.resnet18(pretrained=True)
resnet.fc = torch.nn.Identity()
resnet = resnet.to(DEVICE)
resnet.eval()

# 4. CIFAR-100 train dataset 로드 및 특징 추출
print("▶ CIFAR-100 특징 추출 중...")
cifar100 = datasets.CIFAR100(root=CIFAR_PATH, train=True, download=True, transform=transform)
cifar_loader = torch.utils.data.DataLoader(cifar100, batch_size=512, shuffle=False)

cifar_feats = []
cifar_labels = []

with torch.no_grad():
    for images, labels in tqdm(cifar_loader):
        images = images.to(DEVICE)
        feats = resnet(images).cpu().numpy()
        cifar_feats.append(feats)
        cifar_labels.extend(labels.cpu().numpy())

cifar_feats = np.vstack(cifar_feats)
cifar_labels = np.array(cifar_labels)

# 5. CImages 압축 해제
print("▶ CImages 압축 해제 중...")
if os.path.exists(CIMAGES_EXTRACT_PATH):
    shutil.rmtree(CIMAGES_EXTRACT_PATH)
with zipfile.ZipFile(CIMAGES_ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall(CIMAGES_EXTRACT_PATH)

# 6. CImage 특징 추출 및 가장 유사한 CIFAR 라벨 매핑
print("▶ CImages 라벨 매핑 중...")
cimage_folder = os.path.join(CIMAGES_EXTRACT_PATH, "CImages")
cimage_files = sorted(os.listdir(cimage_folder), key=lambda x: int(x.split('.')[0]))

results = []

with torch.no_grad():
    for fname in tqdm(cimage_files):
        img_path = os.path.join(cimage_folder, fname)
        image = Image.open(img_path).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(DEVICE)
        feat = resnet(tensor).cpu().numpy()

        # 가장 유사한 CIFAR 특징과 비교 (cosine similarity)
        sim = np.dot(cifar_feats, feat.T).squeeze()
        best_idx = np.argmax(sim)
        label = cifar_labels[best_idx]

        number = int(os.path.splitext(fname)[0])
        results.append((f"{number:04d}", label))

# 7. 결과 저장
print("▶ 결과 저장 중...")
with open(SUBMISSION_PATH, "w") as f:
    f.write("number, label\n")
    for number, label in results:
        f.write(f"{number}, {label}\n")

print(f"✅ 완료! 결과가 {SUBMISSION_PATH}에 저장되었습니다.")
