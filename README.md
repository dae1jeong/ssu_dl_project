# ssu_dl_project
딥러닝 팀 프로젝트 코드 공유 

ResNet18 -> cutout, randomcrop+horizontalflip, autoaugmentation cafar10 적용 후 -> 정확도 78.29%
Resnet50 속도 너무 느려서 18로 테스트중..

ResNet50은 conv1이랑, fc만 간단하게 수정된 버전 튜닝 필요 -> 정확도 76.52%


**모델 정의 (ResNet34 + EfficientNet-B0 결합)**

- data augmemtation: Cutout 만 적용
- optimizer: Adam
- Learning rate scheduler 없음
- epochs = 75
- Loss :  CrossEntropyLoss
- Accuracy 계산방식: 일반

모델 수정사항

- data augmemtation:
1. Cutout + Cutmix(0.5) → Epoch 39: Train Acc: 71.42%, Val Loss: 35.1063, Test Acc: 74.14% 
2. Cutout +CutMix (0.3)→ Epoch 38: Train Acc: 80.20%, Val Loss: 36.7414, Test Acc: 73.47%

- optimizer:  **AdamW** (weight decay 적용)+  Learning rate scheduler 추가
1.  AdamW + CosineAnnealingLR 
→ Epoch 49: Train Acc: 77.11%, Val Loss: 32.6328, Test Acc: 77.28%
2. **AdamW + OneCycleLR(max_lr = 0.005)**

→ Epoch 75: Train Acc: 82.11%, Val Loss: 30.2754, Test Acc: 80.42% 

최대학습률(max_lr) = 0.003~ 0.01 부근에서 실험해볼 필요 있음.

- epochs = 75
- Loss : CrossEntropyLoss + CutMix

→ 두 정답 각각에 대해 CrossEntropyLoss를 구해서, *섞인 비율(lam, 1-lam)**만큼 가중치를 곱해 더함.

- Accuracy 계산방식: CutMix 반영(혼합 라벨)

예시) lam = 0.7이고 

예측이  A와 같으면 0.7점,

예측이 B와 같으면 0.3점

만약 예측이 둘 다 아니면 0점

CutMix : 두 이미지를 임의의 비율로 섞고, 라벨도 그 비율대로 섞어서 학습

- 더 다양한 데이터 분포를 학습시켜 **오버피팅을 줄이고** 모델의 일반화 성능을 높임

AdamW Optimizer: 일반화 성능 향상 및 과적합 방지를 위해 더 효과적으로 가중치 감소(regularization)를 적용 / 정확한 weight decay 적용

- **A**dam보다 더 안정적이고 좋은 최적화 결과

OneCycleLR 스케줄러 : 학습률(learning rate)을 한 번 크게 올렸다가 점점 줄이는 방식의 스케줄

- 더 많은 데이터 증강 효과를 누리고, scheduler 효과를 극대화
- 학습이 길어져도 과적합 적고, 일반화 성능이 크게 향상
