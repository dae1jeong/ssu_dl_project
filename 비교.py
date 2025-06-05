
def read_label_file(path):
    with open(path, "r") as f:
        lines = f.readlines()[1:]  # 첫 줄은 헤더
        label_dict = {}
        for line in lines:
            number, label = line.strip().split(",")
            number = str(int(number))  # '0001' → '1'
            label_dict[number] = int(label)
    return label_dict

# 파일 경로 설정
gt_path = "/home/user/predictions.txt"       # 정답 라벨
pred_path = "/home/user/results_가반4조_0604_1030.txt"     # 모델이 만든 예측 결과

# 라벨 읽기
gt_labels = read_label_file(gt_path)
pred_labels = read_label_file(pred_path)

# 정확도 계산
total = len(gt_labels)
correct = 0
wrong_samples = []

for number in gt_labels:
    gt = gt_labels[number]
    pred = pred_labels.get(number, -1)
    if gt == pred:
        correct += 1
    else:
        wrong_samples.append((number, gt, pred))

accuracy = correct / total * 100

# 결과 출력
print(f"✅ 예측 정확도: {accuracy:.2f}% ({correct}/{total})")
print(f"❌ 틀린 샘플 수: {len(wrong_samples)}")

# 틀린 샘플 일부 출력
print("\n예시로 틀린 10개 샘플:")
for num, gt, pred in wrong_samples[:10]:
    print(f"번호: {num.zfill(4)}, 정답: {gt}, 예측: {pred}")
