import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  # 연산결과 일관성
    torch.backends.cudnn.benchmark = False     # 성능보다 재현성

set_seed(42)
