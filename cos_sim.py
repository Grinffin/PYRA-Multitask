import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


model_path_list_lora = [
                    '/data/guoyufei21/outputs/240910_ViT-B_prompt_lora_8_compress_low_PYRA_RUN_/natural_lr-0.001_wd-0.0001_pyra_lr-3e-2_modify_K_7_100epoch/best_checkpoint.pth'
                    ]
models = [torch.load(path) for path in model_path_list_lora]
# 初始化存储结果的列表
flattened_lora_params = []

# 遍历所有模型
for model in models:
    for name in model['model']:
        if 'pyra' in name:
            print(name)
            print(model['model'][name])
