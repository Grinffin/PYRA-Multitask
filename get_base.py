import torch

#model1 = torch.load('best_checkpoint1.pth')
#model2 = torch.load('best_checkpoint2.pth')
#model3 = torch.load('/data/guoyufei21/4.pth')


# 加载 .pth 文件
model = torch.load('/data/guoyufei21/240919_ViT-B_prompt_lora_8_plain_lora/svhn_lr-0.001_wd-0.0001/best_checkpoint.pth', map_location='cuda')
base_model = model.copy()
#model['pos_embed'] = model['pos_embed'][:, 1:, :]
#torch.save(model, 'base_model.pth')
# 遍历 model 中的参数
total_params = 0
for name, param in model['model'].items():  # 假设 'model' 是存储模型参数的部分
    if 'LoRA' in name or 'pyra' in name:
        print(name)
        # 获取当前张量的形状
        tensor_shape = model['model'][name].shape
        # 创建一个相同形状的全0张量
        zero_tensor = torch.zeros(tensor_shape)
        base_model['model'][name] = zero_tensor
        print(f"Parameter name: {name}, item: {base_model['model'][name]}")
    total_params += param.numel()

print(f"Total number of parameters: {total_params}")

output_path = 'base_model_best_checkpoint.pth'
torch.save(base_model, output_path)
print(f'basemodel已保存为{output_path}')