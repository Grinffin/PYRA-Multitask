import torch

import torch

def ties_merge_model_weights(model_path_list, output_path, init_model_path, k, λ, device='cuda'):
    # 加载所有模型的状态字典
    models = [torch.load(path, map_location=device) for path in model_path_list]
    init_model = torch.load(init_model_path, map_location=device)  # 初始化模型
    avg_model = models[0].copy()  # 拷贝模型结构
    num_models = len(models)

    # 第一步：计算任务向量 τ_t = θ_t - θ_init
    task_vectors = []
    for model in models:
        task_vector = {name: model['model'][name].to(device) - init_model['model'][name].to(device) for name in model['model'] 
                       if "LoRA" in name or "pyra" in name}
        task_vectors.append(task_vector)
        

    # 第二步：修剪冗余参数，保留全局 Top-K
    def keep_global_topk_reset_rest_to_zero(task_vectors, k):
        # 将所有模块的参数展平并连接成一个大张量
        all_params = torch.cat([task_vector[name].view(-1) for task_vector in task_vectors for name in task_vector])

        # 计算全局总的元素数量
        total_elements = all_params.numel()
        k = int(total_elements * k)  # 计算需要保留的元素个数

        # 获取绝对值 Top-K 的元素和对应的阈值
        topk_values, _ = torch.topk(all_params.abs(), k)
        threshold = topk_values[-1]

        # 基于阈值创建掩码（保留大于等于阈值的元素）
        mask = all_params.abs() >= threshold

        # 创建一个与原始张量形状相同的掩码张量
        split_sizes = [task_vector[name].numel() for task_vector in task_vectors for name in task_vector]
        mask_split = torch.split(mask, split_sizes)

        # 将掩码应用回原始 task_vectors 的每个模块中
        trimmed_task_vectors = []
        mask_idx = 0
        for task_vector in task_vectors:
            trimmed_vector = {}
            for name in task_vector:
                param_tensor = task_vector[name].view(-1)
                mask_tensor = mask_split[mask_idx].view_as(param_tensor)
                param_tensor *= mask_tensor.float()  # 将不在Top-K中的参数置为0
                trimmed_vector[name] = param_tensor.view_as(task_vector[name])
                mask_idx += 1
            trimmed_task_vectors.append(trimmed_vector)

        return trimmed_task_vectors


    # 使用新的修剪函数
    trimmed_task_vectors = keep_global_topk_reset_rest_to_zero(task_vectors, k)

    # 第三步：选择最终符号
    gamma_m = {}
    for name in task_vectors[0]:
        sign_sum = sum(trimmed_task_vector[name] for trimmed_task_vector in trimmed_task_vectors)
        gamma_m[name] = torch.sign(sign_sum)

    # 第四步：分离式融合
    merged_task_vector = {}
    for name in task_vectors[0]:
        # 获取与该名字关联的 gamma_m 的符号
        gamma_sign = torch.sign(gamma_m[name])
        # 获取与该名字关联的所有 trimmed_task_vectors 的符号
        task_signs = torch.stack([torch.sign(trimmed_task_vector[name]) for trimmed_task_vector in trimmed_task_vectors])
    
        # 初始化计数和和为0的张量，保持与参数维度一致
        summed_values = torch.zeros_like(gamma_m[name], dtype=torch.float32)
        count_values = torch.zeros_like(gamma_m[name], dtype=torch.float32)
    
        # 遍历每个 trimmed_task_vector
        for i, trimmed_task_vector in enumerate(trimmed_task_vectors):
            # 获取当前张量
            current_vector = trimmed_task_vector[name]
            #if i == 6:
            #    current_vector = current_vector*2
            # 逐元素比较符号并取匹配的位置
            matching_mask = (task_signs[i] == gamma_sign)
            # 在匹配的位置上累加对应的值
            summed_values[matching_mask] += current_vector[matching_mask]
            # 在匹配的位置上计数
            count_values[matching_mask] += 1
            #if i == 6:
            #    count_values[matching_mask] += 1


        # 计算平均值，避免除以0（未匹配的地方）
        merged_task_vector[name] = summed_values / torch.where(count_values == 0, torch.ones_like(count_values), count_values)
        # 对于没有匹配的元素，设为0（可选）
        merged_task_vector[name][count_values == 0] = 0

    # 最终模型：θ_m = θ_init + λ * τ_m
    for name in merged_task_vector:
        avg_model['model'][name] = init_model['model'][name].to(device) + λ * merged_task_vector[name].to(device)

    # 对 head.weight 和 head.bias 进行拼接
    # for name in models[0]['model']:
    #     if "head.weight" in name:
    #         weights = [model['model'][name] for model in models]
    #         avg_model['model'][name] = torch.cat(weights, dim=0)
    #     elif "head.bias" in name:
    #         biases = [model['model'][name] for model in models]
    #         avg_model['model'][name] = torch.cat(biases, dim=0)

    # 保存融合后的模型
    torch.save(avg_model, output_path)
    print(f'模型融合完成，已保存为{output_path}')


# 使用示例
model_path_list_pyra = [
                '/data/guoyufei21/ViT-B_PYRA_logs_weights_high_compression_rate/svhn_lr-0.001_wd-0.0001_pyra_lr-1e-3/best_checkpoint.pth',
                '/data/guoyufei21/ViT-B_PYRA_logs_weights_high_compression_rate/cifar100_lr-0.001_wd-0.0001_pyra_lr-3e-6/best_checkpoint.pth',
                '/data/guoyufei21/ViT-B_PYRA_logs_weights_high_compression_rate/caltech101_lr-0.001_wd-0.0001_pyra_lr-3e-5/best_checkpoint.pth',
                '/data/guoyufei21/ViT-B_PYRA_logs_weights_high_compression_rate/dtd_lr-0.001_wd-0.0001_pyra_lr-3e-4/best_checkpoint.pth',
                '/data/guoyufei21/ViT-B_PYRA_logs_weights_high_compression_rate/oxford_flowers102_lr-0.001_wd-0.0001_pyra_lr-1e-5/best_checkpoint.pth',
                '/data/guoyufei21/ViT-B_PYRA_logs_weights_high_compression_rate/oxford_pet_lr-0.001_wd-0.0001_pyra_lr-1e-6/best_checkpoint.pth',
                '/data/guoyufei21/ViT-B_PYRA_logs_weights_high_compression_rate/sun397_lr-0.001_wd-0.0001_pyra_lr-1e-6/best_checkpoint.pth',
                    ]
model_path_list_tome = [
                '/data/guoyufei21/ViT-B_ToMe_logs_weights_high_compression_rate/svhn_lr-0.001_wd-0.0001_pyra_lr-1e-3/best_checkpoint.pth',
                '/data/guoyufei21/ViT-B_ToMe_logs_weights_high_compression_rate/cifar100_lr-0.001_wd-0.0001_pyra_lr-3e-6/best_checkpoint.pth',
                '/data/guoyufei21/ViT-B_ToMe_logs_weights_high_compression_rate/caltech101_lr-0.001_wd-0.0001_pyra_lr-3e-5/best_checkpoint.pth',
                '/data/guoyufei21/ViT-B_ToMe_logs_weights_high_compression_rate/dtd_lr-0.001_wd-0.0001/best_checkpoint.pth',
                '/data/guoyufei21/ViT-B_ToMe_logs_weights_high_compression_rate/oxford_flowers102_lr-0.001_wd-0.0001_pyra_lr-1e-5/best_checkpoint.pth',
                '/data/guoyufei21/ViT-B_ToMe_logs_weights_high_compression_rate/oxford_pet_lr-0.001_wd-0.0001/best_checkpoint.pth',
                '/data/guoyufei21/ViT-B_ToMe_logs_weights_high_compression_rate/sun397_lr-0.001_wd-0.0001_pyra_lr-1e-6/best_checkpoint.pth',
                    ]
model_path_list_lora = [
                    '/data/guoyufei21/240919_ViT-B_prompt_lora_8_plain_lora/clevr_count_lr-0.001_wd-0.0001/best_checkpoint.pth',
                    '/data/guoyufei21/240919_ViT-B_prompt_lora_8_plain_lora/clevr_dist_lr-0.001_wd-0.0001/best_checkpoint.pth',
                    '/data/guoyufei21/240919_ViT-B_prompt_lora_8_plain_lora/dmlab_lr-0.001_wd-0.0001/best_checkpoint.pth',
                    '/data/guoyufei21/240919_ViT-B_prompt_lora_8_plain_lora/dsprites_loc_lr-0.001_wd-0.0001/best_checkpoint.pth',
                    '/data/guoyufei21/240919_ViT-B_prompt_lora_8_plain_lora/dsprites_ori_lr-0.001_wd-0.0001/best_checkpoint.pth',
                    '/data/guoyufei21/240919_ViT-B_prompt_lora_8_plain_lora/kitti_lr-0.001_wd-0.0001/best_checkpoint.pth',
                    '/data/guoyufei21/240919_ViT-B_prompt_lora_8_plain_lora/smallnorb_azi_lr-0.001_wd-0.0001/best_checkpoint.pth',
                    '/data/guoyufei21/240919_ViT-B_prompt_lora_8_plain_lora/smallnorb_ele_lr-0.001_wd-0.0001/best_checkpoint.pth',
                    ]
model_path_list_lora = [
                    '/data/guoyufei21/240919_ViT-B_prompt_lora_8_plain_lora/diabetic_retinopathy_lr-0.001_wd-0.0001/best_checkpoint.pth',
                    '/data/guoyufei21/240919_ViT-B_prompt_lora_8_plain_lora/eurosat_lr-0.001_wd-0.0001/best_checkpoint.pth',
                    '/data/guoyufei21/240919_ViT-B_prompt_lora_8_plain_lora/patch_camelyon_lr-0.001_wd-0.0001/best_checkpoint.pth',
                    '/data/guoyufei21/240919_ViT-B_prompt_lora_8_plain_lora/resisc45_lr-0.001_wd-0.0001/best_checkpoint.pth',
                    ]
model_path_list_lora = [
                '/data/guoyufei21/240920_ViT-S_prompt_lora_4_plain_lora/svhn_lr-0.001_wd-0.0001/best_checkpoint.pth',
                '/data/guoyufei21/240920_ViT-S_prompt_lora_4_plain_lora/cifar100_lr-0.001_wd-0.0001/best_checkpoint.pth',
                '/data/guoyufei21/240920_ViT-S_prompt_lora_4_plain_lora/caltech101_lr-0.001_wd-0.0001/best_checkpoint.pth',
                '/data/guoyufei21/240920_ViT-S_prompt_lora_4_plain_lora/dtd_lr-0.001_wd-0.0001/best_checkpoint.pth',
                '/data/guoyufei21/240920_ViT-S_prompt_lora_4_plain_lora/oxford_flowers102_lr-0.001_wd-0.0001/best_checkpoint.pth',
                '/data/guoyufei21/240920_ViT-S_prompt_lora_4_plain_lora/oxford_pet_lr-0.001_wd-0.0001/best_checkpoint.pth',
                '/data/guoyufei21/240920_ViT-S_prompt_lora_4_plain_lora/sun397_lr-0.001_wd-0.0001/best_checkpoint.pth',
                    ]


ties_merge_model_weights(model_path_list_pyra, f'best_checkpoint_lora.pth', '/data/guoyufei21/base_model_pyra_high.pth', k=0.8, λ=1.0)


# n = len(model_path_list_lora)
#for i in range(n):
#    rotated_models = model_path_list_lora[i:] + model_path_list_lora[:i]  # 旋转列表
#    ties_merge_model_weights(rotated_models, f'best_checkpoint_{dataset[i]}_lora.pth', '/data/guoyufei21/base_model_S.pth', k=0.8, λ=1)

#for i in range(n):
#    rotated_models = model_path_list_tome[i:] + model_path_list_tome[:i]  # 旋转列表
#    ties_merge_model_weights(rotated_models, f'best_checkpoint_{dataset[i]}_tome.pth', '/data/guoyufei21/base_model_B.pth', k=0.8, λ=1)

#for i in range(n):
#    rotated_models = model_path_list_pyra[i:] + model_path_list_pyra[:i]  # 旋转列表
#    ties_merge_model_weights(rotated_models, f'best_checkpoint_{dataset[i]}_pyra.pth', '/data/guoyufei21/base_model_pyra_low.pth', k=0.8, λ=1)

