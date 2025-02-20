import torch

def average_model_weights(model_path_list, output_path):
    # 加载模型的状态字典
    models = [torch.load(path) for path in model_path_list]

    # 只修改 model 部分
    avg_model = models[0].copy()  # 拷贝 model2 的所有结构，包括 optimizer 等
    num_models = len(models)

    # 遍历第一个模型的参数
    for name in models[0]['model']:
        # 对于最后的分类头层 "head.weight" 和 "head.bias"
        if "head.weight" in name:
            # 对权重进行拼接 (dim=0 表示在第一维度上拼接，即类别数量维度)
            weights = [model['model'][name] for model in models]
            avg_model['model'][name] = torch.cat(weights, dim=0)
            # avg_model['model'][name] = model2['model'][name]
        elif "head.bias" in name:
            # 对偏置进行拼接
            biases = [model['model'][name] for model in models]
            avg_model['model'][name] = torch.cat(biases, dim=0)
            # avg_model['model'][name] = model2['model'][name]
        else:
            # 对于其他层，取算术平均
            avg_model['model'][name] = sum(model['model'][name] for model in models) / num_models

    
    # 保存新的模型
    torch.save(avg_model, output_path)

    print(f'模型合并完成，已保存为{output_path}')



# 使用示例
model_path_list_pyra = [
                '/data/guoyufei21/ViT-B_PYRA_logs_weights_high_compression_rate/cifar100_lr-0.001_wd-0.0001_pyra_lr-3e-6/best_checkpoint.pth',
                '/data/guoyufei21/ViT-B_PYRA_logs_weights_high_compression_rate/caltech101_lr-0.001_wd-0.0001_pyra_lr-3e-5/best_checkpoint.pth',
                '/data/guoyufei21/ViT-B_PYRA_logs_weights_high_compression_rate/dtd_lr-0.001_wd-0.0001_pyra_lr-3e-4/best_checkpoint.pth',
                '/data/guoyufei21/ViT-B_PYRA_logs_weights_high_compression_rate/oxford_flowers102_lr-0.001_wd-0.0001_pyra_lr-1e-5/best_checkpoint.pth',
                '/data/guoyufei21/ViT-B_PYRA_logs_weights_high_compression_rate/oxford_pet_lr-0.001_wd-0.0001_pyra_lr-1e-6/best_checkpoint.pth',
                '/data/guoyufei21/ViT-B_PYRA_logs_weights_high_compression_rate/sun397_lr-0.001_wd-0.0001_pyra_lr-1e-6/best_checkpoint.pth',
                '/data/guoyufei21/ViT-B_PYRA_logs_weights_high_compression_rate/svhn_lr-0.001_wd-0.0001_pyra_lr-1e-3/best_checkpoint.pth',
                    ]
model_path_list_tome = [
                    '/data/guoyufei21/ViT-B_ToMe_logs_weights_low_compression_rate/clevr_count_lr-0.001_wd-0.0001/best_checkpoint.pth',
                    '/data/guoyufei21/ViT-B_ToMe_logs_weights_low_compression_rate/clevr_dist_lr-0.001_wd-0.0001/best_checkpoint.pth',
                    '/data/guoyufei21/ViT-B_ToMe_logs_weights_low_compression_rate/dmlab_lr-0.001_wd-0.0001/best_checkpoint.pth',
                    '/data/guoyufei21/ViT-B_ToMe_logs_weights_low_compression_rate/kitti_lr-0.001_wd-0.0001/best_checkpoint.pth',
                    '/data/guoyufei21/ViT-B_ToMe_logs_weights_low_compression_rate/dsprites_loc_lr-0.001_wd-0.0001/best_checkpoint.pth',
                    '/data/guoyufei21/ViT-B_ToMe_logs_weights_low_compression_rate/dsprites_ori_lr-0.001_wd-0.0001/best_checkpoint.pth',
                    '/data/guoyufei21/ViT-B_ToMe_logs_weights_low_compression_rate/smallnorb_azi_lr-0.001_wd-0.0001/best_checkpoint.pth',
                    '/data/guoyufei21/ViT-B_ToMe_logs_weights_low_compression_rate/smallnorb_ele_lr-0.001_wd-0.0001/best_checkpoint.pth',
                    ]
models = ['/data/guoyufei21/240919_ViT-B_prompt_lora_8_plain_lora/cifar100_lr-0.001_wd-0.0001/best_checkpoint.pth',
        '/data/guoyufei21/240919_ViT-B_prompt_lora_8_plain_lora/svhn_lr-0.001_wd-0.0001/best_checkpoint.pth',
        ]
average_model_weights(model_path_list_pyra, f'best_checkpoint_avg.pth')
