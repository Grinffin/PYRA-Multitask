import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torch.backends.cudnn as cudnn

import numpy as np

from lib import utils
from lib.datasets import build_dataset

import argparse

import model as models

from train import timm_load_checkpoint
from engine import evaluate

from model.tome import parse_r, get_merging_schedule, apply_tome
from model.adaptive_merge import bipartite_soft_matching, merge_source, merge_wavg
from timm.utils.model import unwrap_model

import copy



def get_args_parser():
    parser = argparse.ArgumentParser('AutoFormer training and evaluation script', add_help=False)
    parser.add_argument('--data-path', default='/data/guoyufei21/data/vtab-1k/cifar100', type=str,
                        help='dataset path')
     # parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19', 'EVO_IMNET'],
    #                     type=str, help='Image Net dataset path')
    parser.add_argument('--data-set', default='cifar100', type=str, help='Image Net dataset path')
    parser.add_argument('--no_aug', default=True)
    parser.add_argument('--direct_resize', default=True)
    parser.add_argument('--inception', default=False)

    parser.add_argument('--input-size', default=224, type=int)
    parser.add_argument('--patch_size', default=16, type=int)
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                        help='Drop block rate (default: None)')
    parser.add_argument('--drop_rate_LoRA', type=float, default=0)
    parser.add_argument('--drop_rate_prompt', type=float, default=0)
    parser.add_argument('--drop_rate_adapter', type=float, default=0)

    parser.add_argument('--token_merging', default=True)
    parser.add_argument('--merging_schedule', type=str, default='high', choices=['low', 'high'], help='token merging schedule')
    parser.add_argument('--pyra', default=True)

    parser.add_argument('--offset', type=int, default=0)
    return parser

def softmax_entropy(x):
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

class ModelMerger(nn.Module):
    def __init__(self, all_models, num_layers=12):
        super(ModelMerger, self).__init__()
        self.all_models = all_models
        self.num_layers = num_layers
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 768))
        self.num_heads = 8

        # 为每个模型的每层创建独立的标量参数
        self.model_scalars = nn.ParameterList([
            nn.Parameter(torch.ones(num_layers)) for _ in range(len(all_models))
        ])

        # 冻结每个模型的原始参数
        for model in self.all_models:
            for param in model.parameters():
                param.requires_grad = False

    def forward(self, x, head_num=0):
        merged_output = 0
        # 执行模型的patch_embed和pos_drop等初始层
        x = self.all_models[0].patch_embed(x)
        cls_token = self.all_models[0].cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1)
        x = self.all_models[0].pos_drop(x + self.all_models[0].pos_embed)

        # 按照层级顺序对每层进行合并计算
        for i in range(self.num_layers):
            merged_lora_a_weight, merged_lora_b_weight = None, None
            total_weight = 0

            # 遍历每个模型的第i层
            for model, scalars in zip(self.all_models, self.model_scalars):
                lora_a = model.blocks[i].attn.LoRA_a
                lora_b = model.blocks[i].attn.LoRA_b
                scalar = scalars[i].to(x.device)

                # 合并 LoRA_a 和 LoRA_b 权重
                if merged_lora_a_weight is None:
                    merged_lora_a_weight = lora_a.weight * scalar
                    merged_lora_b_weight = lora_b.weight * scalar
                else:
                    merged_lora_a_weight += lora_a.weight * scalar
                    merged_lora_b_weight += lora_b.weight * scalar
                
                total_weight += scalar
            # 对 LoRA_a 和 LoRA_b 的权重进行归一化
            merged_lora_a_weight /= total_weight
            merged_lora_b_weight /= total_weight
            B, N, C = x.shape

            x_processed = model.blocks[i].norm1(x)
            qkv = model.blocks[i].attn.qkv(x_processed).reshape(B, N, 3, model.blocks[i].attn.num_heads, C // model.blocks[i].attn.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
            if model.blocks[i].attn.LoRA_dim > 0:
                qkv_delta = nn.functional.linear(model.blocks[i].attn.LoRA_drop(x_processed), merged_lora_a_weight)
                # qkv_delta = merged_lora_a_weight(model.blocks[i].attn.LoRA_drop(x_processed))
                qkv_delta = nn.functional.linear(qkv_delta, merged_lora_b_weight).reshape(B, N, 3, model.blocks[i].attn.num_heads, C // model.blocks[i].attn.num_heads).permute(2, 0, 3, 1, 4)
                # qkv_delta = merged_lora_b_weight(qkv_delta).reshape(B, N, 3, model.blocks[i].attn.num_heads, C // model.blocks[i].attn.num_heads).permute(2, 0, 3, 1, 4)
                q_delta, k_delta, v_delta = qkv_delta.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
                q,k,v = q+q_delta,k+k_delta,v+v_delta
            attn = (q @ k.transpose(-2, -1)) * model.blocks[i].attn.scale
            attn = attn.softmax(dim=-1)
            attn = model.blocks[i].attn.attn_drop(attn)

            x_processed = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x_processed = model.blocks[i].attn.proj(x_processed)
            x_processed = model.blocks[i].attn.proj_drop(x_processed)
            x = x + model.blocks[i].drop_path(x_processed)
            x = x + model.blocks[i].adapter(model.blocks[i].drop_path(model.blocks[i].mlp(model.blocks[i].norm2(x))))

        # 对合并后的表示执行 norm 和 head
        normed_output = self.all_models[0].norm(x)
        pre_logits = self.all_models[0].pre_logits(normed_output[:, 0])
        if head_num >= 0:
            final_output = self.all_models[head_num].head(pre_logits)
        else:
            final_output = torch.cat([model.head(pre_logits) for model in self.all_models], dim=-1)
        #print(final_output.shape)
        return final_output

    def parameters(self, recurse=True):
        # 返回模型的所有参数，包括每层的 scalars
        return self.model_scalars
    
class ModelMergerTome(nn.Module):
    def __init__(self, all_models, num_layers=12):
        super(ModelMergerTome, self).__init__()
        self.all_models = all_models
        self.num_layers = num_layers

        # 为每个模型的每层创建独立的标量参数
        self.routers = nn.ParameterList([
            nn.Parameter(torch.ones(768, 7)) for _ in range(num_layers)
        ])

        # 冻结每个模型的原始参数
        for model in self.all_models:
            for param in model.parameters():
                param.requires_grad = False

    def forward(self, x, eva=False):
        # tome configurations
        for model in self.all_models:
            model._tome_info["r"] = parse_r(len(model.blocks), model.r)
            model._tome_info["size"] = None
            model._tome_info["source"] = None

        # 执行模型的patch_embed和pos_drop等初始层
        x = self.all_models[0].patch_embed(x)
        cls_token = self.all_models[0].cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1)
        x = self.all_models[0].pos_drop(x + self.all_models[0].pos_embed)

        # 按照层级顺序对每层进行合并计算
        for i, router in zip(range(self.num_layers), self.routers):
            total_weight = 0
            merged_res = 0
            cls = x[:, 0]  # 取 x 的第一个元素，即 cls_token
            result = torch.matmul(cls, router)  # 矩阵乘法结果大小是 [B, 7]
            scores = F.softmax(result, dim=1)  # softmax 之后的大小为 [B, 7]
            if eva==True:
                print(f"block:{i}")
                print(scores)
            for j, model in enumerate(self.all_models):
                attn_size = model.blocks[i]._tome_info["size"] if model.blocks[i]._tome_info["prop_attn"] else None
                B, N, C = x.shape

                model_scores = scores[:, j]  # 对应模型的分数为 [B]

                x_processed = model.blocks[i].norm1(x)
                qkv = model.blocks[i].attn.qkv(x_processed).reshape(B, N, 3, model.blocks[i].attn.num_heads, C // model.blocks[i].attn.num_heads).permute(2, 0, 3, 1, 4)
                q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
                if model.blocks[i].attn.LoRA_dim > 0:
                    #qkv_delta = nn.functional.linear(model.blocks[i].attn.LoRA_drop(x_processed), model.blocks[i].attn.LoRA_a)
                    qkv_delta = model.blocks[i].attn.LoRA_a(model.blocks[i].attn.LoRA_drop(x_processed))
                    #qkv_delta = nn.functional.linear(qkv_delta, model.blocks[i].attn.LoRA_b).reshape(B, N, 3, model.blocks[i].attn.num_heads, C // model.blocks[i].attn.num_heads).permute(2, 0, 3, 1, 4)
                    qkv_delta = model.blocks[i].attn.LoRA_b(qkv_delta).reshape(B, N, 3, model.blocks[i].attn.num_heads, C // model.blocks[i].attn.num_heads).permute(2, 0, 3, 1, 4)
                    q_delta, k_delta, v_delta = qkv_delta.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
                    q,k,v = q+q_delta,k+k_delta,v+v_delta
                attn = (q @ k.transpose(-2, -1)) * model.blocks[i].attn.scale
                if attn_size is not None:
                    attn = attn + attn_size.log()[:, None, None, :, 0]
                attn = attn.softmax(dim=-1)
                attn = model.blocks[i].attn.attn_drop(attn)

                x_processed = (attn @ v).transpose(1, 2).reshape(B, N, C)
                x_processed = model.blocks[i].attn.proj(x_processed)
                x_processed = model.blocks[i].attn.proj_drop(x_processed)
                x_processed = x + model.blocks[i]._drop_path1(x_processed)
                r = model.blocks[i]._tome_info["r"].pop(0)

                if r > 0:
                    # Apply ToMe here
                    merge, _ = bipartite_soft_matching(
                        k.mean(1),
                        r,
                        model.blocks[i]._tome_info["class_token"],
                        model.blocks[i]._tome_info["distill_token"],
                    )
                    if model.blocks[i]._tome_info["trace_source"]:
                        model.blocks[i]._tome_info["source"] = merge_source(
                            merge, x_processed, model.blocks[i]._tome_info["source"]
                        )
                    x_processed, model.blocks[i]._tome_info["size"] = merge_wavg(merge, x_processed, model.blocks[i]._tome_info["size"], 
                                                            pyra_weight=model.blocks[i].pyra,
                                                            #pyra_weight=merged_pyra,
                                                            is_training=model.blocks[i].training)
                model_scores_expanded = model_scores.view(B, 1, 1)
                merged_res = merged_res + x_processed * model_scores_expanded
            x = merged_res + model.blocks[i].adapter(model.blocks[i].drop_path(model.blocks[i].mlp(model.blocks[i].norm2(merged_res))))

        # 对合并后的表示执行 norm 和 head
        normed_output = self.all_models[0].norm(x)
        pre_logits = self.all_models[0].pre_logits(normed_output[:, 0])
        final_output = torch.cat([model.head(pre_logits) for model in self.all_models], dim=-1)
        #print(final_output.shape)
        return final_output

    def parameters(self, recurse=True):
        # 返回模型的所有参数，包括每层的 scalars
        return self.routers


device = torch.device('cuda:0')
# fix the seed for reproducibility
# seed = 42 + utils.get_rank()
# torch.manual_seed(seed)
# np.random.seed(seed)
cudnn.benchmark = True


exam_datasets = ['cifar100', 'caltech101', 'dtd', 'oxford_flowers102', 'oxford_pet', 'sun397', 'svhn']
dataset_paths = [
                '/data/guoyufei21/data/vtab-1k/cifar100',
                '/data/guoyufei21/data/vtab-1k/caltech101',
                '/data/guoyufei21/data/vtab-1k/dtd',
                '/data/guoyufei21/data/vtab-1k/oxford_flowers102',
                '/data/guoyufei21/data/vtab-1k/oxford_pet',
                '/data/guoyufei21/data/vtab-1k/sun397',
                '/data/guoyufei21/data/vtab-1k/svhn'
                ]
model_paths_lora = [
                '/data/guoyufei21/best_checkpoint_0.pth',
                '/data/guoyufei21/best_checkpoint_1.pth',
                '/data/guoyufei21/best_checkpoint_2.pth',
                '/data/guoyufei21/best_checkpoint_3.pth',
                '/data/guoyufei21/best_checkpoint_4.pth',
                '/data/guoyufei21/best_checkpoint_5.pth',
                '/data/guoyufei21/best_checkpoint_6.pth',
                ]
model_paths_tome = [
                '/data/guoyufei21/ViT-B_ToMe_logs_weights_high_compression_rate/cifar100_lr-0.001_wd-0.0001_pyra_lr-3e-6/best_checkpoint.pth',
                '/data/guoyufei21/ViT-B_ToMe_logs_weights_high_compression_rate/caltech101_lr-0.001_wd-0.0001_pyra_lr-3e-5/best_checkpoint.pth',
                '/data/guoyufei21/ViT-B_ToMe_logs_weights_high_compression_rate/dtd_lr-0.001_wd-0.0001/best_checkpoint.pth',
                '/data/guoyufei21/ViT-B_ToMe_logs_weights_high_compression_rate/oxford_flowers102_lr-0.001_wd-0.0001_pyra_lr-1e-5/best_checkpoint.pth',
                '/data/guoyufei21/ViT-B_ToMe_logs_weights_high_compression_rate/oxford_pet_lr-0.001_wd-0.0001/best_checkpoint.pth',
                '/data/guoyufei21/ViT-B_ToMe_logs_weights_high_compression_rate/sun397_lr-0.001_wd-0.0001_pyra_lr-1e-6/best_checkpoint.pth',
                '/data/guoyufei21/ViT-B_ToMe_logs_weights_high_compression_rate/svhn_lr-0.001_wd-0.0001_pyra_lr-1e-3/best_checkpoint.pth',
                ]
model_paths_pyra = [
                '/data/guoyufei21/ViT-B_PYRA_logs_weights_high_compression_rate/cifar100_lr-0.001_wd-0.0001_pyra_lr-3e-6/best_checkpoint.pth',
                '/data/guoyufei21/ViT-B_PYRA_logs_weights_high_compression_rate/caltech101_lr-0.001_wd-0.0001_pyra_lr-3e-5/best_checkpoint.pth',
                '/data/guoyufei21/ViT-B_PYRA_logs_weights_high_compression_rate/dtd_lr-0.001_wd-0.0001_pyra_lr-3e-4/best_checkpoint.pth',
                '/data/guoyufei21/ViT-B_PYRA_logs_weights_high_compression_rate/oxford_flowers102_lr-0.001_wd-0.0001_pyra_lr-1e-5/best_checkpoint.pth',
                '/data/guoyufei21/ViT-B_PYRA_logs_weights_high_compression_rate/oxford_pet_lr-0.001_wd-0.0001_pyra_lr-1e-6/best_checkpoint.pth',
                '/data/guoyufei21/ViT-B_PYRA_logs_weights_high_compression_rate/sun397_lr-0.001_wd-0.0001_pyra_lr-1e-6/best_checkpoint.pth',
                '/data/guoyufei21/ViT-B_PYRA_logs_weights_high_compression_rate/svhn_lr-0.001_wd-0.0001_pyra_lr-1e-3/best_checkpoint.pth',
                ]
classes_per_task = [100, 102, 47, 102, 37, 397, 10]
# classes_per_task = [100, 10]
parser = argparse.ArgumentParser('AutoFormer training and evaluation script', parents=[get_args_parser()])
args = parser.parse_args()
if args.token_merging:
        tome_r = get_merging_schedule('vit_base_patch16_224_in21k', 'high')
        pyra_r = None
        if args.pyra:
            pyra_r = tome_r
else:
    pyra_r = None
    tome_r = None

all_models = []
for nb_classes, path in zip(classes_per_task, model_paths_pyra):
    model = models.__dict__['vit_base_patch16_224_in21k'](   
                                                    img_size=224,
                                                    drop_rate=args.drop,
                                                    drop_path_rate=args.drop_path,
                                                    prompt_tuning_dim=0,LoRA_dim=8,adapter_dim=0,prefix_dim=0,
                                                    drop_rate_LoRA=args.drop_rate_LoRA,drop_rate_prompt=args.drop_rate_prompt,drop_rate_adapter=args.drop_rate_adapter,
                                                    IS_not_position_VPT = False,
                                                    pyra_r = pyra_r
                                                )
    if args.token_merging:
        print("Token merging initialization.")
        model_module = unwrap_model(model)
        apply_tome(model_module)
        model_module.r = tome_r

    if nb_classes != model.head.weight.shape[0]:
        model.reset_classifier(nb_classes)
    incompatible_keys = timm_load_checkpoint(model, path, strict=True)  # 这里使用 strict=False 以处理可能的权重不匹配
    print(f"Loaded model from {path}, incompatible keys: {incompatible_keys}")
    model.to(device)
    # 将加载的模型添加到模型列表中
    all_models.append(model)

# 创建模型合并器
merger = ModelMergerTome(all_models=all_models, num_layers=12)
merger.to(device)
metric_logger = utils.MetricLogger(delimiter="  ")
#metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
header = 'Epoch: [{}]'.format(100)
print_freq = 10

optimizer = optim.Adam(merger.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

merger.train()
criterion.train()

offset = [0, 100, 202, 249, 351, 388, 785]
for epoch in range(10):
    losses = 0.
    i = 0
    for dataset_name, dataset_path in zip(exam_datasets, dataset_paths):
        print(dataset_name)
        args.data_set = dataset_name
        args.data_path = dataset_path
        dataset_train, nb_classes = build_dataset(is_train=True, args=args)
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        data_loader_train = torch.utils.data.DataLoader(
                dataset_train, sampler=sampler_train,
                batch_size=16,
                num_workers=10,
                pin_memory=True,
                drop_last=True,
            )

        for samples, targets in metric_logger.log_every(data_loader_train, print_freq, header):
            samples = samples.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            targets = targets + offset[i]

            output = merger(samples)

            loss = criterion(output, targets)
            # losses += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            break
        i += 1
    
    
    metric_logger.update(loss=loss.item())
    metric_logger.update(lr=optimizer.param_groups[0]["lr"])

metric_logger.synchronize_between_processes()
print("Averaged stats:", metric_logger)
    
    

i = 0
offset = 0
for dataset_name, dataset_path in zip(exam_datasets, dataset_paths):
    args.data_set = dataset_name
    args.data_path = dataset_path
    dataset_val, _ = build_dataset(is_train=False, args=args)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = torch.utils.data.DataLoader(
            dataset_val, batch_size=int(2 * 32),
            sampler=sampler_val, num_workers=10,
            pin_memory=True, drop_last=False
        )
    classes = [classes_per_task[i]]
    test_stats = evaluate(data_loader_val, merger, device, classes_per_task, offset, amp=False)
    print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
    offset += classes_per_task[i]
    i += 1

i = 0
offset = 0
for dataset_name, dataset_path in zip(exam_datasets, dataset_paths):
    args.data_set = dataset_name
    args.data_path = dataset_path
    dataset_val, _ = build_dataset(is_train=False, args=args)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = torch.utils.data.DataLoader(
            dataset_val, batch_size=int(14),
            sampler=sampler_val, num_workers=10,
            pin_memory=True, drop_last=False
        )
    classes = [classes_per_task[i]]
    print(dataset_name)
    test_stats = evaluate(data_loader_val, merger, device, classes_per_task, offset, amp=False, eva=True)
    print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
    offset += classes_per_task[i]
    i += 1


# i = 0
# offset = 0
# for dataset_name, dataset_path in zip(exam_datasets, dataset_paths):
#     args.data_set = dataset_name
#     args.data_path = dataset_path
#     dataset_val, _ = build_dataset(is_train=False, args=args)
#     sampler_val = torch.utils.data.SequentialSampler(dataset_val)
#     data_loader_val = torch.utils.data.DataLoader(
#             dataset_val, batch_size=int(2 * 32),
#             sampler=sampler_val, num_workers=10,
#             pin_memory=True, drop_last=False
#         )
#     classes = [classes_per_task[i]]
#     test_stats = evaluate(data_loader_val, merger, device, classes, 0, amp=False, head_num=i)
#     print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
#     offset += classes_per_task[i]
#     i += 1



