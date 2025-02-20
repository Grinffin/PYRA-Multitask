import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import numpy as np

from lib import utils
from lib.datasets import build_dataset

import argparse

import model as models

from train import timm_load_checkpoint
from engine import evaluate



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

        # 为每个模型的每层创建独立的标量参数
        self.model_scalars = nn.ParameterList([
            nn.Parameter(torch.ones(num_layers)) for _ in range(len(all_models))
        ])

        # 冻结每个模型的原始参数
        for model in self.all_models:
            for param in model.parameters():
                param.requires_grad = False

    def forward(self, x):
        merged_output = 0
        # 执行模型的patch_embed和pos_drop等初始层
        x_processed = self.all_models[0].patch_embed(x)
        cls_token = self.all_models[0].cls_token.expand(x_processed.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x_processed = torch.cat((cls_token, x_processed), dim=1)
        x_processed = self.all_models[0].pos_drop(x_processed + self.all_models[0].pos_embed)

        # 按照层级顺序对每层进行合并计算
        for i in range(self.num_layers):
            merged_layer_output = 0
            total_weight = 0

            # 遍历每个模型的第i层
            for model, scalars in zip(self.all_models, self.model_scalars):
                # 按标量权重调整每层输出并相加
                layer_output = model.blocks[i](x_processed) * scalars[i].to(x.device)
                merged_layer_output += layer_output
                total_weight += scalars[i].to(x.device)
            
            # 计算合并后的输出
            x_processed = merged_layer_output / total_weight

        # 对合并后的表示执行 norm 和 head
        normed_output = self.all_models[0].norm(x_processed)
        pre_logits = self.all_models[0].pre_logits(normed_output[:, 0])
        final_output = self.all_models[0].head(pre_logits)
        #final_output = torch.cat([model.head(pre_logits) for model in self.all_models], dim=-1)
        #print(final_output.shape)
        return final_output

    def parameters(self, recurse=True):
        # 返回模型的所有参数，包括每层的 scalars
        return self.model_scalars


device = torch.device('cuda:1')
# fix the seed for reproducibility
seed = 42 + utils.get_rank()
torch.manual_seed(seed)
np.random.seed(seed)
# random.seed(seed)
cudnn.benchmark = True

model_paths = [
                '/data/guoyufei21/240919_ViT-B_prompt_lora_8_plain_lora/cifar100_lr-0.001_wd-0.0001/best_checkpoint.pth',
                # '/data/guoyufei21/240919_ViT-B_prompt_lora_8_plain_lora/caltech101_lr-0.001_wd-0.0001/best_checkpoint.pth',
                # '/data/guoyufei21/240919_ViT-B_prompt_lora_8_plain_lora/dtd_lr-0.001_wd-0.0001/best_checkpoint.pth',
                # '/data/guoyufei21/240919_ViT-B_prompt_lora_8_plain_lora/oxford_flowers102_lr-0.001_wd-0.0001/best_checkpoint.pth',
                # '/data/guoyufei21/240919_ViT-B_prompt_lora_8_plain_lora/oxford_pet_lr-0.001_wd-0.0001/best_checkpoint.pth',
                # '/data/guoyufei21/240919_ViT-B_prompt_lora_8_plain_lora/sun397_lr-0.001_wd-0.0001/best_checkpoint.pth',
                '/data/guoyufei21/240919_ViT-B_prompt_lora_8_plain_lora/svhn_lr-0.001_wd-0.0001/best_checkpoint.pth',
                ]
classes_per_task = [100, 102, 47, 102, 37, 397, 10]
classes_per_task = [100, 10]

parser = argparse.ArgumentParser('AutoFormer training and evaluation script', parents=[get_args_parser()])
args = parser.parse_args()

all_models = []
for nb_classes, path in zip(classes_per_task, model_paths):
    model = models.__dict__['vit_base_patch16_224_in21k'](   
                                                    img_size=224,
                                                    drop_rate=args.drop,
                                                    drop_path_rate=args.drop_path,
                                                    prompt_tuning_dim=0,LoRA_dim=8,adapter_dim=0,prefix_dim=0,
                                                    drop_rate_LoRA=args.drop_rate_LoRA,drop_rate_prompt=args.drop_rate_prompt,drop_rate_adapter=args.drop_rate_adapter,
                                                    IS_not_position_VPT = False,
                                                    pyra_r = None
                                                )
    if nb_classes != model.head.weight.shape[0]:
        model.reset_classifier(nb_classes)
    incompatible_keys = timm_load_checkpoint(model, path, strict=True)  # 这里使用 strict=False 以处理可能的权重不匹配
    print(f"Loaded model from {path}, incompatible keys: {incompatible_keys}")
    model.to(device)
    # 将加载的模型添加到模型列表中
    all_models.append(model)




dataset_train, nb_classes = build_dataset(is_train=True, args=args)
dataset_val, _ = build_dataset(is_train=False, args=args)

sampler_train = torch.utils.data.RandomSampler(dataset_train)
sampler_val = torch.utils.data.SequentialSampler(dataset_val)

data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=16,
        num_workers=10,
        pin_memory=True,
        drop_last=True,
    )
data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=int(2 * 32),
        sampler=sampler_val, num_workers=10,
        pin_memory=True, drop_last=False
    )


# 创建模型合并器
merger = ModelMerger(all_models=all_models, num_layers=12)
metric_logger = utils.MetricLogger(delimiter="  ")
#metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
header = 'Epoch: [{}]'.format(100)
print_freq = 10

optimizer = optim.Adam(merger.parameters(), lr=1e-3)

# merger.train()
# i = 0
# for samples, targets in metric_logger.log_every(data_loader_train, print_freq, header):
#     samples = samples.to(device, non_blocking=True)
#     targets = targets.to(device, non_blocking=True)
#     #print(targets.shape)

#     optimizer.zero_grad()
#     output = merger(samples)

#     loss = softmax_entropy(output).mean(0)
#     loss.backward()
#     optimizer.step()
    
#     # print(merger.parameters()[0])
#     # print(merger.parameters()[1])
#     print(f'Loss: {loss.item()}')
#     metric_logger.update(loss=loss.item())
    # i += 1 
    # if i>10:
    #     break

classes_per_task = [100]
test_stats = evaluate(data_loader_val, merger, device, classes_per_task, args.offset, amp=False)
print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
