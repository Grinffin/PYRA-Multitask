import math
import sys
from typing import Iterable, Optional
from timm.utils.model import unwrap_model
import torch
import torch.nn.functional as F

from timm.data import Mixup
from timm.utils import accuracy, ModelEma
from lib import utils
import random
import time

from model.tome import apply_tome

# flops count
from fvcore.nn import FlopCountAnalysis

# visualization
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

def test_model_latency(model, device, test_batch_size=512):
    T0 = 10
    T1 = 10
    speed = 0
    model.eval()
    with torch.no_grad():
        x = torch.randn(test_batch_size, 3, 224, 224).to(device)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        start = time.time()
        while time.time() - start < T0:   
            model(x)
        torch.cuda.synchronize()
        print("*****Test model latency (images per second)*****")
        timing = []
        while sum(timing) < T1:
            start = time.time()
            model(x)
            torch.cuda.synchronize()
            timing.append(time.time() - start)
        timing = torch.as_tensor(timing, dtype=torch.float32)
        speed=512/timing.mean().item()
        print("Model latency: {} imgs/s".format(speed))
    
    return speed
    # exit(0)

def count_model_flops(model, device):
    model.eval()
    rand_input = torch.rand((1,3,224,224)).to(device)
    flops = FlopCountAnalysis(model,rand_input)
    print("*****Count model FLOPS (GFLOPS)*****")
    total_flops = flops.total()
    print("Total FLOPS:%d, GFLOPS:%.4f"%(total_flops, float(total_flops)/1073741824))
    print(flops.by_module_and_operator())
    return float(total_flops)/1073741824

def target_to_task_labels(target):
    # 初始化 task_labels 与 target 形状相同
    task_labels = torch.zeros_like(target, dtype=torch.long)

    # 使用条件判断对 task_labels 进行赋值
    task_labels[(target >= 0) & (target <= 99)] = 0
    task_labels[(target >= 100) & (target <= 201)] = 1
    task_labels[(target >= 202) & (target <= 248)] = 2
    task_labels[(target >= 249) & (target <= 350)] = 3
    task_labels[(target >= 351) & (target <= 387)] = 4
    task_labels[(target >= 388) & (target <= 784)] = 5
    task_labels[target >= 785] = 6

    return task_labels

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    amp: bool = True, teacher_model: torch.nn.Module = None,
                    teach_loss: torch.nn.Module = None,
                    deit=False):

    model.train()
    criterion.train()

    # set random seed
    random.seed(epoch)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        if amp:
            with torch.cuda.amp.autocast():
                if teacher_model:
                    with torch.no_grad():
                        teach_output = teacher_model(samples)
                    _, teacher_label = teach_output.topk(1, 1, True, True)
                    outputs = model(samples)
                    loss = 1/2 * criterion(outputs, targets) + 1/2 * teach_loss(outputs, teacher_label.squeeze())
                else:
                    outputs = model(samples)
                    loss = criterion(outputs, targets)
        else:
            if not deit:
                outputs = model(samples, task_labels=target_to_task_labels(targets), epoch=epoch)
            else:
                outputs, _ = model(samples)

            if teacher_model:
                with torch.no_grad():
                    teach_output = teacher_model(samples)
                _, teacher_label = teach_output.topk(1, 1, True, True)
                loss = 1 / 2 * criterion(outputs, targets) + 1 / 2 * teach_loss(outputs, teacher_label.squeeze())
            else:
                loss = criterion(outputs, targets)

        loss_value = loss.item()

        grad_clip = False
        if not math.isfinite(loss_value):
            print("Loss is {}, clipping gradient".format(loss_value))
            grad_clip = True

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        if amp:
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)
        else:
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.trainable_params(), 10)
            optimizer.step()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def visualize_feature(feature, target):
    # 先转换为 NumPy 数组
    cls_token_output = feature.numpy()  # 转换为 NumPy

    # 使用 t-SNE 将特征降维到 2D
    tsne = TSNE(n_components=2, random_state=42)
    cls_token_output_2d = tsne.fit_transform(cls_token_output)  # shape: [batch_size, 2]

    # 可视化 t-SNE 降维结果
    plt.figure(figsize=(8, 8))
    plt.scatter(cls_token_output_2d[:, 0], cls_token_output_2d[:, 1], c=target.cpu(), cmap='jet', s=10)  # c=target是标签
    plt.colorbar()
    plt.title('t-SNE Visualization of CLS Token Features')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')

    # 保存图像为文件，指定文件名和格式
    plt.savefig('tsne_visualization.png', format='png')  # 可以指定保存路径
    plt.close()  # 关闭图像，释放资源


@torch.no_grad()
def evaluate(data_loader, model, device, classes_per_task=[1000], offset=0, amp=True, eva=False):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    # switch to evaluation mode

    # 定义存储特征的字典
    # features = {}

    # # 定义钩子函数
    # def hook_fn(module, input, output):
    #     features['hooked'] = output.detach().cpu()

    # # 将钩子注册到倒数第二层
    # handle = model.pre_logits.register_forward_hook(hook_fn)
    # hooked_output = []
    # label = []

    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        # compute output
        if amp:
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, target)
        else:
            output = model(images)
            loss = criterion(output, target)

        #hooked_output.append(features['hooked'])  # 这是倒数第二层输出
        #label.append(target)
        #handle.remove()

        n = len(classes_per_task)  # 任务数量
        total_classes = sum(classes_per_task)  # 总类别数

        # 对 output 应用 softmax
        output = torch.softmax(output, dim=1)

        # 存储每个任务的平均概率和输出
        task_mean_probs = []
        task_outputs = []

        start_index = 0

        # 遍历每个任务，计算平均概率和输出
        for task_classes in classes_per_task:
            end_index = start_index + task_classes
            task_output = output[:, start_index:end_index]
            #task_output = torch.softmax(task_output, dim=1)
            task_mean_prob = task_output.mean(dim=1)

            task_outputs.append(task_output)  # 存储每个任务的输出
            task_mean_probs.append(task_mean_prob)  # 存储每个任务的平均概率

            start_index = end_index  # 更新开始索引

        # 将所有任务的平均概率组合成一个 tensor
        task_mean_probs = torch.stack(task_mean_probs, dim=1)  # 形状为 [batch_size, n]

        # 比较任务的平均概率，得到每个 batch 中概率最大的任务的索引
        max_task_indices = torch.argmax(task_mean_probs, dim=1)  # 形状为 [batch_size]
        # print(max_task_indices)
        # 创建一个列表来存储每个任务的输出
        selected_outputs = []

        # 遍历每个任务，保留概率最大的任务的输出
        for task_index in range(n):
            task_output = task_outputs[task_index]  # 该任务的输出
            mask = (max_task_indices == task_index)  # 获取该任务的 mask

            # 创建一个临时张量，初始化为负无穷
            temp_output = torch.full((task_output.size(0), task_output.size(1)), -float('inf')).to(device)

            # 将该任务的输出填入 temp_output 中
            temp_output[mask] = task_output[mask]  # 赋值时确保形状匹配

            # 将处理好的 temp_output 添加到 selected_outputs 列表
            selected_outputs.append(temp_output)

        # 将所有的 selected_outputs 拼接成 new_output
        new_output = torch.cat(selected_outputs, dim=1)
        #task_outputs = torch.cat(task_outputs, dim=1)
        #print(f"new output:{new_output}")

        acc1, acc5 = accuracy(new_output, target+offset, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        if eva==True:
            break
    # 可视化
    #hooked_output = torch.cat(hooked_output, dim=0)
    #label = torch.cat(label, dim=0)
    #print(hooked_output.shape)
    #print(label.shape)
    #visualize_feature(hooked_output, label)
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    with open('zzz.txt', 'a', encoding='utf-8') as file:
        # 写入新内容到文件末尾
        file.write('\n')
        file.write('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
