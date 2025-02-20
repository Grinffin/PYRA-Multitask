import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 定义 MixtureOfExperts 模型，使用 top-2 gating
class MixtureOfExperts(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts=4, hidden_dim=64):
        super(MixtureOfExperts, self).__init__()
        
        # 专家层
        self.experts = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(num_experts)])
        
        # 门控网络（Gating Network）
        self.gating_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts)
        )

    def forward(self, x):
        # 门控函数计算每个专家的权重（未归一化）
        gate_scores = self.gating_network(x)
        
        # 对门控分数进行 softmax 操作，得到每个专家的激活权重
        gate_weights = F.softmax(gate_scores, dim=-1)

        # 使用 top-2 gating 选择两个激活概率最大的专家
        _, top2_indices = torch.topk(gate_weights, k=2, dim=-1)

        # 获取 top-2 专家的激活权重
        top2_weights = gate_weights.gather(1, top2_indices)

        # 专家输出
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)  # (batch_size, num_experts, output_dim)
        
        # 选择 top-2 专家的输出进行加权求和
        output = torch.sum(expert_outputs.gather(1, top2_indices.unsqueeze(-1).expand(-1, -1, expert_outputs.size(-1))) * top2_weights.unsqueeze(-1), dim=1)
        
        # 返回输出以及每个专家的权重
        return output, gate_weights, top2_indices  # 返回每个专家的激活权重以及 top-2 专家的索引

# 辅助损失函数 L_aux
def auxiliary_loss(D, gate_weights, T):
    """
    计算辅助损失 L_aux = (1/T) * sum(D_i * P_i)
    D: 每个专家分配的样本数量 (batch_size, num_experts)
    gate_weights: 每个专家的权重
    T: batch_size
    """
    # D_i 是每个专家在 batch 中分配的样本数量
    # 这里 gate_weights 用来表示每个专家的权重比例
    loss = torch.sum(D * gate_weights, dim=-1)  # (batch_size)

    return torch.mean(loss)  # (标量)

# 主任务损失函数 (这里使用 MSE 作为例子)
def main_task_loss(output, target):
    return F.mse_loss(output, target)

# 训练模型
def train_moe_model(model, train_loader, num_epochs=10, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_main_loss = 0
        total_aux_loss = 0

        for inputs, targets in train_loader:
            optimizer.zero_grad()

            # 前向传播
            outputs, gate_weights, top2_indices = model(inputs)

            # 计算主任务损失
            main_loss = main_task_loss(outputs, targets)

            # 计算每个专家分配到的样本数量 D_i
            D = torch.zeros(inputs.size(0), gate_weights.size(1)).to(inputs.device)  # 初始化 D_i
            for i in range(inputs.size(0)):  # 对每个样本
                top2 = top2_indices[i]
                D[i, top2] += 1  # 为 top-2 专家计数

            # 计算辅助损失
            aux_loss = auxiliary_loss(D, gate_weights, inputs.size(0))

            # 总损失 = 主任务损失 + 辅助损失
            total_loss = aux_loss  # 辅助损失的权重是 1

            # 反向传播和优化
            total_loss.backward()
            optimizer.step()

            # 记录损失
            total_main_loss += main_loss.item()
            total_aux_loss += aux_loss.item()
        print(torch.sum(D,dim=0))
        print(torch.sum(gate_weights,dim=0))
        print(f"Epoch {epoch+1}/{num_epochs}, Total Loss: {total_loss.item()}, Main Loss: {total_main_loss/len(train_loader)}, Aux Loss: {total_aux_loss/len(train_loader)}")

# 示例：生成一些随机数据并训练模型
if __name__ == "__main__":
    # 假设输入数据维度是 10，输出数据维度是 1
    input_dim = 10
    output_dim = 1
    num_experts = 4

    # 创建一个简易的训练数据集
    class RandomDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples=1000, input_dim=10):
            self.data = torch.randn(num_samples, input_dim)
            self.targets = torch.randn(num_samples, 1)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx], self.targets[idx]

    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(RandomDataset(num_samples=32, input_dim=input_dim), batch_size=32, shuffle=True)

    # 创建 MOE 模型
    model = MixtureOfExperts(input_dim=input_dim, output_dim=output_dim, num_experts=num_experts)

    # 训练模型
    train_moe_model(model, train_loader, num_epochs=1000)
