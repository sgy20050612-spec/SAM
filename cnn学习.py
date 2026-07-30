import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class MyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 卷积层1: 1通道 -> 32通道, 卷积核5x5
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=5,
            stride=1,
            padding=2
        )
        # 卷积层2: 32通道 -> 64通道, 卷积核5x5
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=5,
            stride=1,
            padding=2
        )
        # 池化层: 2x2 窗口
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 全连接层1: 3136 -> 128
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        # 全连接层2 (输出层): 128 -> 10
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Conv1 + ReLU + Pool
        x = self.pool(F.relu(self.conv1(x)))   # [B,32,14,14]
        # Conv2 + ReLU + Pool
        x = self.pool(F.relu(self.conv2(x)))   # [B,64,7,7]
        # 展平
        x = x.view(-1, 64 * 7 * 7)             # [B,3136]
        # 全连接1 + ReLU
        x = F.relu(self.fc1(x))                # [B,128]
        # 全连接2 (输出)
        x = self.fc2(x)                        # [B,10]
        return x

# ============================================================
# 第 2 步：加载 MNIST 数据集
# ============================================================
# 数据预处理：转张量 + 归一化到 [0,1]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST 的均值和标准差
])

# 下载并加载训练集
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

# 下载并加载测试集
test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)
test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

print(f"训练集样本数: {len(train_dataset)}, 测试集样本数: {len(test_dataset)}")

# ============================================================
# 第 3 步：创建模型、损失函数和优化器
# ============================================================
model = MyCNN().to(device)
criterion = nn.CrossEntropyLoss()                # 交叉熵损失（分类任务标配）
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam 优化器

print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
print(model)

# ============================================================
# 第 4 步：训练函数
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer, epoch):
    model.train()  # 设为训练模式
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        # 1. 清空梯度
        optimizer.zero_grad()
        # 2. 前向传播
        outputs = model(images)
        # 3. 计算损失
        loss = criterion(outputs, labels)
        # 4. 反向传播
        loss.backward()
        # 5. 更新参数
        optimizer.step()

        # 统计
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # 每 100 个 batch 打印一次
        if (batch_idx + 1) % 100 == 0:
            print(f'Epoch [{epoch}], Batch [{batch_idx+1}/{len(loader)}], '
                  f'Loss: {running_loss/(batch_idx+1):.4f}, '
                  f'Acc: {100.*correct/total:.2f}%')

    return running_loss / len(loader), 100. * correct / total

# ============================================================
# 第 5 步：测试/评估函数
# ============================================================
@torch.no_grad()  # 测试时不计算梯度
def evaluate(model, loader, criterion):
    model.eval()  # 设为评估模式
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / len(loader), 100. * correct / total

# ============================================================
# 第 6 步：训练循环
# ============================================================
num_epochs = 5
train_losses, train_accs = [], []
test_losses, test_accs = [], []

for epoch in range(1, num_epochs + 1):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
    test_loss, test_acc = evaluate(model, test_loader, criterion)

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_losses.append(test_loss)
    test_accs.append(test_acc)

    print(f'===== Epoch {epoch} 完成: '
          f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
          f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}% =====')

# ============================================================
# 第 7 步：可视化训练过程
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(range(1, num_epochs+1), train_losses, 'b-o', label='Train Loss')
axes[0].plot(range(1, num_epochs+1), test_losses, 'r-o', label='Test Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Loss Curve')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(range(1, num_epochs+1), train_accs, 'b-o', label='Train Acc')
axes[1].plot(range(1, num_epochs+1), test_accs, 'r-o', label='Test Acc')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title('Accuracy Curve')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('training_curve.png', dpi=150)
plt.show()
print("Training curve saved as training_curve.png")

"""
┌─────────────────────────────────────────────────────────────────────────────┐
│ 阶段一：数据准备                                                             │
│ 1. 收集数据 → 2. 划分训练集/验证集/测试集 → 3. 数据预处理（归一化）→            │
│ 4. 封装为 Dataset → 5. 创建 DataLoader（批量加载）                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 阶段二：模型构建                                                           │
│ 1. 定义网络结构（继承 nn.Module）→ 2. 定义损失函数 → 3. 定义优化器（Adam） │
└─────────────────────────────────────────────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 阶段三：训练循环（核心）                                                   │
│ for epoch in range(epochs):                                               │
│     for batch in train_loader:                                           │
│         ① 前向传播：outputs = model(images)  （dropout，self.dropout = nn.dropout(0.5))   │
│         ② 计算损失：loss = criterion(outputs, labels)                     │
│         ③ 反向传播：loss.backward()  ← 梯度计算                           │
│         ④ 更新参数：optimizer.step()   ← 梯度下降 （adam优化梯度） （L2 正则化 / 权重衰减）
                     optimizer = optim.Adam(model.parameters(), lr=0.001(学习率）))   │
│         ⑤ 清零梯度：optimizer.zero_grad()                                 │
│     ↓                                                                     │
│     验证集评估（可选）：计算 val_loss, val_acc                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 阶段四：最终评估                                                           │
│ 在测试集上评估最终模型 → 计算准确率、损失                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 阶段五：模型保存与推理                                                     │
│ 1. torch.save(model.state_dict(), 'model.pth')                            │
│ 2. 加载模型 → 输入新数据 → 输出预测结果                                    │
└─────────────────────────────────────────────────────────────────────────────┘

卷积核padding计算：
输出尺寸=（⌊输入尺寸+2×padding−卷积核尺寸⌋+1）/步长；
⌊⋅⌋ 表示向下取整（即舍去小数部分）
这个公式对高度和宽度分别独立计算
"""







"""
所有参数分为三大类：

模型参数（可学习）：网络自己从数据中学出来的。例如：权重 W、偏置 b、BatchNorm 的 γ, β。
超参数（你设定的）：训练前你手动指定的。例如：学习率、层数、Batch Size、Dropout 概率。
结构参数（架构设计）：网络骨架，定了就基本不改。例如：卷积核大小、通道数、激活函数类型。


一、模型参数（Model Parameters）

说明：网络在训练过程中自动学习、更新，存储在 model.parameters() 里。

参数名                对应层                      形状                        作用
权重 W                所有 Linear / Conv 层       (out_ch, in_ch, kH, kW)     控制每个输入特征对输出的贡献强度
卷积核权重            Conv2d / Conv1d / Conv3d    (out_channels, in_channels, kernel_h, kernel_w)   滑动窗口提取局部特征
偏置 b                所有 Linear / Conv 层       (out_ch,)                   控制神经元的"激活门槛"
缩放因子 γ            BatchNorm / LayerNorm       (num_features,)             归一化后重新缩放，恢复表达能力
平移因子 β            BatchNorm / LayerNorm       (num_features,)             归一化后重新平移
运行均值 running_mean  BatchNorm1d / 2d           (num_features,)             训练时统计的均值，测试时用于归一化
运行方差 running_var   BatchNorm1d / 2d           (num_features,)             训练时统计的方差，测试时用于归一化
位置编码              Transformer / ViT           (num_patches, embed_dim)    可学习的位置信息
嵌入层权重            Embedding                   (vocab_size, embed_dim)     把离散 ID 映射为稠密向量
LSTM / GRU 权重       LSTM / GRU                  (hidden_size * multiplier, input_size)   循环神经网络的门控权重
LSTM / GRU 偏置       LSTM / GRU                  (hidden_size * multiplier,)  循环神经网络的偏置项


二、训练超参数（Training Hyperparameters）

说明：在启动训练前设定，控制训练过程。是调参的主要对象。

优化器相关：

参数名                典型值                      调参方向
优化器类型            AdamW, SGD, RMSprop         AdamW 最稳妥，SGD 可达到更高精度
学习率 lr             1e-3（Adam）/ 1e-2（SGD）   太大震荡/爆炸，太小收敛慢
权重衰减 weight_decay  0.01                       控制 L2 正则化强度，值越大约束越强
动量 momentum         0.9                        SGD 时使用，让梯度更新保持惯性
Adam 的 betas         (0.9, 0.999)               控制动量和二阶矩衰减，通常保持默认

数据加载相关：

参数名                典型值                      调参方向
Batch Size            64, 128, 256（2的n次方）               太小梯度噪声大；太大受显存限制
Epochs                10, 50, 100                太多过拟合，太少欠拟合
Shuffle               训练集 True，测试集 False   训练集必须打乱
Num Workers           Windows 设 0，Linux 可设 4~8  数据加载并行线程数

损失函数相关：

参数名                典型值                      说明
损失函数类型          CrossEntropyLoss, MSELoss  分类用交叉熵，回归用 MSE
Class Weight          不平衡数据集时设置          给少数类更高权重
Label Smoothing       0.1 左右                   让标签不过硬，提升泛化

正则化相关：

参数名                典型值                      调参方向
Dropout 概率          0.5（FC）/ 0.1（Transformer）  过拟合时增大，欠拟合时减小
权重衰减系数 λ        0.01                       过拟合时增大，欠拟合时减小
早停 Early Stopping   patience=5~10              验证集不提升时停止训练

学习率调度器相关：

参数名                典型值                      说明
调度器类型            StepLR, CosineAnnealingLR  决定学习率在训练过程中的下降方式
Step Size            5, 10                      每隔多少 Epoch 调整一次
Gamma                0.5, 0.1                   每次调整的比例
Patience             5                          ReduceLROnPlateau 用，验证集多少轮不下降后触发衰减
Warmup Steps         总步数的 5%~10%            让学习率从 0 缓慢爬升到设定值


三、模型结构参数（Architecture Hyperparameters）

说明：设计网络时决定，训练过程中几乎不再变动。

参数名                典型值                      调参方向
层数 Depth           2~5（小网络）/ 50~100（大网络）  任务越复杂，层数越多
每层宽度             32→64→128（通道数翻倍）       宽度越大容量越大，也易过拟合
卷积核大小 kernel_size  3×3, 5×5, 7×7            越大感受野越广，参数越多
步长 Stride          1, 2                       stride=1 保持尺寸，stride=2 下采样
填充 Padding         0, 1, 2                    控制输出是否保持尺寸不变
池化类型             MaxPool, AvgPool            前者保留显著特征，后者更平滑
激活函数类型         ReLU, GELU, Sigmoid, Tanh   决定非线性变换的方式
归一化类型           BatchNorm, LayerNorm        CNN 用 BatchNorm，Transformer 用 LayerNorm
输入通道数 in_channels  1（灰度）/ 3（RGB）       取决于数据
输出类别数 num_classes  10, 100, 1000            取决于任务


附：常见问题速查

现象                    调什么                      怎么调
Loss 不下降             学习率                      调大（×3~10）
Loss 震荡剧烈           学习率                      调小（÷3~5）
Loss 变成 NaN           学习率                      大幅调小（如 1e-4）
训练集好，验证集差       过拟合                     增大 Dropout 或 weight_decay
训练集也差              欠拟合                     增加层数/宽度，减小 Dropout
验证 Loss 后期回升      早停 / 调度器              开启早停，或使用 ReduceLROnPlateau
显存不足（OOM）         Batch Size                 调小（64→32→16）
不同运行结果差异大      随机种子 / Batch Size       固定种子，增大 Batch Size
"""



"""
resnet残差链接：

代码实现：
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 如果输入输出通道数不同，用 1x1 卷积对 x 做投影，确保形状匹配
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = x  # ← 保存原始输入
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = out + self.shortcut(identity)  # ← 残差连接：输出 = F(x) + x
        out = self.relu(out)
        return out




1. 一句话说清残差连接
残差连接 = 把输入直接“抄近道”跳到输出，和经过几层变换后的结果叠加在一起。

公式上就是：

text
输出 = F(x) + x
x：输入（原始信息）

F(x)：经过几层卷积/线性变换后的结果（提取的新特征）

+：逐元素相加

这个“跳过几层直接相加”的操作，就是残差连接。

2. 它解决了什么问题？（梯度消失 + 网络退化）
在 ResNet 出现之前，人们发现网络不是越深越好。当网络超过 20 层时，效果反而变差，这被称为网络退化。

按理说：层数越多，表达能力越强，效果应该更好。

但实际上：层数越多，梯度在回传时经过太多层，数值会变得极小（梯度消失），前面的层几乎学不到东西。

残差连接的解法：让梯度有一条“高速公路”可以直接跳过中间层，从输出直接流回输入。即使中间几层学得不好，梯度也可以通过这条“抄近道”无损地传回去。

text
普通网络：输入 → 层1 → 层2 → 层3 → 输出
           梯度经过 3 层，容易消失

残差网络：输入 → 层1 → 层2 → 层3 → (+) → 输出
              ↑_______________________________|
              梯度可以直接从输出跳回输入
这条“跳线”让梯度回传变得更加顺畅，即便堆叠到 100 层、1000 层也能有效训练。

3. 残差连接在现代模型中的地位
模型	残差连接的使用方式
ResNet（2015）	在卷积块之间加残差：输出 = Conv(x) + x
Transformer / ViT（2017+）	在每个子层（Self-Attention、FFN）后加残差：输出 = LayerNorm(x + Sublayer(x))
DINOv2 / SAM	完全沿用了 Transformer 的残差设计
你看，残差连接已经从“ResNet 的专属特征”变成了“几乎所有现代深度模型的标配组件”。

4. 你只需要记住这 3 点
什么是残差连接：输出 = F(x) + x，把输入抄近道加到输出上。

为什么需要它：让梯度能“跳”过中间层，直接回传，解决深层网络梯度消失问题。

哪里能看到它：所有现代模型（ViT、DINOv2、SAM、Transformer）都用它。

5. 总结一句话
残差连接不是一堆层数的数字，而是一个设计思想：让信息（前向）和梯度（反向）都能有一条“高速公路”直接穿过网络，避免在中间层被损耗。你理解了这个思想，就知道它为什么无处不在。 😊

"""

# ============================================================================
# 如何通过训练结果数据判断模型问题
# ============================================================================
# 训练过程中你会得到这些数据:
#   - train_loss / test_loss (每 epoch 的平均损失)
#   - train_acc / test_acc (准确率)
#   - 梯度范数 (gradient norm, 通过 hook 获取)
#   - 参数值分布 (权重的 mean/std/min/max)
# 下面逐一说明如何从这些数据中诊断模型问题。

# ============================================================================
# 1. 过拟合 (Overfitting) — 最常见
# ============================================================================
"""
典型数据特征:
  Train Loss: 持续下降 -> 0.05, 0.03, 0.01 ...
  Test Loss:  先下降, 然后拐头上升 -> 0.30, 0.28, 0.35, 0.50 ...
  Train Acc:  持续上升 -> 95%, 97%, 99% ...
  Test Acc:   先上升, 然后停滞或下降 -> 85%, 87%, 86%, 84% ...

可视化判断 (看 loss/acc 曲线图):
  loss 图: train 持续下降, test 先降后升 -> 两条线分叉 -> 过拟合
  acc 图:  train 持续上升, test 先升后平/降 -> 两条线分叉 -> 过拟合

量化判断标准:
  - Train Acc - Test Acc > 10%     -> 明确过拟合
  - Test Loss 连续 5+ epoch 上升   -> 过拟合
  - Train Loss < Test Loss 且差值持续扩大 -> 过拟合

解决方法: 增大 Dropout / weight_decay / 数据增强 / 早停 / 减小模型
"""

# ============================================================================
# 2. 欠拟合 (Underfitting)
# ============================================================================
"""
典型数据特征:
  Train Loss: 下降很慢或卡住 -> 0.69, 0.68, 0.67 ... (居高不下)
  Test Loss:  和 Train Loss 差不多高
  Train Acc:  很低, 远低于预期
  Test Acc:   和 Train Acc 差不多低

可视化判断:
  两条线贴在一起, 都在很差的位置 -> 欠拟合 (不是过拟合!)

量化判断标准:
  - 最终 Train Acc < 预期值 (如 MNIST 低于 95%)
  - Train Acc == Test Acc, 但两者都低 -> 欠拟合

解决方法: 增大模型 / 训练更久 / 增大 lr / 减少正则化 / 检查数据预处理
"""

# ============================================================================
# 3. 梯度消失 (Vanishing Gradient)
# ============================================================================
"""
典型数据特征:
  Train Loss: 前几个 epoch 几乎不动, 或下降极慢
  梯度范数:   浅层 (conv1, conv2) 梯度 ~ 0 (如 1e-7 级别)
  参数更新量: 浅层参数几乎不变

代码诊断 (在 loss.backward() 之后):
  for name, p in model.named_parameters():
      if p.grad is not None:
          n = p.grad.data.norm(2).item()
          if n < 1e-6:
              print(f"WARNING: {name} 梯度几乎为 0 ({n:.2e}) -> 梯度消失!")

层间梯度诊断 (打印每层 grad_norm, 观察衰减趋势):
    conv1.grad: 0.0001   <- 太小! 几乎没学到
    conv2.grad: 0.002
    fc1.grad:   0.05
    fc2.grad:   0.5      <- 正常
  越靠近输入层梯度越小 -> 梯度消失确认

解决方法: 换 ReLU / 加 BatchNorm / 加残差连接 / Kaiming 初始化
"""

# ============================================================================
# 4. 梯度爆炸 (Exploding Gradient)
# ============================================================================
"""
典型数据特征:
  Loss:   突然从正常值 (如 0.3) 跳到 NaN 或 inf
  梯度范数: 某一步突然变成 1000+ 甚至 inf
  参数值:   突然出现极大数值 (如 1e10)

代码诊断 (在 optimizer.step() 之前):
  total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
  if total_norm > 100:
      print(f"WARNING: 梯度爆炸! grad_norm = {total_norm:.1f}")
  if torch.isnan(total_norm):
      print("ERROR: 梯度 NaN! 模型已崩溃, 降低 lr 重新训练")

量化判断:
  - grad_norm > 1000  -> 严重梯度爆炸
  - loss NaN          -> 模型已崩溃
  - loss 单步跳变 10x -> 参数可能已飞出去

解决方法: 梯度裁剪 / 降低 lr / 合适初始化 / 加 BatchNorm
"""

# ============================================================================
# 5. 学习率不当
# ============================================================================
"""
lr 太大: Loss 剧烈震荡 (0.2->0.8->0.3->1.2) / 不降反升 / 直接 NaN
lr 太小: Loss 下降极慢, 每个 epoch 只降 0.001
lr 合适: 前几个 epoch 快速下降(指数级), 然后平缓收敛
         如: 0.69 -> 0.30 -> 0.15 -> 0.08 -> 0.05 -> 0.04 -> 0.03 ...

LR Range Test: 从 lr=1e-7 开始, 每个 batch 指数增大, 最佳 lr = loss 下降最快区间的中间值
"""

# ============================================================================
# 6. 权重分布异常
# ============================================================================
"""
诊断代码:
  for name, param in model.named_parameters():
      if 'weight' in name:
          print(f"{name:20s} mean={param.data.mean():+.4f} "
                f"std={param.data.std():.4f} min={param.data.min():+.4f}")

异常信号:
  - std 持续增大 (每个 epoch 都在变大) -> lr 太大/无正则化
  - 大量权重趋近于 0                  -> L1 正则化过强/梯度消失
  - 某层 std 很大, 某层 std~0         -> 初始化不当/梯度流动问题
  - 权重 NaN                         -> 梯度爆炸
"""

# ============================================================================
# 7. 训练/验证曲线综合判读速查
# ============================================================================
"""
+-----------------------------+-----------------------------------------------+
| 曲线表现                     | 诊断 + 解决方案                                |
+-----------------------------+-----------------------------------------------+
| train↘, test↗ (分叉拉大)    | 过拟合 -> 加 Dropout/weight_decay/早停          |
| train→, test→ (贴一起,都高) | 欠拟合 -> 加模型/加 lr/减正则化                 |
| loss 阶梯状 (平->降->平->降) | 正常的平台期, lr 合适时的正常现象               |
| loss 剧烈锯齿               | batch_size 太小或 lr 太大                      |
| loss 直接 NaN               | 梯度爆炸 -> 梯度裁剪 + 大幅减 lr               |
| loss 几乎不动 (~常数)       | 梯度消失 -> 换 ReLU/加 BN/换初始化             |
| test_acc >> train_acc       | 数据增强开太大, 训练比测试难, 正常             |
| 两者都接近 100%              | 数据泄露! 检查训练/测试集划分                   |
+-----------------------------+-----------------------------------------------+
"""

# ============================================================================
# 8. 完整诊断函数 (可直接复制使用)
# ============================================================================
"""
import torch

def diagnose_training(model, train_loss, test_loss, train_acc, test_acc, epoch,
                      grad_norm=None):
    '''
    每个 epoch 结束后调用, 自动诊断训练状态
    返回 True = 应停止训练, False = 继续
    '''
    # 1. 检查 loss NaN (崩溃)
    if torch.isnan(torch.tensor(train_loss)):
        print("[崩溃] Loss 是 NaN! -> 降低 lr, 加梯度裁剪后重试")
        return True

    # 2. 检查过拟合
    gap = train_acc - test_acc
    if gap > 15:
        print(f"[严重过拟合] Train-Test gap = {gap:.1f}% -> 加正则化/早停!")
    elif gap > 10:
        print(f"[轻微过拟合] Train-Test gap = {gap:.1f}% -> 考虑加 Dropout")

    # 3. 检查欠拟合
    if train_acc < 60 and test_acc < 60 and epoch > 5:
        print(f"[欠拟合] 训练/测试都低 -> 增大模型或 lr")

    # 4. 检查梯度 (需在 train_one_epoch 中记录 grad_norm)
    if grad_norm is not None:
        if grad_norm > 1000:
            print(f"[梯度爆炸] grad_norm={grad_norm:.1f} -> 应用梯度裁剪!")
        elif grad_norm < 1e-4 and epoch > 1:
            print(f"[梯度消失] grad_norm={grad_norm:.2e} -> 检查激活函数/加BN")

    return False
"""




# ============================================================================
# 经典问题 1: 梯度消失 (Vanishing Gradient)
# ============================================================================
"""
问题描述:
  反向传播时，梯度从输出层向输入层逐层连乘。
  如果每一层的梯度都 < 1（如 sigmoid 导数最大只有 0.25），
  到浅层时梯度趋近于 0，浅层参数几乎不更新。

直观理解:
  输出层  ←──  ×0.25  ←──  ×0.25  ←──  ×0.25  ←──  ×0.25  ←──  输入层
           0.25^4 = 0.004   几乎等于 0！

深层网络 → 连乘次数更多 → 梯度消失更严重

核心原因:
  1. 激活函数导数过小: sigmoid' ∈ (0, 0.25], tanh' ∈ (0, 1]
  2. 权重初始化不当: 权重太小 → 每层输出值缩小 → 梯度连乘后消失
  3. 网络过深: 层数越多, 连乘的项越多

解决方案:
  1. ReLU 系列激活函数 → 正区间导数恒为 1, 不会缩小梯度
  2. 残差连接 (ResNet) → 梯度可以跳过中间层直接回传
  3. Batch Normalization → 把每层输出拉到合适的分布, 缓解梯度缩小
  4. 合适的权重初始化 → Kaiming / Xavier 初始化, 保证前向传播方差稳定
  5. Skip Connection → 在浅层和深层间建立直接的信息通道

现象判断:
  - 训练开始后 loss 几乎不动
  - 浅层梯度范数 (gradient norm) 接近 0
  - 深层参数更新, 浅层参数几乎不变
"""

# ============================================================================
# 经典问题 2: 梯度爆炸 (Exploding Gradient)
# ============================================================================
"""
问题描述:
  与梯度消失相反。如果每层的梯度都 > 1，
  连乘后梯度趋近于无穷大，参数一步更新"飞出去"。

直观理解:
  输出层  ←──  ×2.0  ←──  ×2.0  ←──  ×2.0  ←──  ×2.0  ←──  输入层
           2.0^4 = 16  越来越大！

无限增长 → loss 变 NaN → 模型崩溃

核心原因:
  1. 权重初始化过大: 权重大 → 前向传播时每层放大信号 → 反向梯度连乘更大
  2. 学习率过大: 一次更新步长太大, 参数跳到非正常区域
  3. RNN 中时序过长: 时间步展开后的连乘次数多
  4. 损失函数设计缺陷: 如对离群点过敏感

解决方案:
  1. 梯度裁剪 (Gradient Clipping): 限制梯度的最大范数, 超过阈值就等比缩小
     代码: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
  2. 降低学习率: 最直接的缓解手段
  3. 梯度归一化: 对梯度做 LayerNorm 或 BatchNorm 后再更新
  4. 合适的权重初始化: Xavier (sigmoid/tanh) / Kaiming (ReLU)
  5. Batch Normalization: 把激活值稳定在合理范围

现象判断:
  - loss 突然变成 NaN
  - 梯度范数异常大 (>1000)
  - 参数值异常大
"""

# ============================================================================
# 经典问题 3: 参数优化 — 学习率的选择
# ============================================================================
"""
学习率 (Learning Rate, lr) 是训练最重要的超参数。

  太大 → 震荡/发散/loss NaN
  太小 → 收敛慢/卡在局部最优
  适中 → 又快又好

学习率调参策略:
  1. 学习率范围测试 (LR Range Test):
     从极小 lr (1e-7) 开始, 每个 batch 指数增大, 观察 loss 下降最快的区间
  2. 常用起点:
     Adam/AdamW: 1e-4 ~ 1e-3
     SGD:        1e-2 ~ 1e-1
  3. Warmup (预热):
     前 N 步学习率从 0 线性增加到目标值
     原因: 模型刚初始化时不稳定, 大学习率会"踢歪"早期的参数方向
     代码: scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
  4. 学习率衰减:
     StepLR:   每 N 个 epoch × gamma
     Cosine:   按余弦曲线从 lr_max 降到 lr_min, 当前最常用
     ReduceLROnPlateau: 验证 loss 不降时自动衰减

常见学习率调度器:
  ┌─────────────────────┬──────────────────────────────────────┐
  │ 调度器              │ 行为                                  │
  ├─────────────────────┼──────────────────────────────────────┤
  │ StepLR              │ 每 step_size epoch, lr = lr * gamma  │
  │ CosineAnnealingLR   │ 余弦曲线平滑下降, 周期 T_max          │
  │ ReduceLROnPlateau   │ 验证指标停止改善时自动降低             │
  │ OneCycleLR          │ 先升后降, 一个周期内完成训练           │
  │ LinearLR (Warmup)   │ 前几步线性增长, 用于热身               │
  └─────────────────────┴──────────────────────────────────────┘
"""

# ============================================================================
# 经典问题 4: 过拟合 (Overfitting)
# ============================================================================
"""
问题描述:
  训练集上效果很好, 测试集上效果很差。
  模型"背下来了"训练数据而非学到了通用规律。

核心原因:
  1. 模型容量过大 (参数太多, 数据太少)
  2. 训练时间过长
  3. 数据有噪声, 模型连噪声也拟合了
  4. 数据分布不均衡

解决方案:
  1. 增加数据 (数据增强):
     - 随机翻转/旋转/裁剪
     - 颜色抖动 (Color Jitter)
     - Mixup / CutMix: 两张图按比例混合
  2. 正则化:
     - L1 正则化: 加 |w| 到 loss → 产生稀疏权重
     - L2 正则化 (Weight Decay): 加 w^2 到 loss → 缩小所有权重
       代码: optimizer = AdamW(lr=1e-3, weight_decay=0.01)
     - Dropout: 训练时随机丢弃神经元 → 强制每个神经元独立学习
       代码: self.dropout = nn.Dropout(p=0.5)
  3. 早停 (Early Stopping):
     验证集 loss 连续 N 个 epoch 不降 → 停止训练
  4. Batch Normalization: 自带轻微正则化效果
  5. 减小模型: 减少层数/通道数

现象判断:
  - Train Acc >> Test Acc (差 10%+)
  - Train Loss 持续下降, Val Loss 先降后升
"""

# ============================================================================
# 经典问题 5: 欠拟合 (Underfitting)
# ============================================================================
"""
问题描述:
  训练集和测试集效果都不好。模型没学会数据的规律。

核心原因:
  1. 模型容量过小 → 表达能力不够
  2. 特征不足 → 输入信息不够丰富
  3. 训练不充分 → epoch 不够或学习率太小
  4. 数据预处理错误 → 如归一化不当

解决方案:
  1. 增大模型: 加层 / 加宽通道
  2. 减少正则化: 降低 weight_decay / 关闭 Dropout
  3. 增加训练轮次或学习率
  4. 检查数据预处理: 归一化范围是否正确
  5. 使用更复杂的模型架构
"""

# ============================================================================
# 经典问题 6: 内部协变量偏移 (Internal Covariate Shift)
# ============================================================================
"""
问题描述:
  训练过程中, 每层的输入分布在不断变化 (因为前面层的参数在更新)。
  后面的层需要不断适应新的分布, 降低了训练效率。

解决方案: Batch Normalization
  对每个 mini-batch:
    1. 计算均值和方差 (沿 batch 维)
    2. 标准化: x_norm = (x - mean) / sqrt(var + eps)
    3. 缩放偏移: y = gamma * x_norm + beta  (可学习参数)

  好处:
    - 每层输入分布稳定 → 学习率可以设得更大
    - 轻微正则化效果 → 有时可以不用 Dropout
    - 加速收敛 → 同样的 epoch 达到更好的效果

  Transformer/ViT 不用 BN 而用 LayerNorm:
    BN 依赖于 batch 统计, Transformer 的输入长度可变, batch 统计不稳定
    LN 沿特征维归一化, 与 batch 无关
"""

# ============================================================================
# 经典问题 7: 权重初始化不当
# ============================================================================
"""
问题描述:
  初始化太大 → 梯度爆炸; 初始化太小 → 梯度消失。
  不合适的初始化让网络一开始就"走歪了"。

常用初始化方法:
  1. Xavier (Glorot) 初始化 — 适合 sigmoid/tanh
     权重 ~ U(-sqrt(6/(fan_in + fan_out)), sqrt(6/(fan_in + fan_out)))
     原理: 保证输入输出方差相等
     代码: nn.init.xavier_uniform_(tensor)

  2. Kaiming (He) 初始化 — 适合 ReLU 及其变体
     权重 ~ N(0, sqrt(2 / fan_in))
     原理: 考虑到 ReLU 清零一半神经元, 方差需要 ×2 补偿
     代码: nn.init.kaiming_normal_(tensor, mode='fan_in', nonlinearity='relu')

  3. 截断正态 (Truncated Normal) — ViT/Transformer 常用
     从 N(0, std) 采样, 超过 2 个标准差的截断重采样
     代码: nn.init.trunc_normal_(tensor, std=0.02)

  fan_in vs fan_out:
    fan_in  = 上一层神经元的数量
    fan_out = 下一层神经元的数量
  Linear(C_in, C_out): fan_in=C_in, fan_out=C_out
"""

# ============================================================================
# 经典问题 8: 优化器选择
# ============================================================================
"""
优化器 对比:
  ┌──────────┬──────────────────────────────────────────────────┐
  │ SGD      │ θ = θ - lr * ∇θ                                    │
  │          │ 最基础, 收敛慢但泛化好 (适合精调)                  │
  │          │ 常加 Momentum(0.9) 让更新带惯性                    │
  ├──────────┼──────────────────────────────────────────────────┤
  │ Adam     │ 自适应学习率 + Momentum 的结合体                   │
  │          │ 收敛快, 不挑超参, 默认选择                         │
  │          │ 缺点: 有时泛化不如 SGD                             │
  ├──────────┼──────────────────────────────────────────────────┤
  │ AdamW    │ Adam + 解耦的权重衰减 (decoupled weight decay)      │
  │          │ 当前最常用, 学习率和正则化解耦                      │
  │          │ ViT/SAM/ResNet 的训练标配                           │
  └──────────┴──────────────────────────────────────────────────┘

AdamW 参数理解:
  lr=1e-3         — 学习率，控制每一步的步长
  betas=(0.9, 0.999) — β1 动量衰减, β2 二阶矩衰减
  weight_decay=0.01— 权重衰减系数, 直接对权重做衰减 (不是加在 loss 里)
  eps=1e-8        — 数值稳定性, 防除零
"""

# ============================================================================
# 经典问题 9: 损失函数选择
# ============================================================================
"""
分类任务:
  CrossEntropyLoss (交叉熵): 分类标配
    loss = -Σ y_true * log(y_pred)
    自带 softmax, 模型输出 logits 即可, 不要在前面手动加 softmax

  BCEWithLogitsLoss (二分类交叉熵):
    自带 sigmoid, 输入 logits 即可
    多标签分类 (一个样本属于多个类) 用这个而非 CrossEntropy

回归任务:
  MSELoss (均方误差):   对离群点敏感, 梯度和大误差成正比
  L1Loss (MAE, 平均绝对误差): 对离群点鲁棒, 梯度恒定
  SmoothL1Loss (Huber):  误差小时用 MSE, 大时用 L1, 兼顾两者

分割任务:
  Dice Loss: 直接优化 IoU, 适合类别不平衡
    常与 CrossEntropy 组合使用 (SAM mask_decoder 训练就用 CE + Dice)
"""

# ============================================================================
# 训练调试速查表
# ============================================================================
"""
┌────────────────────────┬──────────────────────────────────────────┐
│ 现象                   │ 可能原因 + 解决                           │
├────────────────────────┼──────────────────────────────────────────┤
│ Loss 不下降            │ lr 太小 → 增大 ×3~10                      │
│                        │ 梯度消失 → 换 ReLU/加 BN/加残差           │
│                        │ 数据有问题 → 检查预处理                   │
├────────────────────────┼──────────────────────────────────────────┤
│ Loss 震荡剧烈          │ lr 太大 → 减小 ÷3~5                       │
│                        │ batch_size 太小 → 增大                    │
├────────────────────────┼──────────────────────────────────────────┤
│ Loss 变成 NaN          │ lr 太大 → 大幅减小                        │
│                        │ 梯度爆炸 → 加梯度裁剪                     │
│                        │ 除零 → 检查 eps 参数                      │
├────────────────────────┼──────────────────────────────────────────┤
│ 训练好 验证差 (过拟合) │ 增大 weight_decay/Dropout                 │
│                        │ 数据增强 / 早停 / 减小模型                │
├────────────────────────┼──────────────────────────────────────────┤
│ 训练差 验证也差 (欠拟合)│ 增加层数/宽度 / 减小正则化                 │
│                        │ 加大 lr 或 epoch                          │
├────────────────────────┼──────────────────────────────────────────┤
│ 验证 Loss 后期回升     │ 过拟合 → 早停 / ReduceLROnPlateau         │
├────────────────────────┼──────────────────────────────────────────┤
│ 显存不足 (OOM)         │ 减小 batch_size (64→32→16)                │
│                        │ 减小模型 / 使用梯度累积                   │
├────────────────────────┼──────────────────────────────────────────┤
│ 训练很慢               │ num_workers 太小 → 增大                   │
│                        │ 用了很重的数据增强 → 简化                 │
│                        │ 考虑混合精度训练 (AMP)                    │
└────────────────────────┴──────────────────────────────────────────┘
"""