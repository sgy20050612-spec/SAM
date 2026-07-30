"""
输入图片 (3×224×224)                        3=RGB通道, 224=图像高, 224=图像宽 (像素)
        │
        ▼
┌─────────────────────────────────────────┐
│  Step 1: 切块（Patchify）               │
│  把图片切成 14×14 = 196 个小方块        │ 14=224/16, 196=14*14=patch总数
│  每个小方块是 16×16×3 = 768 维          │ 16=patch边长, 3=RGB, 768=16*16*3
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│  Step 2: Patch Embedding                │
│  把每个小方块线性映射到固定的向量维度      │
│  196 个块 → 196 个向量（每个 768 维）     │ 196=序列长度, 768=嵌入维度
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│  Step 3: 添加 [CLS] 标记                 │
│  在序列最前面加一个可学习的特殊向量        │
│  用于最终分类                           │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│  Step 4: 加入位置编码（Positional）     │
│  告诉模型每个小方块在图片里的位置       │
│  如 "第3行第5列"                        │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│  Step 5: 输入 Transformer Encoder       │
│  重复 L 层（如 12 层）                  │ 12=Transformer层数 (depth)
│  每层：Self-Attention → FFN → 残差连接  │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│  Step 6: 分类输出                       │
│  取出 [CLS] 标记对应的输出              │
│  通过分类头输出类别概率（如 1000 类）   │ 1000=ImageNet类别数
└─────────────────────────────────────────┘
        │
        ▼
输出：类别预测
"""


"""

QKV的详细计算过程

Attention 直觉: 2 个 patch × 2 维向量的手算例子
=================================================

假设只有 2 个 patch, 每个用 2 维向量表示 (768→2 便于手算):

  patch_0: x0 = [2, 3]    (比如: 猫的耳朵)
  patch_1: x1 = [1, 0]    (比如: 猫的身子)

--- Step 0: 投影出 Q, K, V (同一个 x 配三套不同的权重) ---

  Q0 = W_q @ [2,3] = [0.8, 1.5]   K0 = W_k @ [2,3] = [1.2, 0.3]   V0 = W_v @ [2,3] = [2.1, 2.8]
  Q1 = W_q @ [1,0] = [0.6, 0.2]   K1 = W_k @ [1,0] = [0.5, 0.9]   V1 = W_v @ [1,0] = [1.0, 0.4]

  三种投影功能不同: Q="我想找什么"  K="我是什么类型"  V="我的实际内容"

--- Step 1: Q_i · K_j = 注意力分数 (判断关系) ---

           K0=[1.2,0.3]   K1=[0.5,0.9]
  Q0=[0.8,1.5]:  1.41           1.75       ← patch_0 更关注 patch_1 (耳朵找身子)
  Q1=[0.6,0.2]:  0.78           0.48       ← patch_1 更关注 patch_0 (身子找耳朵)

  分数矩阵:         patch_0  patch_1
          patch_0    1.41     1.75
          patch_1    0.78     0.48

--- Step 2: softmax → 百分比权重 (归一化, 和为1) ---

  patch_0: softmax([1.41, 1.75]) = [0.416, 0.584]
  patch_1: softmax([0.78, 0.48]) = [0.574, 0.426]

--- Step 3: 权重 × V → 加权求和 = 新表示 ---

  new_0 = 0.416×V0 + 0.584×V1
        = 0.416×[2.1,2.8] + 0.584×[1.0,0.4]
        = [1.46, 1.40]           ← 耳朵融入了身子的 V

  new_1 = 0.574×V0 + 0.426×V1
        = 0.574×[2.1,2.8] + 0.426×[1.0,0.4]
        = [1.63, 1.78]           ← 身子融入了耳朵的 V

--- 总结 ---

  Q·K 判断关系强弱 → softmax 转百分比 → 百分比 × V 加权求和
  Q和K是"裁判"(决定关注谁), V是"选手"(提供被提取的实际内容).

"""

import torch                         # 深度学习框架
import torch.nn as nn                # 神经网络模块 (Linear, LayerNorm, Conv2d 等)
import torch.nn.functional as F      # 函数式API (softmax, gelu 等)
import math                          # 数学函数 (sqrt)


# ============================================================
# Step 1+2: Patch Embedding -- 切块 + 线性映射
# ============================================================
class PatchEmbed(nn.Module):
    """把图像切成 patch，每个 patch 投影为向量

    img_size=224 是像素尺寸 (不是 patch 数!)
    224 / 16 = 14, 所以切成 14x14 = 196 个 patch
    """

    def __init__(self, 
                 img_size=224,        # img_size=224: 输入图像边长 (像素), 标准 ViT 用 224
                 patch_size=16,             # patch_size=16: 每个 patch 的边长 (像素), 224/16=14 个 patch/边
                 in_chans=3,                # in_chans=3: 输入通道数, RGB 彩色图为 3
                 embed_dim=768):            # embed_dim=768: 每个 patch 映射到的向量维度, ViT-Base 用 768
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        # 224px / 16px = 14 patches per side, total 14*14 = 196  (=序列长度, 类比 NLP 中句子有 196 个词)
        self.num_patches = (img_size // patch_size) ** 2   # **2 是因为高度和宽度两个维度都要切

        # Conv2d 一步完成 "切块 + 线性映射":
        #   kernel_size=16, stride=16 → 每个 16x16 区域被卷积核"看一次"，无重叠
        #   in_chans=3 → embed_dim=768 → 每个 16x16x3=768 维的小块被映射为一个 768 维向量
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """[B, 3, 224, 224] -> [B, 196, 768]

        B = batch_size (批量大小, 一次处理多少张图)
        3 = RGB 三通道
        224 = 图像高/宽 (像素)
      2
        768 = 每个 patch 的嵌入向量维度
        """
        x = self.proj(x)                # Conv2d: [B, 3, 224, 224] -> [B, 768, 14, 14]
                                        #   768=embed_dim (通道数变成嵌入维度)
                                        #   14=img_size/patch_size=224/16 (空间尺寸)

        x = x.flatten(2)                # flatten(dim=2): [B, 768, 14, 14] -> [B, 768, 196]
                                        #   dim=2 表示从第2维开始压平, 即把 14x14=196 压成一维
                                        #   196 = 序列长度 (14*14)

        x = x.transpose(1, 2)           # transpose(dim1=1, dim2=2): [B, 768, 196] -> [B, 196, 768]
                                        #   把通道维和序列维交换, 变成 Transformer 的输入格式
                                        #   (batch, seq_len, embed_dim)
        return x


# ============================================================
# Step 4: Position Embedding -- 位置编码
# ============================================================
class PositionEmbedding(nn.Module):
    """可学习的位置编码, 给每个 patch 加一个唯一的位置向量"""

    def __init__(self, 
                 num_patches,            # num_patches: 位置数 = patch数 + CLS, 如 196+1=197
                 embed_dim=768):               # embed_dim=768: 每个位置向量的维度, 必须和 patch 嵌入维度一致
        super().__init__()
        # (1, num_patches, embed_dim): 1=仅一个样本的位置编码 (广播到 batch), N=位置数, C=维度
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)  # std=0.02: 截断正态分布的标准差, 小值初始化防梯度爆炸

    def forward(self, x):
        return x + self.pos_embed           # 逐元素相加: 每个 patch 的 768 维向量 + 该位置的 768 维编码


# ============================================================
# Step 5: Multi-Head Self-Attention -- 多头自注意力
# ============================================================
class MultiHeadSelfAttention(nn.Module):
    """
    每个 patch 和所有 patch 算相似度, 加权聚合
    Q = 我要找什么, K = 你有什么, V = 你的实际内容
    """

    def __init__(self, embed_dim=768,         # embed_dim=768: 嵌入维度, 也是 Q,K,V 投影的维度
                 num_heads=12,                # num_heads=12: 注意力头数, embed_dim/num_heads=768/12=64=每个头的维度
                 dropout=0.1):                # dropout=0.1: 随机丢弃 10% 的注意力权重, 防过拟合
        super().__init__()
        assert embed_dim % num_heads == 0     # 必须整除: 768/12=64 OK, 如果 768/7 就报错
        self.num_heads = num_heads            # 12: 12 个注意力头并行计算
        self.head_dim = embed_dim // num_heads  # 768/12=64: 每个头的维度, 越小计算越快
        self.scale = math.sqrt(self.head_dim)   # sqrt(64)=8: 缩放因子, 防止内积过大导致 softmax 饱和

        # 合并 Q,K,V 三个投影为一个 Linear: 768 -> 768*3=2304, 比分开三个线性层更高效
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)  # embed_dim*3: 同时输出 Q,K,V 三个矩阵
        self.out_proj = nn.Linear(embed_dim, embed_dim) # 768->768: 多头拼接后的输出投影
        self.dropout = nn.Dropout(dropout)               # dropout=0.1: 训练时随机屏蔽 10% 神经元

    def forward(self, x):
        # x: [B, N, C] = [batch, 196或197, 768]
        B, N, C = x.shape                    # B=batch_size, N=序列长度(196/197), C=embed_dim(768)

        # 1. Q,K,V 投影并分头 -> [3, B, num_heads, N, head_dim]
        #                     3=Q,K,V三分量, B=batch, H=12头, N=seq_len, D=64头维度
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        #   reshape: [B, N, 2304] -> [B, N, 3, 12, 64]
        #   2304 = 3 * 12 * 64 = 3 * embed_dim (Q,K,V各768)

        qkv = qkv.permute(2, 0, 3, 1, 4)     # 重排维度: [B,N,3,12,64] -> [3,B,12,N,64]
        q, k, v = qkv[0], qkv[1], qkv[2]     # 各取各的: q/k/v 都是 [B, 12, N, 64]

        # 2. Attention: softmax(Q @ K^T / sqrt(d))
        #    q @ k^T: [B,12,N,64] @ [B,12,64,N] -> [B,12,N,N] (每个头内的所有token两两计算相似度)
        #    除以 self.scale=8: 防止数值过大, softmax 后梯度消失
        attn = (q @ k.transpose(-2, -1)) / self.scale    # transpose(-2,-1)=转置最后两维
        attn = F.softmax(attn, dim=-1)       # dim=-1: 沿最后维度(每个query对所有key)做归一化, 和为1
        attn = self.dropout(attn)            # 随机丢弃注意力权重, 增强鲁棒性

        # 3. Attention @ V
        #    [B,12,N,N] @ [B,12,N,64] -> [B,12,N,64] (加权聚合所有token的信息)
        out = attn @ v

        # 4. 合并多头 + 输出投影
        #    transpose(1,2): [B,12,N,64] -> [B,N,12,64]
        #    view(B,N,C):    [B,N,12,64] -> [B,N,768]  (12*64=768)
        out = out.transpose(1, 2).contiguous().view(B, N, C)
        out = self.out_proj(out)             # Linear: [B,N,768] -> [B,N,768] (保持维度不变)
        return out


# ============================================================
# Step 5: MLP (Feed-Forward Network) -- 前馈网络
# ============================================================
class MLP(nn.Module):
    """两层全连接, 中间扩展 4 倍, GELU 激活"""

    def __init__(self, embed_dim=768,         # embed_dim=768: 输入/输出维度
                 mlp_ratio=4.0,               # mlp_ratio=4.0: 中间层扩展倍数, 768*4=3072
                 dropout=0.1):                # dropout=0.1: 训练时随机丢弃 10%
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)  # 768*4=3072: 中间隐藏层维度, 4x 扩展给模型更大容量
        self.fc1 = nn.Linear(embed_dim, hidden_dim)    # 768 -> 3072: 升维
        self.fc2 = nn.Linear(hidden_dim, embed_dim)    # 3072 -> 768: 降维, 恢复原维度
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)                      # [B,N,768] -> [B,N,3072]
        x = F.gelu(x)                        # GELU: 比 ReLU 更平滑, ViT/Transformer 标配
        x = self.dropout(x)                  # 防过拟合
        x = self.fc2(x)                      # [B,N,3072] -> [B,N,768]
        x = self.dropout(x)                  # 又一次 dropout, 增强泛化
        return x


# ============================================================
# Step 5: Transformer Encoder Block -- 组装 Attention + MLP
# ============================================================
class TransformerBlock(nn.Module):
    """
    Pre-Norm 结构: LN 放在子层前而不是后面, 训练更稳定
      x = x + Attention(LN(x))   <- 自注意力 + 残差
      x = x + MLP(LN(x))         <- FFN + 残差
    """

    def __init__(self, embed_dim=768,         # embed_dim=768: 整个 Block 内统一的嵌入维度
                 num_heads=12,                # num_heads=12: 注意力头数
                 mlp_ratio=4.0,               # mlp_ratio=4.0: MLP 隐藏层扩展 4 倍
                 dropout=0.1):                # dropout=0.1: 统一的 dropout 概率
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)  # 第一个 LN: Self-Attention 前的归一化, 沿最后一维做标准化
        self.norm2 = nn.LayerNorm(embed_dim)  # 第二个 LN: MLP 前的归一化
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)

    def forward(self, x):
        # 残差连接: 原始输入 x 直接加到子层输出上
        #   好处: 梯度可以"抄近道"直接流回输入, 即使堆 100 层也能有效训练
        x = x + self.attn(self.norm1(x))    # 子层1: LayerNorm -> Self-Attention -> +x(残差)
        x = x + self.mlp(self.norm2(x))     # 子层2: LayerNorm -> MLP -> +x(残差)
        return x


# ============================================================
# Step 1-6: 完整 ViT -- 组装所有组件
# ============================================================
class ViT(nn.Module):
    """Vision Transformer -- 图像分类"""

    def __init__(self, img_size=224,          # img_size=224: 输入图像边长(像素)
                 patch_size=16,               # patch_size=16: 每个 patch 边长(像素), 224/16=14
                 in_chans=3,                  # in_chans=3: RGB 三通道
                 num_classes=1000,            # num_classes=1000: ImageNet-1K 分类类别数
                 embed_dim=768,               # embed_dim=768: 嵌入维度 (ViT-Base)
                 depth=12,                    # depth=12: Transformer 层数 (ViT-Base 用 12 层)
                 num_heads=12,                # num_heads=12: 注意力头数 (ViT-Base 用 12 头)
                 mlp_ratio=4.0,               # mlp_ratio=4.0: MLP 中间层扩展倍数
                 dropout=0.1):                # dropout=0.1: 全模型统一的 dropout 率
        super().__init__()

        # Step 1+2: Patch Embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)

        # Step 3: CLS Token  (Classification Token)
        #   (1, 1, 768): 1个样本, 1个CLS token, 768维向量
        #   这个向量在训练过程中学习聚合全局信息，最终用于分类
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)  # std=0.02: 小值正态初始化

        # Step 4: Position Embedding
        #   num_patches + 1: 196个patch + 1个CLS = 197 个位置需要编码
        self.pos_embed = PositionEmbedding(self.patch_embed.num_patches + 1, embed_dim)
        self.pos_drop = nn.Dropout(dropout)

        # Step 5: Transformer Encoder x depth 层
        #   depth=12: 堆叠 12 个完全相同的 TransformerBlock
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)  # 最后的 LayerNorm, 在分类头之前

        # Step 6: 分类头
        #   768 -> num_classes: 从 CLS token 的 768 维向量映射到 1000 个类别分数
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        """权重初始化: Linear 层用截断正态, bias 初始化为 0"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)  # std=0.02: 小方差, 训练前期更稳定
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """[B, 3, 224, 224] -> [B, 1000]"""
        B = x.shape[0]                       # B=batch_size: 一次处理多少张图

        # Step 1+2: 切块 + 映射
        x = self.patch_embed(x)                           # [B, 196, 768]

        # Step 3: 添加 CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)     # [1,1,768] -> [B,1,768], -1=保持该维度不变
        x = torch.cat([cls_tokens, x], dim=1)              # dim=1: 沿序列维度拼接 -> [B, 197, 768]
                                                           #   197 = 1个CLS + 196个patch

        # Step 4: 位置编码
        x = self.pos_embed(x)
        x = self.pos_drop(x)

        # Step 5: Transformer Encoder
        for block in self.blocks:
            x = block(x)                    # 每层输入输出形状不变: [B, 197, 768]
        x = self.norm(x)

        # Step 6: CLS -> 分类头
        x = x[:, 0]                         # [B, 768]: 取第 0 个位置 (CLS token), 丢弃所有 patch token
        x = self.head(x)                    # [B, 768] -> [B, 1000]: 全连接输出各类别 logits
        return x


# ============================================================
# 训练示例 (CIFAR-10)
# ============================================================
if __name__ == "__main__":
    from torch.utils.data import DataLoader         # 批量数据加载
    from torchvision import datasets, transforms    # 数据集和预处理
    from torch import optim                         # 优化器

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- 数据 ----
    transform = transforms.Compose([
        transforms.Resize(224),              # CIFAR-10 原图 32x32, 放大到 224x224 适配 ViT
        transforms.ToTensor(),               # PIL Image / numpy -> torch.Tensor, 值从 [0,255] 映射到 [0,1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        #   mean=(0.5,0.5,0.5): 每个通道减 0.5, 使数据以 0 为中心
        #   std=(0.5,0.5,0.5):  每个通道除 0.5, 使数据范围从 [0,1] 变为 [-1,1]
    ])
    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_dataset  = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    #   batch_size=64: 每批 64 张图, shuffle=True: 每个 epoch 随机打乱, num_workers=2: 用 2 个子进程加载数据
    test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False, num_workers=2)

    # ---- 模型 (小ViT: embed_dim=384, depth=6, heads=6) ----
    #   384 = 小 ViT 的嵌入维度 (标准 ViT-Base 的一半)
    #   6 = 仅 6 层 Transformer (标准 ViT-Base 的一半)
    #   6 = 6 个注意力头 (标准 ViT-Base 的一半)
    #   10 = CIFAR-10 有 10 个类别 (飞机/汽车/鸟/猫/鹿/狗/青蛙/马/船/卡车)
    model = ViT(img_size=224, patch_size=16, in_chans=3, num_classes=10,
                embed_dim=384, depth=6, num_heads=6).to(device)
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    # ---- 训练 ----
    criterion = nn.CrossEntropyLoss()        # 交叉熵损失: 分类任务标准损失函数
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    #   lr=1e-3=0.001: 学习率, 控制每一步参数更新的步长
    #   weight_decay=0.05: 权重衰减 (L2正则化), 防止过拟合, 0.05 是常用值
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    #   T_max=10: 10 个 epoch 内, 学习率按余弦曲线从 lr 降到 0

    for epoch in range(1, 11):               # 1 到 10 个 epoch, 每个 epoch 完整过一遍训练集
        model.train()                        # 设置训练模式 (开启 dropout)
        train_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()            # 清空上一步的梯度 (PyTorch 默认累积梯度)
            loss = criterion(model(images), labels)
            loss.backward()                  # 反向传播: 自动计算所有参数的梯度
            optimizer.step()                 # 用梯度 + AdamW 算法更新参数
            train_loss += loss.item()
            _, pred = model(images).max(1)   # max(dim=1): 沿类别维度取最大值, 得到预测类别
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)
        scheduler.step()                     # 每个 epoch 结束后更新学习率

        model.eval()                         # 设置评估模式 (关闭 dropout)
        test_correct, test_total = 0, 0
        with torch.no_grad():                # 不计算梯度, 节省显存和计算
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                _, pred = model(images).max(1)
                test_correct += pred.eq(labels).sum().item()
                test_total += labels.size(0)

        print(f"Epoch {epoch:2d} | Train Loss: {train_loss/len(train_loader):.4f} | "
              f"Train Acc: {100.*correct/total:.2f}% | Test Acc: {100.*test_correct/test_total:.2f}%")
    print("Done.")





# ============================================================
# ViT 完整流程表
# ============================================================
# 步骤    │ 组件名称                             │ 输入维度        │ 操作说明                                                   │ 输出维度
# ────────┼──────────────────────────────────────┼─────────────────┼───────────────────────────────────────────────────────────┼────────────────
# 第1步   │ 图像分块（Patch Partition）           │ [224, 224, 3]   │ 切成14×14=196个Patch，每个16×16×3，拉平成向量                │ [196, 768]
# 第2步   │ 线性投影（Linear Projection）         │ [196, 768]      │ 全连接层做线性变换（若Patch拉平后维度不等于768则做映射）        │ [196, 768]
# 第3步   │ 拼接[CLS]标记（Append [CLS] Token）   │ [196, 768]      │ 在序列最前面拼接一个可学习的[CLS]向量                         │ [197, 768]
# 第4步   │ 位置编码（Positional Embedding）      │ [197, 768]      │ 加上可学习的1D位置编码矩阵（逐元素相加）                       │ [197, 768]
# 第5步   │ Dropout                              │ [197, 768]      │ 对序列整体做Dropout，防止过拟合                                │ [197, 768]
# 第6步   │ LayerNorm（层归一化）                  │ [197, 768]      │ 对每个token的768维做归一化（均值为0，方差为1）                  │ [197, 768]
# 第7步   │ 多头自注意力（Multi-Head Self-Attention）│ [197, 768]   │ Q=K=V=X，拆12个头（每头64维），计算缩放点积注意力，拼接后线性投影 │ [197, 768]
# 第8步   │ 残差连接（Residual Connection）        │ [197, 768]      │ 第7步输出 + 第6步输入（LN前的原始输入），逐元素相加             │ [197, 768]
# 第9步   │ LayerNorm（层归一化）                  │ [197, 768]      │ 对每个token的768维做归一化                                    │ [197, 768]
# 第10步  │ 前馈网络（MLP / FFN）                  │ [197, 768]      │ 全连接升维768→3072，GELU激活，全连接降维3072→768              │ [197, 768]
# 第11步  │ 残差连接（Residual Connection）        │ [197, 768]      │ 第10步输出 + 第9步输入（LN前的原始输入），逐元素相加            │ [197, 768]
# 第12步  │ 重复第6步至第11步 L次                  │ [197, 768]      │ ViT-Base重复12次，ViT-Large重复24次                           │ [197, 768]
# 第13步  │ 提取[CLS]标记（Extract [CLS] Token）   │ [197, 768]      │ 只取序列第一个token（[CLS]）对应的那一行向量，其余丢弃          │ [768]
# 第14步  │ 分类头（Classification Head）          │ [768]           │ 一层或两层全连接网络，输出类别logits，接Softmax得到概率          │ [1000]（ImageNet）

