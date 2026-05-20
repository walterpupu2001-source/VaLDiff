"""
Downstream-prediction model architectures.

All eight predictor classes are **copied verbatim** from the original
Block-3 script provided by the author. The implementations may differ
from textbook versions of their namesake architectures (PatchTST,
Informer, iTransformer) — what matters for this codebase is faithful
reproduction of the Block-3 versions used to produce the paper results.

Do NOT "improve" these — any change to architecture, hidden sizes,
dropout, activation, or default kwargs will break numerical reproducibility.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. MLP — Block 3 verbatim
# ============================================================
class MLPPredictor(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# 2. CNN — Block 3 verbatim
# ============================================================
class CNNPredictor(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * (input_size // 4), 128), nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        x = self.conv(x.unsqueeze(1))
        return self.fc(x.view(x.size(0), -1))


# ============================================================
# 3. RNN (GRU) — Block 3 verbatim
# ============================================================
class RNNPredictor(nn.Module):
    """GRU-based RNN Predictor"""
    def __init__(self, input_size, output_size, hidden=128):
        super().__init__()
        self.rnn = nn.GRU(1, hidden, 2, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden, output_size)

    def forward(self, x):
        out, _ = self.rnn(x.unsqueeze(-1))
        return self.fc(out[:, -1, :])


# ============================================================
# 4. LSTM — Block 3 verbatim
# ============================================================
class LSTMPredictor(nn.Module):
    """
    LSTM Predictor (Hochreiter & Schmidhuber, Neural Computation 1997)
    经典的长短期记忆网络，与GRU对比
    """
    def __init__(self, input_size, output_size, hidden=128):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, num_layers=2, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden, output_size)

    def forward(self, x):
        # x: (batch, seq_len)
        out, (h_n, c_n) = self.lstm(x.unsqueeze(-1))  # (batch, seq_len, hidden)
        return self.fc(out[:, -1, :])  # 取最后一个时间步的隐状态


# ============================================================
# 5. Vanilla Transformer — Block 3 verbatim
# ============================================================
class TransformerPredictor(nn.Module):
    """Vanilla Transformer Encoder (Vaswani et al., NeurIPS 2017)"""
    def __init__(self, input_size, output_size, d_model=64):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, input_size, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(d_model, nhead=4, dim_feedforward=128, batch_first=True, dropout=0.1)
        self.transformer = nn.TransformerEncoder(enc, num_layers=2)
        self.fc = nn.Linear(d_model, output_size)

    def forward(self, x):
        x = self.input_proj(x.unsqueeze(-1)) + self.pos_enc
        x = self.transformer(x)
        return self.fc(x[:, -1, :])


# ============================================================
# 6. PatchTST — Block 3 verbatim (Nie et al., ICLR 2023)
# ============================================================
class PatchTSTPredictor(nn.Module):
    """
    PatchTST (Nie et al., ICLR 2023)
    "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers"
    核心思想：将时间序列分成patch，每个patch作为一个token
    """
    def __init__(self, input_size, output_size, patch_len=16, stride=8, d_model=64, n_heads=4, n_layers=2):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.input_size = input_size

        # 计算patch数量
        self.num_patches = max(1, (input_size - patch_len) // stride + 1)

        # Patch embedding: 将每个patch线性投影到d_model维
        self.patch_embedding = nn.Linear(patch_len, d_model)

        # 可学习的位置编码
        self.pos_enc = nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            batch_first=True,
            dropout=0.1,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Layer norm (PatchTST原文使用)
        self.norm = nn.LayerNorm(d_model)

        # Output head: flatten所有patch的表示
        self.fc = nn.Linear(d_model * self.num_patches, output_size)

    def forward(self, x):
        # x: (batch, seq_len)
        batch_size, seq_len = x.shape

        # 如果序列太短，padding到至少一个patch
        if seq_len < self.patch_len:
            x = F.pad(x, (0, self.patch_len - seq_len), mode='replicate')
            seq_len = self.patch_len

        # 创建patches: (batch, num_patches, patch_len)
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        actual_num_patches = patches.shape[1]

        # Patch embedding: (batch, num_patches, d_model)
        x = self.patch_embedding(patches)

        # 添加位置编码（处理patch数量不匹配的情况）
        if actual_num_patches <= self.num_patches:
            x = x + self.pos_enc[:, :actual_num_patches, :]
        else:
            pos_enc_interp = F.interpolate(
                self.pos_enc.transpose(1, 2),
                size=actual_num_patches,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
            x = x + pos_enc_interp

        # Transformer编码
        x = self.transformer(x)
        x = self.norm(x)

        # Flatten并预测
        x = x.reshape(batch_size, -1)

        # 如果维度不匹配，使用自适应池化
        expected_dim = self.fc.in_features
        if x.shape[1] != expected_dim:
            x = F.adaptive_avg_pool1d(x.unsqueeze(1), expected_dim).squeeze(1)

        return self.fc(x)


# ============================================================
# 7. Informer — Block 3 verbatim (Zhou et al., AAAI 2021)
# ============================================================
class ProbSparseAttention(nn.Module):
    """
    ProbSparse Self-Attention (Informer的核心组件)
    通过选择top-u个最"活跃"的query来减少计算量
    参考: Zhou et al., AAAI 2021
    """
    def __init__(self, d_model, n_heads, factor=5, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.factor = factor

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, L, _ = x.shape
        H, D = self.n_heads, self.d_k

        # 线性投影
        Q = self.W_Q(x).view(B, L, H, D).transpose(1, 2)  # (B, H, L, D)
        K = self.W_K(x).view(B, L, H, D).transpose(1, 2)
        V = self.W_V(x).view(B, L, H, D).transpose(1, 2)

        # 计算attention scores
        scale = 1.0 / math.sqrt(D)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # (B, H, L, L)

        # ProbSparse核心：计算query的"稀疏性度量" M(q_i)
        # M(q_i) = max_j(q_i * k_j) - mean_j(q_i * k_j)
        # 选择M值最大的top-u个query
        u = max(1, L // self.factor)  # 采样的query数量

        M = scores.max(dim=-1)[0] - scores.mean(dim=-1)  # (B, H, L)
        top_indices = M.topk(u, dim=-1)[1]  # (B, H, u)

        # 对所有位置计算attention（简化实现，保持输出形状一致）
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)  # (B, H, L, D)
        out = out.transpose(1, 2).contiguous().view(B, L, -1)

        return self.out_proj(out)


class InformerPredictor(nn.Module):
    """
    Informer (Zhou et al., AAAI 2021 Best Paper)
    "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting"
    核心创新：
    1. ProbSparse Self-Attention: O(L log L) 复杂度
    2. Self-attention Distilling: 金字塔式降采样
    """
    def __init__(self, input_size, output_size, d_model=64, n_heads=4, n_layers=2, factor=5):
        super().__init__()
        self.input_size = input_size
        self.d_model = d_model

        # Input embedding
        self.input_proj = nn.Linear(1, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, input_size, d_model) * 0.02)

        # Encoder layers with ProbSparse attention and distilling
        self.encoder_layers = nn.ModuleList()
        self.distill_convs = nn.ModuleList()

        current_len = input_size
        for i in range(n_layers):
            self.encoder_layers.append(nn.ModuleDict({
                'attn': ProbSparseAttention(d_model, n_heads, factor),
                'ffn': nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(d_model * 4, d_model),
                    nn.Dropout(0.1)
                ),
                'norm1': nn.LayerNorm(d_model),
                'norm2': nn.LayerNorm(d_model)
            }))

            # Distilling: 每层后序列长度减半（除了最后一层）
            if i < n_layers - 1:
                self.distill_convs.append(
                    nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)
                )
                current_len = (current_len + 1) // 2

        self.final_len = current_len
        self.fc = nn.Linear(d_model * current_len, output_size)

    def forward(self, x):
        # x: (batch, seq_len)
        B, L = x.shape

        # Input embedding + positional encoding
        x = self.input_proj(x.unsqueeze(-1))  # (B, L, d_model)

        # 处理位置编码长度不匹配
        if L <= self.input_size:
            x = x + self.pos_enc[:, :L, :]
        else:
            pos_enc_interp = F.interpolate(
                self.pos_enc.transpose(1, 2), size=L, mode='linear', align_corners=False
            ).transpose(1, 2)
            x = x + pos_enc_interp

        # Encoder with distilling
        for i, layer in enumerate(self.encoder_layers):
            # Pre-norm architecture
            x = x + layer['attn'](layer['norm1'](x))
            x = x + layer['ffn'](layer['norm2'](x))

            # Distilling (降采样)
            if i < len(self.distill_convs):
                x = self.distill_convs[i](x.transpose(1, 2)).transpose(1, 2)
                x = F.gelu(x)

        # Flatten并预测
        x = x.reshape(B, -1)

        # 自适应处理维度不匹配
        expected_dim = self.fc.in_features
        if x.shape[1] != expected_dim:
            x = F.adaptive_avg_pool1d(x.unsqueeze(1), expected_dim).squeeze(1)

        return self.fc(x)


# ============================================================
# 8. iTransformer — Block 3 verbatim (Liu et al., ICLR 2024)
# ============================================================
class iTransformerPredictor(nn.Module):
    """
    iTransformer (Liu et al., ICLR 2024)
    "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting"
    核心思想：反转attention方向
    - 传统Transformer: 在时间维度做attention
    - iTransformer: 在变量维度做attention，用FFN捕捉时间依赖

    对于单变量：将序列分成多个segment，每个segment视为一个"伪变量"
    """
    def __init__(self, input_size, output_size, n_segments=8, d_model=64, n_heads=4, n_layers=2):
        super().__init__()
        self.n_segments = n_segments
        self.segment_len = max(1, input_size // n_segments)
        self.actual_input = self.segment_len * n_segments

        # Variate embedding: 将每个segment（伪变量）的时间序列embed为一个token
        self.variate_embed = nn.Linear(self.segment_len, d_model)

        # 可学习的variate token（类似CLS token的作用）
        self.variate_pos = nn.Parameter(torch.randn(1, n_segments, d_model) * 0.02)

        # Inverted Transformer: 在segment/variate维度做attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            batch_first=True,
            dropout=0.1,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Layer norm
        self.norm = nn.LayerNorm(d_model)

        # Projection head
        self.fc = nn.Linear(d_model * n_segments, output_size)

    def forward(self, x):
        # x: (batch, seq_len)
        B, L = x.shape

        # 调整序列长度以适应segment划分
        if L < self.actual_input:
            # Padding
            x = F.pad(x, (0, self.actual_input - L), mode='replicate')
        elif L > self.actual_input:
            # 使用插值缩放
            x = F.interpolate(x.unsqueeze(1), size=self.actual_input, mode='linear', align_corners=False).squeeze(1)

        # 将序列分成n_segments个segment: (B, n_segments, segment_len)
        x = x.view(B, self.n_segments, self.segment_len)

        # Variate embedding: 每个segment embed为一个d_model维的token
        # 这是iTransformer的核心 - 整个时间片段作为一个token的"特征"
        x = self.variate_embed(x)  # (B, n_segments, d_model)

        # 添加variate位置编码
        x = x + self.variate_pos

        # Inverted attention: 在variate/segment维度做attention
        # 这让模型学习不同时间段之间的关系
        x = self.transformer(x)
        x = self.norm(x)

        # Flatten并预测
        x = x.reshape(B, -1)
        return self.fc(x)


# ============================================================
# Registry: predictor name → class
# ============================================================
PREDICTOR_REGISTRY = {
    "MLP": MLPPredictor,
    "CNN": CNNPredictor,
    "RNN": RNNPredictor,
    "LSTM": LSTMPredictor,
    "Transformer": TransformerPredictor,
    "PatchTST": PatchTSTPredictor,
    "Informer": InformerPredictor,
    "iTransformer": iTransformerPredictor,
}
