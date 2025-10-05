import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import gaussian_filter


class HeatmapGenerator:
    """
    将fixation坐标转换为高斯热图
    论文方法: 每个fixation建模为σ=1°视角的高斯分布
    """

    def __init__(self, image_size=(224, 224), sigma_degrees=1.0):
        self.image_size = image_size
        self.sigma_pixels = sigma_degrees * 30  # 假设30像素≈1°视角

    def generate_heatmap(self, fixations, fixation_mask):
        """
        Args:
            fixations: (T, 3) [FPOGX, FPOGY, FPOGD]
            fixation_mask: (T,)
        Returns:
            heatmap: (H, W)
        """
        H, W = self.image_size
        heatmap = np.zeros((H, W))

        for i, (x, y, duration) in enumerate(fixations):
            if fixation_mask[i] == 0:
                break

            # 归一化坐标 [0,1] → 像素坐标
            x_pixel = int(x * W)
            y_pixel = int(y * H)

            if 0 <= x_pixel < W and 0 <= y_pixel < H:
                # 创建高斯权重,持续时间越长权重越大
                weight = duration / 1000.0  # 转换为秒
                temp_map = np.zeros((H, W))
                temp_map[y_pixel, x_pixel] = weight
                temp_map = gaussian_filter(temp_map, sigma=self.sigma_pixels)
                heatmap += temp_map

        # 归一化到 [0, 1]
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        return heatmap


class PaperStyleGazeEncoder(nn.Module):
    """
    论文风格的眼动编码器
    使用热图序列 + Conv + BiLSTM + Self-Attention
    """

    def __init__(self, args):
        super(PaperStyleGazeEncoder, self).__init__()
        self.d_model = args.d_model
        self.max_fixations = args.max_fixations

        # Heatmap encoder (类似图像编码器)
        self.conv_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        # Temporal encoder
        self.lstm = nn.LSTM(64, 64, batch_first=True, bidirectional=True)

        # Self-attention
        self.attention = nn.MultiheadAttention(128, num_heads=4, batch_first=True)

        # Output projection
        self.fc_out = nn.Linear(128, self.d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, heatmap_sequence):
        """
        Args:
            heatmap_sequence: (B, T, H, W) - 时序热图
        Returns:
            gaze_feats: (B, d_model)
        """
        B, T, H, W = heatmap_sequence.size()

        # Encode each heatmap
        heatmap_feats = []
        for t in range(T):
            h = self.conv_encoder(heatmap_sequence[:, t:t + 1, :, :])  # (B, 64, 1, 1)
            h = h.squeeze(-1).squeeze(-1)  # (B, 64)
            heatmap_feats.append(h)

        heatmap_feats = torch.stack(heatmap_feats, dim=1)  # (B, T, 64)

        # Temporal encoding
        lstm_out, _ = self.lstm(heatmap_feats)  # (B, T, 128)

        # Self-attention pooling
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        gaze_feats = attn_out.mean(dim=1)  # (B, 128)

        # Project to d_model
        gaze_feats = self.fc_out(gaze_feats)
        gaze_feats = self.dropout(gaze_feats)

        return gaze_feats


class DualStreamFusion(nn.Module):
    """
    时序眼动与视觉特征的融合
    使用门控机制保留区域差异性
    """

    def __init__(self, d_visual=2048, d_gaze=512, d_output=2048):
        super(DualStreamFusion, self).__init__()

        # 门控网络：决定每个区域受眼动影响的程度
        self.gate = nn.Sequential(
            nn.Linear(d_visual + d_gaze, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, d_visual),
            nn.Sigmoid()
        )

        # 残差连接的投影层（可选）
        self.residual_proj = nn.Linear(d_visual, d_output)

    def forward(self, visual_feats, gaze_feats):
        """
        Args:
            visual_feats: (B, R, 2048) - 视觉区域特征
            gaze_feats: (B, 512) - 时序眼动编码后的全局特征
        Returns:
            fused_feats: (B, R, 2048) - 融合后的特征
        """
        B, R, D = visual_feats.size()

        # 扩展眼动特征到所有区域
        gaze_expanded = gaze_feats.unsqueeze(1).expand(-1, R, -1)  # (B, R, 512)

        # 拼接视觉和眼动特征
        combined = torch.cat([visual_feats, gaze_expanded], dim=-1)  # (B, R, 2048+512)

        # 计算门控权重
        gate_weights = self.gate(combined)  # (B, R, 2048)

        # 门控调制：保留原始信息 + 眼动调制
        modulated = visual_feats * gate_weights  # (B, R, 2048)

        # 残差连接
        fused_feats = self.residual_proj(visual_feats + modulated)  # (B, R, 2048)

        return fused_feats