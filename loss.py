import torch
import torch.nn as nn
import torch.nn.functional as F


class LanguageModelCriterion(nn.Module):
    """基础语言模型损失 - 支持Label Smoothing"""

    def __init__(self, label_smoothing=0.0):
        super(LanguageModelCriterion, self).__init__()
        self.label_smoothing = label_smoothing

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]

        if self.label_smoothing > 0:
            # Label Smoothing实现
            # input: [B, L, V] log probabilities
            # target: [B, L] indices

            B, L, V = input.shape

            # 创建one-hot目标
            target_one_hot = torch.zeros_like(input).scatter_(2, target.long().unsqueeze(2), 1.0)

            # 应用label smoothing
            smoothed_target = target_one_hot * (1 - self.label_smoothing) + self.label_smoothing / V

            # 计算KL散度
            output = -torch.sum(input * smoothed_target, dim=-1) * mask
            output = torch.sum(output) / torch.sum(mask)
        else:
            # 原始交叉熵
            output = -input.gather(2, target.long().unsqueeze(2)).squeeze(2) * mask
            output = torch.sum(output) / torch.sum(mask)

        return output


class GazeAwareLoss(nn.Module):
    """眼动感知损失函数 - 支持Label Smoothing"""

    def __init__(self, lambda_gaze=0.1, lambda_consistency=0.05, label_smoothing=0.0):
        super().__init__()
        self.lm_criterion = LanguageModelCriterion(label_smoothing=label_smoothing)
        self.lambda_gaze = lambda_gaze
        self.lambda_consistency = lambda_consistency

    def forward(self, output, reports_ids, reports_masks,
                gaze_attention=None, visual_features=None):
        """
        Args:
            output: 模型输出 [B, L, V]
            reports_ids: 目标ID [B, L]
            reports_masks: 掩码 [B, L]
            gaze_attention: 眼动注意力权重 [B, 49] (可选)
            visual_features: 视觉特征 [B, 49, D] (可选)
        """
        # 1. 主要的语言模型损失
        lm_loss = self.lm_criterion(output, reports_ids[:, 1:], reports_masks[:, 1:])

        total_loss = lm_loss
        loss_dict = {'lm_loss': lm_loss.item()}

        # 2. 眼动注意力正则化损失
        if gaze_attention is not None:
            # 防止注意力分布过于平坦或过于尖锐
            # 使用熵正则化
            entropy = -torch.sum(gaze_attention * torch.log(gaze_attention + 1e-8), dim=-1)
            entropy_loss = torch.abs(entropy - 2.0).mean()  # 目标熵值约为2.0

            # 防止注意力权重全部塌陷到背景值
            attention_std = torch.std(gaze_attention, dim=-1).mean()
            diversity_loss = -torch.log(attention_std + 1e-8)

            gaze_reg_loss = entropy_loss + 0.5 * diversity_loss
            total_loss = total_loss + self.lambda_gaze * gaze_reg_loss
            loss_dict['gaze_reg_loss'] = gaze_reg_loss.item()

        # 3. 视觉-文本一致性损失(可选)
        if visual_features is not None and gaze_attention is not None:
            # 确保被眼动关注的区域产生的特征与生成的文本相关
            # 使用加权池化获取关注区域的特征
            weighted_visual = torch.sum(
                visual_features * gaze_attention.unsqueeze(-1),
                dim=1
            )  # [B, D]

            # 获取文本的表示(使用输出的平均)
            text_repr = torch.mean(output, dim=1)  # [B, V]

            # 投影到相同维度并计算余弦相似度
            if weighted_visual.shape[-1] != text_repr.shape[-1]:
                # 简单的维度匹配
                min_dim = min(weighted_visual.shape[-1], text_repr.shape[-1])
                weighted_visual = weighted_visual[:, :min_dim]
                text_repr = text_repr[:, :min_dim]

            # 余弦相似度损失
            cosine_sim = F.cosine_similarity(weighted_visual, text_repr, dim=-1)
            consistency_loss = 1 - cosine_sim.mean()

            total_loss = total_loss + self.lambda_consistency * consistency_loss
            loss_dict['consistency_loss'] = consistency_loss.item()

        return total_loss, loss_dict


class MultiTaskLoss(nn.Module):
    """多任务损失(用于同时训练报告生成和眼动预测)"""

    def __init__(self, lambda_gaze_pred=0.2, label_smoothing=0.0):
        super().__init__()
        self.gaze_aware_loss = GazeAwareLoss(label_smoothing=label_smoothing)
        self.lambda_gaze_pred = lambda_gaze_pred
        self.mse_loss = nn.MSELoss()

    def forward(self, output, reports_ids, reports_masks,
                gaze_attention=None, predicted_gaze=None, true_gaze=None):
        """
        Args:
            predicted_gaze: 模型预测的眼动模式 [B, 49]
            true_gaze: 真实的眼动数据 [B, 49]
        """
        # 基础损失
        total_loss, loss_dict = self.gaze_aware_loss(
            output, reports_ids, reports_masks, gaze_attention
        )

        # 眼动预测损失(辅助任务)
        if predicted_gaze is not None and true_gaze is not None:
            gaze_pred_loss = self.mse_loss(predicted_gaze, true_gaze)
            total_loss = total_loss + self.lambda_gaze_pred * gaze_pred_loss
            loss_dict['gaze_pred_loss'] = gaze_pred_loss.item()

        return total_loss, loss_dict


def compute_loss(output, reports_ids, reports_masks, **kwargs):
    """
    统一的损失计算接口

    Args:
        output: 模型输出
        reports_ids: 目标ID
        reports_masks: 掩码
        **kwargs: 其他可选参数(如gaze_attention等)
    """
    # 获取label_smoothing参数
    label_smoothing = kwargs.get('label_smoothing', 0.0)

    # 检查是否有眼动相关数据
    use_gaze = 'gaze_attention' in kwargs and kwargs['gaze_attention'] is not None

    if use_gaze:
        # 使用眼动感知损失
        criterion = GazeAwareLoss(label_smoothing=label_smoothing)
        loss, loss_dict = criterion(
            output, reports_ids, reports_masks,
            gaze_attention=kwargs.get('gaze_attention'),
            visual_features=kwargs.get('visual_features')
        )

        # 打印详细损失(用于调试)
        if kwargs.get('verbose', False):
            print(f"Loss breakdown: {loss_dict}")

        return loss
    else:
        # 使用基础损失
        criterion = LanguageModelCriterion(label_smoothing=label_smoothing)
        loss = criterion(output, reports_ids[:, 1:], reports_masks[:, 1:])
        return loss


class FocalLanguageModelLoss(nn.Module):
    """
    Focal Loss变体,用于处理不平衡的词汇分布
    对于眼动数据,可以根据注视区域调整损失权重
    """

    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, input, target, mask, gaze_weights=None):
        """
        Args:
            input: [B, L, V] 模型输出
            target: [B, L] 目标
            mask: [B, L] 掩码
            gaze_weights: [B, L] 基于眼动的权重(可选)
        """
        # 获取预测概率
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]

        # 计算交叉熵
        ce_loss = -input.gather(2, target.long().unsqueeze(2)).squeeze(2)

        # 计算focal权重
        p = torch.exp(-ce_loss)
        focal_weight = (1 - p) ** self.gamma

        # 应用focal权重
        focal_loss = self.alpha * focal_weight * ce_loss

        # 如果有眼动权重,进一步调整
        if gaze_weights is not None:
            gaze_weights = gaze_weights[:, :focal_loss.size(1)]
            focal_loss = focal_loss * gaze_weights

        # 应用掩码并求平均
        focal_loss = focal_loss * mask
        return torch.sum(focal_loss) / torch.sum(mask)


class ContrastiveLoss(nn.Module):
    """
    对比学习损失,用于确保眼动关注区域与生成文本的一致性
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, visual_features, text_features, gaze_mask=None):
        """
        Args:
            visual_features: [B, D] 视觉特征
            text_features: [B, D] 文本特征
            gaze_mask: [B, B] 指示哪些样本有相似的眼动模式
        """
        # 归一化特征
        visual_features = F.normalize(visual_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        # 计算相似度矩阵
        similarity = torch.matmul(visual_features, text_features.T) / self.temperature

        # 对角线是正样本
        labels = torch.arange(similarity.size(0), device=similarity.device)

        # 如果有眼动掩码,调整负样本权重
        if gaze_mask is not None:
            # 降低眼动模式相似样本作为负样本的权重
            similarity = similarity - 1000 * gaze_mask

        # 计算交叉熵损失
        loss = F.cross_entropy(similarity, labels)

        return loss