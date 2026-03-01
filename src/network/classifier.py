import torch
import torch.nn.functional as F
from torch import nn


class AAMClassifier(nn.Module):
    """
    AAM-Softmax分类器
    训练时需结合CrossEntropyLoss使用
    """

    def __init__(
        self,
        embedding_size: int,
        num_classes: int,
        margin: float = 0.2,
        scale: float = 30.0,
    ):
        super(AAMClassifier, self).__init__()
        self.margin = margin
        self.scale = scale
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        self.num_classes = num_classes
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, labels: torch.Tensor):
        assert (
            x.shape[0] == labels.shape[0]
        ), "Batch size of inputs and labels must be the same"
        assert (
            torch.min(labels) >= 0 and torch.max(labels) < self.num_classes
        ), "Labels must be in the range [0, num_classes-1]"

        cos_beta = F.linear(
            F.normalize(x), F.normalize(self.weight)
        )  # [B, num_classes]
        cos_beta = torch.clamp(cos_beta, -1.0 + 1e-9, 1.0 - 1e-9)  # 数值稳定性
        beta = torch.acos(cos_beta)  # [B, num_classes]
        label_margin = F.one_hot(labels, self.num_classes) * self.margin
        beta = beta + label_margin
        cos_beta = torch.cos(beta)  # [B, num_classes]
        cos_beta = cos_beta * self.scale  # [B, num_classes]
        return cos_beta
