"""MViTv2-S adapter. Random weights are explicit and never a silent fallback."""
from torch import nn
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights


class MViTEncoder(nn.Module):
    def __init__(self, pretrained=True, representation='embedding'):
        super().__init__()
        if representation not in ('embedding', 'kinetics_logits'):
            raise ValueError('representation must be embedding or kinetics_logits')
        weights = MViT_V2_S_Weights.KINETICS400_V1 if pretrained else None
        self.network = mvit_v2_s(weights=weights)
        self.output_dim = 768 if representation == 'embedding' else 400
        if representation == 'embedding':
            self.network.head = nn.Identity()

    def forward(self, x):
        if x.ndim != 5 or tuple(x.shape[1:]) != (3, 16, 224, 224):
            raise ValueError('MViTv2-S expects N,3,16,224,224 input')
        return self.network(x)
