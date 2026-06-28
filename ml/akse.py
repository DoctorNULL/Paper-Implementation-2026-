import torch
import torch.nn as nn

from ml.csml import CSMLModule
from ml.encoder import ModelEncoder


class TOPKModule(nn.Module):
    def __init__(self, knowledge_base: torch.Tensor):
        super().__init__()

        self.l = knowledge_base     # (L, 512)

    def forward(self, text_features: torch.Tensor, k = 3) -> torch.Tensor:
        # (B, 512)

        t = text_features.norm(p=2, dim=-1, keepdim=True)   # (B, 512)
        normed_l = self.l.norm(p=2, dim=-1, keepdim=True)   # (basis, 512)

        scores = t @ normed_l.transpose(0, 1)    # (B, basis)

        y = []

        for batch in scores:
            index = batch.topk(k, -1).indices   # (k, 1)

            y.append(self.l[index])     # (k, 512)

        y = torch.stack(y, 0)   # (B, k, 512)

        return y


class AKSELayer(nn.Module):
    def __init__(self):
        super().__init__()

        self.sigmoid_fc = nn.Linear(512, 512)
        self.softmax_fc = nn.Linear(512, 512)

        self.fc = nn.Linear(512, 512)

    def forward(self, v_center: torch.Tensor, vl: torch.Tensor) -> torch.Tensor:
        # v_center (B, 1, 512), vl (B, k + 1, 512)

        r = v_center * vl    # (B, K + 1, 512)
        r_cl = self.sigmoid_fc(r).sigmoid()      # (B, K + 1, 512)

        r_sl = self.softmax_fc(r).softmax(dim=-1).mean(dim=1, keepdim=True)    # (B, 1, 512)

        v_hat = r_cl * r_sl * vl    # (B, K + 1, 512)

        new_vl = self.fc(v_hat) + vl    # (B, K + 1, 512)

        return new_vl


class AKSEModule(nn.Module):
    def __init__(self, encoder: ModelEncoder, knowledge_base: torch.Tensor, num_layers = 5):
        super().__init__()

        self.top = TOPKModule(knowledge_base)
        self.csml = CSMLModule(encoder)
        self.layers = nn.ModuleList([
            AKSELayer()

            for _ in range(num_layers)
        ])


    def forward(self, text_features: torch.Tensor, k = 3) -> tuple[torch.Tensor, torch.Tensor]:
        # (B, 512)

        vs = self.csml.forward(text_features).unsqueeze(1)   # (B, 1, 512)
        va = self.top.forward(text_features, k) # (B, k, 512)

        v = torch.cat([vs, va], 1)  # (B, k + 1, 512)

        v_center = v.mean(dim=1, keepdim=True)  # (B, 1, 512)

        for layer in self.layers:
            v = layer.forward(v_center, v) # (B, k + 1, 512)

        return v_center, v