import torch
import torch.nn as nn

from ml.encoder import ModelEncoder


class CSMLModule(nn.Module):
    def __init__(self, encoder: ModelEncoder):
        super().__init__()

        self.encoder = encoder
        self.fc = nn.Linear(512, 3*224*224)

    def forward(self, text_features: torch.Tensor) -> torch.Tensor:
        # (B, 512)

        y = self.fc(text_features)
        y = torch.relu(y)
        # (B, 224 * 224)
        y = y.reshape(text_features.size(0), 3, 224, 224)
        # (B, 3, 224, 224)

        y = self.encoder.model.get_image_features(y)
        # (B, 512)

        return y
