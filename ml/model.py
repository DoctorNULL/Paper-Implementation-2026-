import torch
import torch.nn as nn

from ml.akse import AKSEModule
from ml.decoder import CaptionDecoder
from ml.encoder import ModelEncoder


class CaptioningModel(nn.Module):
    def __init__(self,
                 path: str,
                 knowledge_base: torch.Tensor,
                 num_layers: int = 5,
                 top_k_selection: int = 3
                 ):
        super().__init__()

        self.top_k_selection = top_k_selection

        self.encoder = ModelEncoder(path)
        self.akse = AKSEModule(self.encoder, knowledge_base, num_layers)
        self.decoder = CaptionDecoder(self.encoder.text_tokenizer.vocab_size)

    def process_text(self, text: list[str]) -> torch.Tensor:
        return self.encoder.extract_text_features(text)

    def process_video(self, path: str, frames = 4) -> torch.Tensor:
        return self.encoder.extract_video_features(path, frames)

    def forward(self, features: torch.Tensor):

        v_center, v0 = self.akse.forward(features, self.top_k_selection)

        return self.decoder.forward(v_center, v0)

    def generate(self, path: str):
        visual = self.process_video(path)
