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

    def generate(self, features: torch.Tensor, max_len = 30):
        eos = 49407

        res = []

        v_center, v_base = self.akse.forward(features, self.top_k_selection)

        logits, w, v, h1, c1, h2, c2 = None, None, None, None, None, None, None

        for _ in range(max_len):
            logits, w, v, h1, c1, h2, c2 = self.decoder.forward(v_center, v_base, w, v, h1, c1, h2, c2)

            word = w.item()

            w = self.encoder.get_text_features({"input_ids": w}, normalize=False).unsqueeze(0)

            res.append(word)

            if word == eos:
                break

        return res