import torch
from transformers import CLIPTokenizer, CLIPModel, CLIPProcessor
from torchcodec.decoders import VideoDecoder

class ModelEncoder(torch.nn.Module):
    def __init__(self, path: str):
        super().__init__()

        self.text_tokenizer = CLIPTokenizer.from_pretrained(path)
        self.visual_encoder = CLIPProcessor.from_pretrained(path, use_fast=True)
        self.model = CLIPModel.from_pretrained(path)

    def encode_text(self, caps: list[str]) -> dict[str, torch.Tensor]:
        encodes = self.text_tokenizer(caps, padding=True, return_tensors="pt")

        return encodes # (B, SeqLength)

    def get_text_features(self, encoded_text: dict[str, torch.Tensor], normalize = True) -> torch.Tensor:
        # (B, SeqLength)
        with torch.inference_mode():
            features = self.model.get_text_features(**encoded_text)
            # (B, 512)

        if normalize:
            features = features / features.norm(p=2, dim=-1, keepdim=True)

        return features

    def extract_text_features(self, text: list[str], normalize=True) -> torch.Tensor:

        encodes = self.encode_text(text)

        return self.get_text_features(encodes, normalize)

    def encode_video(self, path: str, sampled_frames = 4) -> dict[str, torch.Tensor]:
        frames = VideoDecoder(path)[:]

        frames_index = torch.linspace(0, frames.size(0) - 1, sampled_frames).long()

        sampled = frames[frames_index]

        processed = self.visual_encoder(images=sampled, return_tensors="pt")

        return processed

    def get_visual_features(self, processed:dict[str, torch.Tensor], normalize=True) -> torch.Tensor:

        with torch.inference_mode():
            features = self.model.get_image_features(**processed)

        if normalize:
            features = features / features.norm(p=2, dim=-1, keepdim=True)

        return features

    def extract_video_features(self, path: str, normalize = True, sample_size = 4) -> torch.Tensor:

        encodes = self.encode_video(path, sample_size)

        return self.get_visual_features(encodes, normalize)