import torch

from ml.model import CaptioningModel

encoder = CaptioningModel("CLIP_vit_base_patch32", torch.rand(50, 512))

print(encoder)
