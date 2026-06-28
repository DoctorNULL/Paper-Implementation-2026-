import torch
from data.knowledge_base import build_knowledge_base
from data.text_encoding import encode_text
from ml.encoder import ModelEncoder

text = r"D:\MachineLearning\Datasets\MSRVTT\high-quality\structured-symlinks\raw-captions.pkl"

encode_text(text, "encodes/text.cache", ModelEncoder("CLIP_vit_base_patch32").cuda())

visual = "encodes/visual.cache"
text = "encodes/text.cache"

kb = build_knowledge_base(text, visual)

torch.save(kb, "encodes/kb.cache")

print(kb)