import torch
from ml.model import CaptioningModel
from pickle import load
from random import choice

kb = torch.load("encodes/kb.cache", weights_only=True)
model = CaptioningModel("CLIP_vit_base_patch32", kb, 3).to("cuda")

with open("encodes/text.cache", 'rb') as file:
    all_data = load(file)

with open(r"D:\MachineLearning\Datasets\MSRVTT\high-quality\structured-symlinks\train_list_full.txt", 'r') as file:
    train_data = [x.replace("\n", "") for x in file.readlines()]


cost = torch.nn.CrossEntropyLoss()
tau = 0.04

for item in all_data:
    if item in train_data:

        encodes = all_data[item]["encodes"]
        caps = all_data[item]["caps"]
        inputs = all_data[item]["inputs"]
        outputs = all_data[item]["outputs"]

        v_center, v_ne = model.akse.forward(encodes)

        loss = torch.zeros(1, device="cuda")
        w = None
        new_v = None
        h1 = None
        c1 = None
        h2 = None
        c2 = None

        i = -1

        for word in inputs:
            if not word:
                i += 1
                logits, w, new_v, h1, c1, h2, c2 = model.decoder.forward(v_center[i].unsqueeze(0), v_ne[i].unsqueeze(0))
            else:
                logits, w, new_v, h1, c1, h2, c2 = model.decoder.forward(v_center[i].unsqueeze(0), v_ne[i].unsqueeze(0), w, new_v, h1, c1, h2, c2)

            w = model.encoder.get_text_features({"input_ids": w}, normalize=False).unsqueeze(0)
            loss += cost(logits, outputs[i].unsqueeze(0))

        all_ls = model.akse.top.l # (L, 512)
        all_mls = model.akse.csml.forward(all_ls) # (L, 512)

        pos_ls = model.akse.top.forward(encodes) # (B, 512)
        pos_mls = model.akse.csml.forward(encodes) # (B, 512)

        nom = (pos_ls * pos_mls / tau).exp().sum().sum()
        dom = (all_ls * all_mls / tau).exp().sum().sum()
        loss -= (nom / dom).log()

        print(loss)

        print(caps)

    exit()
