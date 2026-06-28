from pickle import load
from tqdm import tqdm
import torch


def build_knowledge_base(text_cache_path: str, video_cache_path: str, batch_size = 10000, top_k = 1000):

    print("Reading Text...")
    with open(text_cache_path, 'rb') as file:
        text = load(file)

    print("Reading Visual...")
    with open(video_cache_path, 'rb') as file:
        visual = load(file)

    print("Processing Text...")
    all_text = [x["encodes"] for x in text.values()]

    all_text = torch.cat(all_text, 0)   # (199994, 512)

    all_text = all_text / all_text.norm(p=2, dim=-1, keepdim=True)

    print("Processing Visual...")
    all_visual = [x for x in visual.values()]

    all_visual = torch.cat(all_visual, 0)   # (40000, 512)

    all_visual = all_visual / all_visual.norm(p=2, dim=-1, keepdim=True)

    batch_len = all_text.size(0) // batch_size

    best = []

    print("Batching...")

    bar = tqdm(total=batch_len + 1)
    for batch in range(0, (batch_len + 1) * batch_size, batch_size):
        textual = all_text[batch: batch + batch_size, :]

        sim_mat = textual @ all_visual.transpose(0, 1)

        indices = sim_mat.topk(top_k).values.argmax(-1).topk(top_k).indices

        best.append(textual[indices])
        bar.update(1)
        
    bar.close()

    print("Finalizing...")
    best = torch.cat(best, 0)

    sim_mat = best @ all_visual.transpose(0, 1)

    indices = sim_mat.topk(top_k).values.argmax(-1).topk(top_k).indices

    best = best[indices]

    return best