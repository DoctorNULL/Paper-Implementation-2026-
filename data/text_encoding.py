from tqdm import tqdm

from ml.encoder import ModelEncoder
from pickle import dump, load
import os

def encode_text(base_path: str, out_path: str, encoder: ModelEncoder):
    with open(base_path, 'rb') as file:
        data = load(file)

    cache = {}

    if os.path.isfile(out_path):
        with open(out_path, 'rb') as file:
            cache = load(file)

    bar = tqdm(total=10000)
    try:
        for key, val in data.items():
            bar.update(1)
            if key in cache.keys():
                continue
                
            val = val[:5]

            caps = [" ".join(x) for x in val]
            encodes = encoder.extract_text_features(caps, normalize=False)

            inputs = []
            outputs = []

            for sent in val:
                first = True
                for word in sent:
                    if first:
                        inputs.append('')
                        first = False
                    else:
                        inputs.append(outputs[-1])
                    outputs.append(word)

                inputs.append(outputs[-1])
                outputs.append("<|endoftext|>")

            outputs = encoder.encode_text(outputs)["input_ids"][:, 1]
            cache[key] = {
                "encodes": encodes,
                "caps": caps,
                "inputs": inputs,
                "encoded_inputs": encoder.encode_text(inputs)["input_ids"][:, 1],
                "outputs": outputs
            }

    except Exception as e:
        print(f"{e}")
        print("Crashed")
        exit()
    bar.close()

    with open(out_path, 'wb') as file:
        dump(cache, file)
