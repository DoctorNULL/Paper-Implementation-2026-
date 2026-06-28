import os.path

from ml.encoder import ModelEncoder
from glob import glob
from tqdm import tqdm
from pathlib import Path
from pickle import dump, load

def encode_videos(base_path: str, out_path: str, encoder: ModelEncoder):

    data = {}

    if os.path.isfile(out_path):
        with open(out_path, 'rb') as file:
            data = load(file)

    bar = tqdm(total=10000)
    try:
        for video in glob(base_path):
            name = Path(video).name

            if name in data.keys():
                continue

            data[name] = encoder.extract_video_features(video, normalize=False)
            bar.update(1)
    except Exception as e:
        print("Crashed")
    bar.close()

    with open(out_path, 'wb') as file:
        dump(data, file)
