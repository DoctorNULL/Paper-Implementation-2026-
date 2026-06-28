from ml.encoder import ModelEncoder

encoder = ModelEncoder("CLIP_vit_base_patch32")
video_path = r"C:\Users\dell\Desktop\test desktop\DHARIA_-_Sugar_&_Brownies_(by_Monoir)_[Official_Video].mp4"

text = ["A woman is singing", "Woman moving on the beach", "A plane in the sky", "Music video", "Someone is playing a game", "A man is talking"]

text_features = encoder.extract_text_features(text) # (B, 512)
visual_features = encoder.extract_video_features(video_path) # (4, 512)

mat = visual_features @ text_features.transpose(0, 1)

print(mat)