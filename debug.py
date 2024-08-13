import os
from PIL import Image
import numpy as np
import pickle

if __name__ == "__main__":
    # === Test PIL image format after reading ===
    # img_dir = os.path.join(
    #     "dataset", "Clean", "COCO", "Img-1.png"
    # )
    # img_pil = Image.open(img_dir).convert('RGB')
    # img_np = np.asarray(img_pil)
    # print()
    

    # === Test if decode result is correct ===
    test_dir = os.path.join(
        "Result-Decoded", "Rosteals", "COCO", 
        "vae", "cheng2020-anchor", "Img-1.pkl"
    )
    with open(test_dir, 'rb') as handle:
        data_dict = pickle.load(handle)
    print()