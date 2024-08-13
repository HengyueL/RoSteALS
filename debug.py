import os
from PIL import Image
import numpy as np


if __name__ == "__main__":
    img_dir = os.path.join(
        "dataset", "Clean", "COCO", "Img-1.png"
    )
    img_pil = Image.open(img_dir).convert('RGB')
    img_np = np.asarray(img_pil)
    print()