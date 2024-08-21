"""
    This script is use dwtDctSvd/rivaGan to decode clean images to compute FPR.
"""
import sys, os
dir_path = os.path.abspath(".")
sys.path.append(dir_path)
dir_path = os.path.abspath("..")
sys.path.append(dir_path)
import os, argparse, torch
import cv2
import numpy as np
import pandas as pd
from general import rgb2bgr, save_image_bgr, set_random_seeds, \
    watermark_np_to_str, watermark_str_to_numpy
from torchvision import transforms
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from PIL import Image


def main(args):
    # === Some dummt configs ===
    device = torch.device("cuda")
    set_random_seeds(args.random_seed)

    # === Set the dataset paths ===
    dataset_input_path = os.path.join(
        args.clean_data_root, args.dataset
    )
    # === Scan all images in the dataset (clean) ===
    img_files = [f for f in os.listdir(dataset_input_path) if ".png" in f]
    print("Total number of images: [{}] --- (Should be 100)".format(len(img_files)))

    output_root_path = os.path.join(
        ".", "dataset", "Clean_Watermark_Evasion", args.watermarker, args.dataset
    )
    os.makedirs(output_root_path, exist_ok=True)

    # ==== Init watermarker ===
    config = OmegaConf.load(args.config).model
    secret_len = config.params.control_config.params.secret_len
    config.params.decoder_config.params.secret_len = secret_len
    model = instantiate_from_config(config)
    state_dict = torch.load(args.weight, map_location=torch.device('cpu'))
    if 'global_step' in state_dict:
        print(f'Global step: {state_dict["global_step"]}, epoch: {state_dict["epoch"]}')

    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    misses, ignores = model.load_state_dict(state_dict, strict=False)
    print(f'Missed keys: {misses}\nIgnore keys: {ignores}')
    model = model.cuda()
    model.eval()
    tform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


    # Init the dict to save watermarking summary
    save_csv_dir = os.path.join(output_root_path, "water_mark.csv")
    res_dict = {
        "ImageName": [],
        "Decoder": [],
    }

    for img_name in img_files:
        img_clean_path = os.path.join(dataset_input_path, img_name)
        print("***** ***** ***** *****")
        print("Processing Image: {} ...".format(img_clean_path))

        cover_org = Image.open(img_clean_path).convert('RGB')
        w,h = cover_org.size
        cover = tform(cover_org).unsqueeze(0).cuda()  # 1, 3, 256, 256
        secret_decoded = np.where(model.decoder(cover).detach().cpu().numpy() > 0, 1, 0)[0]  # 1, 100
        watermark_decode_str = watermark_np_to_str(secret_decoded)

        res_dict["ImageName"].append(img_name)
        res_dict["Decoder"].append([watermark_decode_str])

    df = pd.DataFrame(res_dict)
    df.to_csv(save_csv_dir, index=False)


if __name__ == "__main__":
    print("Use this script to download DiffusionDB dataset.")
    parser = argparse.ArgumentParser(description='Some arguments to play with.')
    parser.add_argument(
        "--random_seed", dest="random_seed", type=int, help="Manually set random seed for reproduction.",
        default=13
    )

    # ===============
    parser.add_argument('-c', "--config", default='models/VQ4_mir_inference.yaml', help="Path to config file.")
    parser.add_argument('-w', "--weight", default='models/RoSteALS/epoch=000017-step=000449999.ckpt', help="Path to checkpoint file.")
    parser.add_argument(
        "--image_size", type=int, default=256, help="Height and width of square images."
    )
    # ====================

    parser.add_argument(
        '--clean_data_root', type=str, help="Root dir where the clean image dataset is located.",
        default=os.path.join(".", "dataset", "Clean")
    )
    parser.add_argument(
        "--dataset", dest="dataset", type=str, help="The dataset name: [COCO, DiffusionDB]",
        default="COCO"
    )
    parser.add_argument(
        "--watermarker", dest="watermarker", type=str, help="Specification of watermarking method. ['dwtDctSvd', 'rivaGan']",
        default="Rosteals"
    )
    args = parser.parse_args()
    main(args)
    print("Completd")