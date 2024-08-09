
import os, sys, torch 
import argparse
from pathlib import Path
import numpy as np
from torchvision import transforms
import argparse
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from PIL import Image
import pandas as pd


# def unormalize(x):
#     # convert x in range [-1, 1], (B,C,H,W), tensor to [0, 255], uint8, numpy, (B,H,W,C)
#     x = torch.clamp((x + 1) * 127.5, 0, 255).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
#     return x

def watermark_np_to_str(watermark_np):
    """
        Convert a watermark in np format into a str to display.
    """
    return "".join([str(i) for i in watermark_np.tolist()])


def main(args):
    # print(welcome_message())
    # Load model
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

    # === Set the dataset paths ===
    args.watermarker = "Rosteals"
    dataset_input_path = os.path.join(
        args.clean_data_root, args.dataset_name
    )
    output_root_path = os.path.join(
        args.clean_data_root, "..", args.watermarker, args.dataset_name
    )
    output_img_root = os.path.join(output_root_path, "encoder_img")
    os.makedirs(output_img_root, exist_ok=True)

    # === Scan all images in the dataset (clean) ===
    img_files = [f for f in os.listdir(dataset_input_path) if ".png" in f]
    print("Total number of images: [{}] --- (Should be 2000)".format(len(img_files)))

    # Init the dict to save watermarking summary
    save_csv_dir = os.path.join(output_root_path, "water_mark.csv")
    res_dict = {
        "ImageName": [],
        "Encoder": [],
        "Decoder": [],
        "Match": []
    }

    # cover
    tform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    # secret
    # ecc = ECC()
    # secret = ecc.encode_text([args.secret])  # 1, 100
    secret = np.random.binomial(1, 0.5, secret_len) 
    secret_str = watermark_np_to_str(secret)
    secret = torch.from_numpy(secret).cuda().float()  # 1, 100
    print("What is the w: [{}]".format(secret_str))
    
    for img_name in img_files:
        img_clean_path = os.path.join(dataset_input_path, img_name)
        print("***** ***** ***** *****")
        print("Processing Image: {} ...".format(img_clean_path))

        cover_org = Image.open(img_clean_path).convert('RGB')
        w,h = cover_org.size
        cover = tform(cover_org).unsqueeze(0).cuda()  # 1, 3, 256, 256
        # inference
        with torch.no_grad():
            z = model.encode_first_stage(cover)
            z_embed, _ = model(z, None, secret)
            stego = model.decode_first_stage(z_embed)  # 1, 3, 256, 256
            res = stego.clamp(-1,1) - cover  # (1,3,256,256) residual
            res = torch.nn.functional.interpolate(res, (h,w), mode='bilinear')
            res = res.permute(0,2,3,1).cpu().numpy()  # (1,h,w,3)
            stego_uint8 = np.clip(res[0] + np.array(cover_org)/127.5-1., -1,1)*127.5+127.5  
            stego_uint8 = stego_uint8.astype(np.uint8)  # (h,w, 3), ndarray, uint8

            # Save Image
            img_w_path = os.path.join(output_img_root, img_name)
            Image.fromarray(stego_uint8).save(img_w_path)
            print("Watermarked img saved to: {}".format(img_w_path))

            # decode secret
            print('Extracting secret...')
            secret_pred = np.where(model.decoder(stego).cpu().numpy() > 0, 1, 0)  # 1, 100
            bit_acc = np.mean(secret_pred == secret.cpu().numpy())
            print(f'Bit acc: {bit_acc}')
            secret_decoded = secret_pred[0]
            secret_decoded_str = watermark_np_to_str(secret_decoded)
            print(f'Recovered secret: [{secret_decoded_str}]')

        res_dict["ImageName"].append(img_name)
        res_dict["Encoder"].append([secret_str])
        res_dict["Decoder"].append([secret_decoded_str])
        res_dict["Match"].append(bit_acc > 0.95)

    df = pd.DataFrame(res_dict)
    df.to_csv(save_csv_dir, index=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', "--config", default='models/VQ4_mir_inference.yaml', help="Path to config file.")
    parser.add_argument('-w', "--weight", default='models/RoSteALS/epoch=000017-step=000449999.ckpt', help="Path to checkpoint file.")
    parser.add_argument(
        "--image_size", type=int, default=256, help="Height and width of square images."
    )
    parser.add_argument(
        "--random_seed", dest="random_seed", type=int, help="Manually set random seed for reproduction.",
        default=13
    )
    parser.add_argument(
        '--clean_data_root', type=str, help="Root dir where the clean image dataset is located.",
        default=os.path.join("dataset", "Clean")
    )
    parser.add_argument(
        "--dataset_name", dest="dataset_name", type=str, help="The dataset name: [COCO, DiffusionDB]",
        default="COCO"
    )
    args = parser.parse_args()
    main(args)
    print("Completed")