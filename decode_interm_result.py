"""
    This script is a skeleton file for **Taihui** to:

    1) Read in the watermark evasion interm. results

    2) Decode each of the interm. result using the encoder/decoder API

    3) Save the result with standardized format
"""
import sys, os
dir_path = os.path.abspath(".")
sys.path.append(dir_path)
dir_path = os.path.abspath("..")
sys.path.append(dir_path)

import argparse, pickle, os, cv2, torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from torchvision import transforms
from PIL import Image

# =====
from general import watermark_str_to_numpy, watermark_np_to_str, uint8_to_float, compute_ssim


def calc_mse(img_1_bgr_uint8, img_2_bgr_uint8):
    img_1_float = uint8_to_float(img_1_bgr_uint8)
    img_2_float = uint8_to_float(img_2_bgr_uint8)
    mse = np.mean((img_1_float - img_2_float)**2)
    return mse


def main(args):
    # === This is where the interm. results are saved ===
    # data_root_dir = os.path.join(
    #     "..", "DIP_Watermark_Evasion", "Result-Interm", 
    #     args.watermarker, args.dataset, args.evade_method, args.arch
    # )
    data_root_dir = os.path.join(
        "Result-Interm", 
        args.watermarker, args.dataset, args.evade_method, args.arch
    )
    file_names = [f for f in os.listdir(data_root_dir) if ".pkl" in f]  # Data are saved as dictionary in pkl format.

    # === This is where the watermarked image is stored ===
    im_w_root_dir = os.path.join("dataset", args.watermarker, args.dataset, "encoder_img")
    # === This is where the original clean image is stored ===
    im_orig_root_dir = os.path.join("dataset", "Clean", args.dataset)

    # === Save the result in a different location in case something went wrong ===
    save_root_dir = os.path.join("Result-Decoded", args.watermarker, args.dataset, args.evade_method, args.arch)
    os.makedirs(save_root_dir, exist_ok=True)
    
    # === Process each file ===
    for file_name in file_names:
        # Retrieve the im_w name
        im_w_file_name = file_name.replace(".pkl", ".png")
        if "_hidden" in im_w_file_name:
            im_orig_name = im_w_file_name.replace("_hidden", "")
        else:
            im_orig_name = im_w_file_name

        # Readin the intermediate files
        data_file_path = os.path.join(data_root_dir, file_name)
        with open(data_file_path, 'rb') as handle:
            data_dict = pickle.load(handle)
        # Readin the im_w into bgr uint8 format
        im_w_path = os.path.join(im_w_root_dir, im_w_file_name)
        im_w_bgr_uint8 = cv2.imread(im_w_path)
        # Readin the 
        im_orig_path = os.path.join(im_orig_root_dir, im_orig_name)
        im_orig_bgr_uint8 = cv2.imread(im_orig_path)
        
        # Get the reconstructed data from the interm. result
        if args.evade_method == "WevadeBQ":
            img_recon_list = data_dict["best_recon"]
        else:
            img_recon_list = data_dict["interm_recon"]  # A list of recon. image in "bgr uint8 np" format (cv2 standard format)
        n_recon = len(img_recon_list)
        print("Total number of interm. recon. to process: [{}]".format(n_recon))

        # === Initiate a encoder & decoder ===
        watermark_gt_str = data_dict["watermark_gt_str"]
        if watermark_gt_str[0] == "[":  # Some historical none distructive bug :( will cause this reformatting
            watermark_gt_str = eval(data_dict["watermark_gt_str"])[0]
        watermark_gt = watermark_str_to_numpy(watermark_gt_str)

        # === Init Watermarker (Rosteals) ===
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
        # Standard Image Transform for Rosteals
        tform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])


        # Process each inter. recon
        watermark_decoded_log = []  # A list to save decoded watermark
        index_log = data_dict["index"]
        psnr_orig_log = []
        mse_orig_log = []
        psnr_w_log = []
        mse_w_log = []
        ssim_orig_log = []
        ssim_w_log = []
        for img_idx in range(n_recon):
            img_bgr_uint8 = img_recon_list[img_idx]    # shape [512, 512, 3]
            if args.watermarker == "StegaStamp" and args.arch in ["cheng2020-anchor", "mbt2018"]:
                img_bgr_uint8 = cv2.resize(img_bgr_uint8, (400, 400), interpolation=cv2.INTER_LINEAR)

            # =================== YOUR CODE HERE =========================== #
            
            # Step 0: if you need to change the input format
            img_rgb_uint8 = cv2.cvtColor(img_bgr_uint8, cv2.COLOR_BGR2RGB)
            img_input = Image.fromarray(img_rgb_uint8)
            img_input = tform(img_input).unsqueeze(0).cuda()

            # Step 1: Decode the interm. result
            print('Extracting secret...')
            with torch.no_grad():
                secret_pred = np.where(model.decoder(img_input).cpu().numpy() > 0, 1, 0)  # 1, 100
            watermark_decoded = secret_pred[0]
            watermark_decoded_str = watermark_np_to_str(watermark_decoded)

            # Step 2: log the result
            watermark_decoded_log.append(watermark_decoded_str)

            # ============================================================= #

            # Calculate the quality: mse and psnr
            mse_recon_orig = calc_mse(im_orig_bgr_uint8, img_bgr_uint8)
            mse_recon_w = calc_mse(im_w_bgr_uint8, img_bgr_uint8)

            psnr_recon_orig = compute_psnr(
                im_orig_bgr_uint8.astype(np.int16), img_bgr_uint8.astype(np.int16), data_range=255
            )
            psnr_recon_w = compute_psnr(
                im_w_bgr_uint8.astype(np.int16), img_bgr_uint8.astype(np.int16), data_range=255
            )
            ssim_recon_orig = compute_ssim(
                im_orig_bgr_uint8.astype(np.int16), img_bgr_uint8.astype(np.int16), data_range=255
            )
            ssim_recon_w = compute_ssim(
                im_w_bgr_uint8.astype(np.int16), img_bgr_uint8.astype(np.int16), data_range=255
            )

            
            mse_orig_log.append(mse_recon_orig)
            mse_w_log.append(mse_recon_w)
            psnr_orig_log.append(psnr_recon_orig)
            psnr_w_log.append(psnr_recon_w)
            ssim_orig_log.append(ssim_recon_orig)
            ssim_w_log.append(ssim_recon_w)

        # Save the result
        processed_dict = {
            "index": index_log,
            "watermark_gt_str": watermark_gt_str, # Some historical none distructive bug :( will cause this reformatting
            "watermark_decoded": watermark_decoded_log,
            # "mse_orig": mse_orig_log,
            "psnr_orig": psnr_orig_log,
            "ssim_orig": ssim_orig_log,
            # "mse_w": mse_w_log,
            "psnr_w": psnr_w_log,
            "ssim_w": ssim_w_log
        }

        save_name = os.path.join(save_root_dir, file_name)
        with open(save_name, 'wb') as handle:
            pickle.dump(processed_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Decoded Interm. result saved to {}".format(save_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Some arguments to play with.')
    # === Do not change ===
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
    # =====================

    parser.add_argument(
        "--watermarker", dest="watermarker", type=str, 
        help="Specification of watermarking method. [rivaGan, dwtDctSvd]",
        default="Rosteals"
    )
    parser.add_argument(
        "--dataset", dest="dataset", type=str, 
        help="Dataset [COCO, DiffusionDB]",
        default="COCO"
    )
    parser.add_argument(
        "--evade_method", dest="evade_method", type=str, help="Specification of evasion method.",
        default="vae"
    )
    parser.add_argument(
        "--arch", dest="arch", type=str, 
        help="""
            Secondary specification of evasion method (if there are other choices).

            Valid values a listed below:
                dip --- ["vanila", "random_projector"],
                vae --- ["cheng2020-anchor", "mbt2018", "bmshj2018-factorized"],
                corrupters --- ["gaussian_blur", "gaussian_noise", "bm3d", "jpeg", "brightness", "contrast"]
                diffuser --- Do not need.
        """,
        default="cheng2020-anchor"
    )
    args = parser.parse_args()
    main(args)
    
    # root_lv1 = os.path.join("Result-Interm", args.watermarker, args.dataset)
    # corrupter_names = [f for f in os.listdir(root_lv1)]
    # for corrupter in corrupter_names:
    #     root_lv2 = os.path.join(root_lv1, corru
    # pter)
    #     arch_names = [f for f in os.listdir(root_lv2)]
    #     for arch in arch_names:
    #         args.evade_method = corrupter
    #         args.arch = arch
    #         print("Processing: {} - {} - {} - {}".format(args.watermarker, args.dataset, args.evade_method, args.arch))
    #         main(args)
    print("\n***** Completed. *****\n")