import os
import sys
from pathlib import Path
import re
from tqdm import tqdm
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from piqa.lpips import LPIPS
from piqa.ssim import SSIM
from pytorch_fid import fid_score
from datetime import datetime
import time
import tempfile


def find_image_pairs(source_dir):
    source_path = Path(source_dir)
    
    if not source_path.exists():
        print(f"Error: Source directory does not exist: {source_dir}")
        return []
    
    apy_folder = source_path / "Apy"
    if not apy_folder.exists() or not apy_folder.is_dir():
        print(f"Error: Cannot find Apy subfolder: {apy_folder}")
        return []
    
    generated_files = {}
    for file_path in source_path.glob("*_0.png"):
        match = re.match(r"(\d+)_0\.png", file_path.name)
        if match:
            identifier = match.group(1)
            generated_files[identifier] = file_path
    
    paired_files = []
    original_files = {}
    for orig_path in apy_folder.glob("orig_*.png"):
        match = re.match(r"orig_(\d+)\.png", orig_path.name)
        if match:
            identifier = match.group(1)
            original_files[identifier] = orig_path
            
            if identifier in generated_files:
                paired_files.append((
                    generated_files[identifier],
                    orig_path,
                    identifier
                ))
    
    print(f"Found {len(paired_files)} pairs")
    
    return paired_files


def calculate_psnr(img1, img2):
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same dimensions")
    
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    
    max_pixel = 1.0 if img1.max() <= 1 else 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    
    return psnr


def calculate_ssim_piqa(img1, img2):
    if len(img1.shape) == 2:
        img1 = img1[:, :, np.newaxis]
        img2 = img2[:, :, np.newaxis]
    
    img1_tensor = torch.from_numpy(img1).float().permute(2, 0, 1).unsqueeze(0)
    img2_tensor = torch.from_numpy(img2).float().permute(2, 0, 1).unsqueeze(0)
    
    ssim_metric = SSIM()
    ssim_val = ssim_metric(img1_tensor, img2_tensor)
    
    return ssim_val.item()


def calculate_lpips(img1, img2):
    if len(img1.shape) == 2:
        img1 = img1[:, :, np.newaxis]
        img2 = img2[:, :, np.newaxis]
    
    img1_tensor = torch.from_numpy(img1).float().permute(2, 0, 1).unsqueeze(0)
    img2_tensor = torch.from_numpy(img2).float().permute(2, 0, 1).unsqueeze(0)
    
    lpips_metric = LPIPS(network='alex')
    lpips_val = lpips_metric(img1_tensor, img2_tensor)
    
    return lpips_val.item()


def rmetrics(generated, original):
    if generated.max() > 1.0:
        generated = generated / 255.0
    if original.max() > 1.0:
        original = original / 255.0
    
    psnr = calculate_psnr(generated, original)
    ssim = calculate_ssim_piqa(generated, original)
    lpips = calculate_lpips(generated, original)
    
    return psnr, ssim, lpips


def calculate_fid_from_arrays(real_images, fake_images, batch_size=8, device='cuda', dims=2048):
    if len(real_images) == 0 or len(fake_images) == 0:
        return None
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            real_temp_dir = os.path.join(temp_dir, 'real')
            fake_temp_dir = os.path.join(temp_dir, 'fake')
            os.makedirs(real_temp_dir)
            os.makedirs(fake_temp_dir)
            
            for i, img in enumerate(real_images):
                plt.imsave(os.path.join(real_temp_dir, f'{i:04d}.png'), img)
            for i, img in enumerate(fake_images):
                plt.imsave(os.path.join(fake_temp_dir, f'{i:04d}.png'), img)
            
            fid_value = fid_score.calculate_fid_given_paths(
                [real_temp_dir, fake_temp_dir],
                batch_size=batch_size,
                device=device,
                dims=dims,
                num_workers=4
            )
            return fid_value
    except Exception as e:
        print(f"Error calculating FID: {str(e)}")
        return None


def calculate_metrics_direct(paired_files, base_name="metrics"):
    if not paired_files:
        print("Error: No paired files for evaluation!")
        return None
    
    sumpsnr, sumssim, sumlpips = 0., 0., 0.
    N = 0
    failed_count = 0
    
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    results_file = f"{base_name}_{timestamp}.txt"
    f_results = open(results_file, 'w')
    
    real_images = []
    fake_images = []
    
    print(f"\nProcessing {len(paired_files)} image pairs...")
    
    for gen_file, orig_file, identifier in tqdm(paired_files, desc="Processing"):
        try:
            generated = plt.imread(str(gen_file))
            original = plt.imread(str(orig_file))
            
            if generated is None or original is None:
                failed_count += 1
                continue
            
            if len(generated.shape) == 0 or len(original.shape) == 0:
                failed_count += 1
                continue
            
            real_images.append(original)
            fake_images.append(generated)
            
            psnr, ssim, lpips = rmetrics(generated, original)
            
            f_results.write(f"Pair: {identifier}\n")
            f_results.write(f"Generated: {gen_file.name}\n")
            f_results.write(f"Original: {orig_file.name}\n")
            f_results.write(f"PSNR: {psnr:.4f}, SSIM: {ssim:.4f}, LPIPS: {lpips:.4f}\n\n")
            f_results.flush()
            
            sumpsnr += psnr
            sumssim += ssim
            sumlpips += lpips
            N += 1
            
        except Exception as e:
            print(f"Error processing pair {identifier}: {e}")
            failed_count += 1
            continue
    
    print(f"\nSuccessfully processed: {N} pairs, Failed: {failed_count} pairs")
    print("\nCalculating FID score...")
    
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        fid = calculate_fid_from_arrays(real_images, fake_images, batch_size=8, device=device, dims=2048)
        if fid is not None:
            print(f"FID score: {fid:.2f}")
    except Exception as e:
        print(f"Cannot calculate FID: {str(e)}")
        fid = None
    
    if N > 0:
        mpsnr = sumpsnr / N
        mssim = sumssim / N
        mlpips = sumlpips / N
        
        print("\nResults summary:")
        print(f"Average PSNR: {mpsnr:.4f}")
        print(f"Average SSIM: {mssim:.4f}")
        print(f"Average LPIPS: {mlpips:.4f}")
        if fid is not None:
            print(f"FID score: {fid:.4f}")
        
        f_results.write("\n" + "=" * 50 + "\n")
        f_results.write("Average metrics:\n")
        f_results.write(f"A_PSNR: {mpsnr:.4f}, A_SSIM: {mssim:.4f}, A_LPIPS: {mlpips:.4f}, ")
        f_results.write(f"A_FID: {(f'{fid:.4f}' if fid is not None else 'N/A')}\n")
        f_results.write(f"Successfully processed: {N}/{len(paired_files)} pairs\n")
    else:
        print("\nWarning: No image pairs successfully processed!")
    
    f_results.close()
    print(f"\nResults saved to: {results_file}")
    
    return results_file


def main():
    print("\nImage Quality Evaluation Tool")
    
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("python cleaned_metrics.py <source_folder_path>")
        print("\nExample:")
        print("python cleaned_metrics.py /path/to/DDNM_celeba_4SR")
        sys.exit(1)
    
    source_dir = sys.argv[1]
    start_time = time.time()
    
    paired_files = find_image_pairs(source_dir)
    
    if not paired_files:
        print("\nError: No paired files found!")
        sys.exit(1)
    
    source_path = Path(source_dir)
    folder_name = source_path.name
    safe_folder_name = re.sub(r'[<>:"/\\|?*]', '_', folder_name)
    
    time.sleep(2)
    
    try:
        metrics_file = calculate_metrics_direct(paired_files, safe_folder_name)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print("\nAll tasks completed!")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Paired files: {len(paired_files)}")
        print(f"Results file: {metrics_file}")
        
    except Exception as e:
        print(f"\nEvaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()