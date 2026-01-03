#!/usr/bin/env python
import os
import sys
import multiprocessing
from collections import Counter
import numpy as np
import time

try:
    from PIL import Image
except ImportError:
    print("Pillow (PIL) library not found. Please install it using: pip install Pillow")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    # Dummy tqdm if not available
    def tqdm(iterable, total=None, desc=""):
        return iterable

def get_image_size(file_path):
    try:
        with Image.open(file_path) as img:
            return img.size # (width, height)
    except Exception as e:
        return None

def find_images(root_dir):
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() in image_extensions:
                image_files.append(os.path.join(root, file))
    return image_files

def analyze_resolutions(image_paths, num_workers=None):
    if num_workers is None:
        num_workers = min(multiprocessing.cpu_count(), 16)
    
    print(f"Processing {len(image_paths)} images with {num_workers} workers...")
    
    sizes = []
    # Use chunksize for better performance with large lists
    chunksize = max(1, len(image_paths) // (num_workers * 4))
    
    with multiprocessing.Pool(num_workers) as pool:
        # Use imap_unordered for potentially slightly better performance if order doesn't matter
        # But we need to wrap in list to consume iterator
        results = list(tqdm(pool.imap(get_image_size, image_paths, chunksize=chunksize), 
                           total=len(image_paths), 
                           desc="Reading images"))
    
    sizes = [s for s in results if s is not None]
    return sizes

def print_stats(sizes, dataset_name="All"):
    if not sizes:
        print(f"No images found for {dataset_name}")
        return

    widths = [s[0] for s in sizes]
    heights = [s[1] for s in sizes]
    ratios = [s[0]/s[1] for s in sizes]

    print(f"\n{'='*20} Statistics for {dataset_name} {'='*20}")
    print(f"Total valid images: {len(sizes)}")
    
    print(f"\n[Width]")
    print(f"  Min: {min(widths)}")
    print(f"  Max: {max(widths)}")
    print(f"  Mean: {np.mean(widths):.2f}")
    print(f"  Median: {np.median(widths)}")
    
    print(f"\n[Height]")
    print(f"  Min: {min(heights)}")
    print(f"  Max: {max(heights)}")
    print(f"  Mean: {np.mean(heights):.2f}")
    print(f"  Median: {np.median(heights)}")
    
    print(f"\n[Aspect Ratio (W/H)]")
    print(f"  Mean: {np.mean(ratios):.2f}")
    print(f"  Median: {np.median(ratios):.2f}")

    # Top 10 resolutions
    res_counts = Counter(sizes)
    print("\n[Top 10 Resolutions (WxH)]")
    for res, count in res_counts.most_common(10):
        print(f"  {res[0]}x{res[1]}: {count} ({count/len(sizes)*100:.2f}%)")
    
    print("="*60)

def main():
    base_dir = '/root/multimodal/person_search/training_data'
    datasets = ['CUHK-PEDES', 'ICFG-PEDES', 'RSTPReid']
    
    all_sizes = []
    
    start_time = time.time()
    
    for dataset in datasets:
        img_dir = os.path.join(base_dir, dataset, 'imgs')
        if not os.path.exists(img_dir):
            print(f"Warning: {img_dir} does not exist. Skipping {dataset}.")
            continue
            
        print(f"\nScanning {dataset} in {img_dir}...")
        images = find_images(img_dir)
        print(f"Found {len(images)} images.")
        
        if not images:
            continue

        # Analyze individual dataset
        sizes = analyze_resolutions(images)
        print_stats(sizes, dataset)
        
        all_sizes.extend(sizes)

    if len(datasets) > 1:
        print("\nComputing Overall Statistics...")
        print_stats(all_sizes, "ALL DATASETS")
        
    print(f"\nTotal time elapsed: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
