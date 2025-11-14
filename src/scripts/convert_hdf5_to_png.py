
import os
import argparse
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm


def normalize_image(img):
    """Normalize image to 0-255 range"""
    img = img.astype(np.float32)
    img_min = img.min()
    img_max = img.max()
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    img = (img * 255).astype(np.uint8)
    return img


def convert_hdf5_to_png(input_dir, output_dir, max_images=None, start_index=None):
    """
    Convert HDF5 files to PNG images.
    Supports multiple HDF5 structures.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine starting index based on existing PNGs (append mode)
    if start_index is None:
        existing = [f for f in os.listdir(output_dir) if f.lower().endswith('.png') and f.startswith('image_')]
        if existing:
            try:
                nums = []
                for f in existing:
                    stem = os.path.splitext(f)[0]
                    num = int(stem.split('_')[-1])
                    nums.append(num)
                start_index = max(nums) + 1
            except Exception:
                start_index = len(existing)
        else:
            start_index = 0
    print(f"Starting index: {start_index}")
    
    # Find all HDF5 files
    hdf5_files = []
    for ext in ['.h5', '.hdf5', '.hdf']:
        hdf5_files.extend([f for f in os.listdir(input_dir) if f.lower().endswith(ext)])
    
    if not hdf5_files:
        print(f"No HDF5 files found in {input_dir}")
        return 0
    
    print(f"Found {len(hdf5_files)} HDF5 file(s)")
    
    total_images = 0
    next_index = start_index
    
    for hdf5_file in hdf5_files:
        filepath = os.path.join(input_dir, hdf5_file)
        print(f"\nProcessing: {hdf5_file}")
        
        try:
            with h5py.File(filepath, 'r') as f:
                # Print structure to understand the HDF5 format
                print(f"  Keys in file: {list(f.keys())}")
                
                # Try common dataset names
                data = None
                for key in ['data', 'images', 'ground_truth', 'observation', 'image', 'dataset']:
                    if key in f.keys():
                        data = f[key]
                        print(f"  Using key: '{key}'")
                        break
                
                # If no common key found, use first key
                if data is None:
                    first_key = list(f.keys())[0]
                    data = f[first_key]
                    print(f"  Using first key: '{first_key}'")
                
                # Get shape
                print(f"  Shape: {data.shape}")
                print(f"  Dtype: {data.dtype}")
                
                # Handle different dimensions
                if len(data.shape) == 3:
                    # (N, H, W) - multiple 2D images
                    num_images = data.shape[0]
                    if max_images:
                        num_images = min(num_images, max_images - total_images)
                    
                    print(f"  Extracting {num_images} images...")
                    for i in tqdm(range(num_images)):
                        img = data[i, :, :]
                        img = normalize_image(img)
                        img_pil = Image.fromarray(img, mode='L')
                        output_path = os.path.join(output_dir, f'image_{next_index:05d}.png')
                        img_pil.save(output_path)
                        total_images += 1
                        next_index += 1
                        
                        if max_images and total_images >= max_images:
                            break
                
                elif len(data.shape) == 2:
                    # (H, W) - single 2D image
                    img = data[:, :]
                    img = normalize_image(img)
                    img_pil = Image.fromarray(img, mode='L')
                    output_path = os.path.join(output_dir, f'image_{next_index:05d}.png')
                    img_pil.save(output_path)
                    total_images += 1
                    next_index += 1
                    print(f"  Extracted 1 image")
                
                elif len(data.shape) == 4:
                    # (N, C, H, W) or (N, H, W, C) - multiple images with channels
                    num_images = data.shape[0]
                    if max_images:
                        num_images = min(num_images, max_images - total_images)
                    
                    print(f"  Extracting {num_images} images (taking first channel)...")
                    for i in tqdm(range(num_images)):
                        # Take first channel
                        if data.shape[1] <= 3:  # (N, C, H, W)
                            img = data[i, 0, :, :]
                        else:  # (N, H, W, C)
                            img = data[i, :, :, 0]
                        
                        img = normalize_image(img)
                        img_pil = Image.fromarray(img, mode='L')
                        output_path = os.path.join(output_dir, f'image_{next_index:05d}.png')
                        img_pil.save(output_path)
                        total_images += 1
                        next_index += 1
                        
                        if max_images and total_images >= max_images:
                            break
                else:
                    print(f"  Unsupported shape: {data.shape}")
                    continue
                
        except Exception as e:
            print(f"  Error processing {hdf5_file}: {e}")
            continue
        
        if max_images and total_images >= max_images:
            print(f"\nReached maximum of {max_images} images")
            break
    
    print(f"\n✓ Total images extracted: {total_images}")
    print(f"✓ Saved to: {output_dir}")
    return total_images


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert HDF5 files to PNG images')
    parser.add_argument('--input_dir', required=True, help='Directory containing HDF5 files')
    parser.add_argument('--output_dir', default='data/raw/normal/test', help='Output directory for PNG files')
    parser.add_argument('--max_images', type=int, default=None, help='Maximum number of images to extract (default: all)')
    parser.add_argument('--start_index', type=int, default=None, help='Manual start index for output file numbering (default: auto-detect)')
    
    args = parser.parse_args()
    
    convert_hdf5_to_png(args.input_dir, args.output_dir, args.max_images, args.start_index)
