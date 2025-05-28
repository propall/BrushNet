import os
import json
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import math
import glob

class CustomBrushNetDataset(Dataset):
    """
    Custom dataset class for BrushNet training with folder structure:
    dataset/
    ├── images/
    ├── masks/
    └── captions.json
    
    Now handles different file extensions intelligently!
    """
    
    def __init__(self, dataset_dir, resolution=512, tokenizer=None, random_mask=False):
        self.dataset_dir = dataset_dir
        self.images_dir = os.path.join(dataset_dir, "images")
        self.masks_dir = os.path.join(dataset_dir, "masks")
        self.captions_file = os.path.join(dataset_dir, "captions.json")
        self.resolution = resolution
        self.tokenizer = tokenizer
        self.random_mask = random_mask
        
        # Load captions
        with open(self.captions_file, 'r') as f:
            self.captions = json.load(f)
        
        print(f"Loaded {len(self.captions)} caption entries from {self.captions_file}")
        
        # Get list of image files that have both image and mask
        self.image_files = []
        self._find_matching_pairs()
        
        print(f"Found {len(self.image_files)} valid image-mask-caption triplets")
        
        if len(self.image_files) == 0:
            print("ERROR: No matching image-mask pairs found!")
            self._debug_file_matching()
    
    def _find_matching_pairs(self):
        """
        Intelligently find matching image-mask pairs, handling different file extensions
        """
        # Create a mapping of all available mask files by their base names
        mask_files_map = {}
        
        # Look for mask files with common extensions
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
            mask_pattern = os.path.join(self.masks_dir, ext)
            for mask_path in glob.glob(mask_pattern):
                mask_filename = os.path.basename(mask_path)
                # Extract the base name (remove the "mask_" prefix if present)
                if mask_filename.startswith('mask_'):
                    # Get the part after "mask_" and remove extension
                    base_name = mask_filename[5:]  # Remove "mask_" prefix
                    base_name_no_ext = os.path.splitext(base_name)[0]
                    mask_files_map[base_name_no_ext] = mask_path
                    # Also try with the original extension from image
                    mask_files_map[base_name] = mask_path
        
        print(f"Found {len(mask_files_map)} mask files")
        if len(mask_files_map) > 0:
            print(f"Sample mask mappings: {list(mask_files_map.items())[:3]}")
        
        # Now try to match each caption entry with image and mask
        for img_name in self.captions.keys():
            img_path = os.path.join(self.images_dir, img_name)
            
            # Check if image exists
            if not os.path.exists(img_path):
                print(f"Warning: Image not found: {img_path}")
                continue
            
            # Try to find corresponding mask
            mask_path = None
            
            # Strategy 1: Look for mask using the full filename (with extension)
            if img_name in mask_files_map:
                mask_path = mask_files_map[img_name]
            
            # Strategy 2: Look for mask using just the base name (without extension)
            if mask_path is None:
                base_name = os.path.splitext(img_name)[0]  # Remove extension from img_name
                if base_name in mask_files_map:
                    mask_path = mask_files_map[base_name]
            
            # Strategy 3: Direct file checking with different extensions
            if mask_path is None:
                base_name = os.path.splitext(img_name)[0]
                for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
                    potential_mask = os.path.join(self.masks_dir, f"mask_{base_name}{ext}")
                    if os.path.exists(potential_mask):
                        mask_path = potential_mask
                        break
            
            # If we found both image and mask, add to valid files
            if mask_path and os.path.exists(mask_path):
                self.image_files.append({
                    'name': img_name,
                    'image_path': img_path,
                    'mask_path': mask_path
                })
                print(f"✓ Matched: {img_name} -> {os.path.basename(mask_path)}")
            else:
                print(f"✗ No mask found for: {img_name}")
    
    def _debug_file_matching(self):
        """
        Provide detailed debugging information when no matches are found
        """
        print("\n" + "="*60)
        print("DEBUGGING FILE MATCHING ISSUES")
        print("="*60)
        
        print(f"\n1. Checking directories:")
        print(f"   Images dir exists: {os.path.exists(self.images_dir)}")
        print(f"   Masks dir exists: {os.path.exists(self.masks_dir)}")
        
        if os.path.exists(self.images_dir):
            image_files = os.listdir(self.images_dir)
            print(f"\n2. Found {len(image_files)} files in images directory:")
            for f in image_files[:5]:  # Show first 5
                print(f"   {f}")
            if len(image_files) > 5:
                print(f"   ... and {len(image_files) - 5} more")
        
        if os.path.exists(self.masks_dir):
            mask_files = os.listdir(self.masks_dir)
            print(f"\n3. Found {len(mask_files)} files in masks directory:")
            for f in mask_files[:5]:  # Show first 5
                print(f"   {f}")
            if len(mask_files) > 5:
                print(f"   ... and {len(mask_files) - 5} more")
        
        print(f"\n4. Sample caption entries:")
        for i, (key, value) in enumerate(list(self.captions.items())[:3]):
            print(f"   '{key}': '{value}'")
        
        print(f"\n5. Expected matching pattern:")
        if len(self.captions) > 0:
            sample_key = list(self.captions.keys())[0]
            expected_image = os.path.join(self.images_dir, sample_key)
            expected_mask_base = os.path.splitext(sample_key)[0]
            print(f"   For caption key '{sample_key}':")
            print(f"   Looking for image: {expected_image}")
            print(f"   Looking for mask: mask_{expected_mask_base}.[png|jpg|jpeg]")
        
        print("="*60 + "\n")
    
    def __len__(self):
        return len(self.image_files)
    
    def random_brush_gen(self, max_tries, h, w, min_num_vertex=0, max_num_vertex=8, 
                         mean_angle=2*math.pi/5, angle_range=2*math.pi/15, 
                         min_width=128, max_width=128):
        """Generate random brush strokes for data augmentation"""
        H, W = h, w
        average_radius = math.sqrt(H*H + W*W) / 8
        mask = Image.new('L', (W, H), 0)
        
        for _ in range(np.random.randint(max_tries)):
            num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
            angle_min = mean_angle - np.random.uniform(0, angle_range)
            angle_max = mean_angle + np.random.uniform(0, angle_range)
            angles = []
            vertex = []
            
            for i in range(num_vertex):
                if i % 2 == 0:
                    angles.append(2*math.pi - np.random.uniform(angle_min, angle_max))
                else:
                    angles.append(np.random.uniform(angle_min, angle_max))
            
            h, w = mask.size
            vertex.append((int(np.random.randint(0, w)), int(np.random.randint(0, h))))
            
            for i in range(num_vertex):
                r = np.clip(
                    np.random.normal(loc=average_radius, scale=average_radius//2),
                    0, 2*average_radius)
                new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
                new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
                vertex.append((int(new_x), int(new_y)))
            
            draw = ImageDraw.Draw(mask)
            width = int(np.random.uniform(min_width, max_width))
            draw.line(vertex, fill=1, width=width)
            for v in vertex:
                draw.ellipse((v[0] - width//2, v[1] - width//2,
                            v[0] + width//2, v[1] + width//2), fill=1)
            
            if np.random.random() > 0.5:
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            if np.random.random() > 0.5:
                mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        
        mask = np.asarray(mask, np.uint8)
        if np.random.random() > 0.5:
            mask = np.flip(mask, 0)
        if np.random.random() > 0.5:
            mask = np.flip(mask, 1)
        return mask
    
    def random_mask_gen(self, h, w):
        """Generate random mask for training"""
        mask = np.ones((h, w), np.uint8)
        mask = np.logical_and(mask, 1 - self.random_brush_gen(4, h, w))
        return mask[np.newaxis, ...].astype(np.float32)
    
    def tokenize_captions(self, caption, proportion_empty_prompts=0):
        """Tokenize text captions for training"""
        if random.random() < proportion_empty_prompts:
            caption = ""
        elif isinstance(caption, (list, np.ndarray)):
            caption = random.choice(caption)
        
        inputs = self.tokenizer(
            caption, 
            max_length=self.tokenizer.model_max_length, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )
        return inputs.input_ids[0]  # Remove batch dimension
    
    def __getitem__(self, idx):
        # Get file information for this index
        file_info = self.image_files[idx]
        img_name = file_info['name']
        img_path = file_info['image_path']
        mask_path = file_info['mask_path']
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Could not load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not load mask: {mask_path}")
        
        # Convert binary mask (0-255) to float mask (0-1)
        mask = (mask > 127).astype(np.float32)[:, :, np.newaxis]
        
        # Load caption
        caption = self.captions[img_name]
        
        # Use random mask if specified
        if self.random_mask:
            mask = self.random_mask_gen(image.shape[0], image.shape[1])[0][:, :, np.newaxis]
        
        # Apply morphological operations randomly (erosion/dilation)
        if random.random() < 0.3:
            kernel = np.ones((8, 8), np.uint8)
            mask_erosion = cv2.erode(mask, kernel, iterations=1)
            mask_dilation = cv2.dilate(mask_erosion, kernel, iterations=1)
            mask = (mask_dilation > 0)[:, :, np.newaxis].astype(np.float32)
        
        # Create masked image
        masked_image = image * mask
        
        # Randomly invert the masking operation
        if random.random() < 0.5:
            masked_image = image - masked_image
            mask = 1 - mask
        
        # Resize to target resolution while maintaining aspect ratio
        h, w, c = image.shape
        if w > h:
            scale = self.resolution / h
        else:
            scale = self.resolution / w
        
        w_new = int(np.ceil(w * scale))
        h_new = int(np.ceil(h * scale))
        
        # Resize all components
        image = cv2.resize(image, (w_new, h_new), interpolation=cv2.INTER_CUBIC)
        masked_image = cv2.resize(masked_image, (w_new, h_new), interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(mask, (w_new, h_new), interpolation=cv2.INTER_CUBIC)[:, :, np.newaxis]
        
        # Random crop to exact resolution
        random_crop = [
            random.randint(0, h_new - self.resolution),
            random.randint(0, w_new - self.resolution)
        ]
        
        image = image[random_crop[0]:random_crop[0] + self.resolution,
                     random_crop[1]:random_crop[1] + self.resolution, :]
        masked_image = masked_image[random_crop[0]:random_crop[0] + self.resolution,
                                  random_crop[1]:random_crop[1] + self.resolution, :]
        mask = mask[random_crop[0]:random_crop[0] + self.resolution,
                   random_crop[1]:random_crop[1] + self.resolution, :]
        
        # Normalize images to [-1, 1] range
        image = (image.astype(np.float32) / 127.5) - 1.0
        masked_image = (masked_image.astype(np.float32) / 127.5) - 1.0
        mask = mask.astype(np.float32)
        
        # Convert to tensors and rearrange dimensions
        pixel_values = torch.tensor(image).permute(2, 0, 1)
        conditioning_pixel_values = torch.tensor(masked_image).permute(2, 0, 1)
        masks = torch.tensor(mask).permute(2, 0, 1)
        input_ids = self.tokenize_captions(caption)
        
        return {
            "pixel_values": pixel_values,
            "conditioning_pixel_values": conditioning_pixel_values,
            "masks": masks,
            "input_ids": input_ids,
        }


def collate_fn(examples):
    """Custom collate function for batching"""
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    
    conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in examples])
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()
    
    masks = torch.stack([example["masks"] for example in examples])
    masks = masks.to(memory_format=torch.contiguous_format).float()
    
    input_ids = torch.stack([example["input_ids"] for example in examples])
    
    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        "masks": masks,
        "input_ids": input_ids,
    }