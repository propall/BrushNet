import os
import json
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import math

class CustomBrushNetDataset(Dataset):
    """
    Custom dataset class for BrushNet training with folder structure:
    dataset/
    ├── images/
    ├── masks/
    └── captions.json
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
        
        # Get list of image files that have both image and mask
        self.image_files = []
        for img_name in self.captions.keys():
            img_path = os.path.join(self.images_dir, img_name)
            # Create mask filename by adding "mask_" prefix
            mask_name = f"mask_{img_name}"
            mask_path = os.path.join(self.masks_dir, mask_name)
            
            if os.path.exists(img_path) and os.path.exists(mask_path):
                self.image_files.append(img_name)
            else:
                print(f"Warning: Missing files for {img_name}")
        
        print(f"Found {len(self.image_files)} valid image-mask-caption triplets")
    
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
        img_name = self.image_files[idx]
        
        # Load image
        img_path = os.path.join(self.images_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask_name = f"mask_{img_name}"
        mask_path = os.path.join(self.masks_dir, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
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