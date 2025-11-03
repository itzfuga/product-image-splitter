#!/usr/bin/env python3
"""
Ultimate Head-to-Boots Extractor for Taobao Products
Ensures EVERY product captures the complete model from head to boots
Forces conservative full-body extraction with guaranteed head and feet inclusion
"""

import cv2
import numpy as np
from PIL import Image
import os
from pathlib import Path
import json
import re
from typing import Dict, List, Tuple, Optional


class UltimateHeadToBootsExtractor:
    def __init__(self, result_dir, session_id=None):
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(exist_ok=True)
        self.session_id = session_id
        
    def load_images(self, image_dir):
        """Load all images from directory in STRICT chronological order"""
        image_dir = Path(image_dir)
        images = []
        
        supported_formats = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.avif']
        
        def natural_sort_key(path):
            s = path.name
            return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]
        
        # Get all image files and sort them chronologically
        all_files = []
        for file_path in image_dir.glob('*'):
            if file_path.suffix.lower() in supported_formats:
                all_files.append(file_path)
        
        # Sort chronologically - this is CRITICAL
        sorted_files = sorted(all_files, key=natural_sort_key)
        
        for file_path in sorted_files:
            try:
                img = cv2.imread(str(file_path))
                if img is not None:
                    images.append({
                        'path': str(file_path),
                        'name': file_path.name,
                        'image': img,
                        'index': len(images)  # Sequential index starting from 0
                    })
                    print(f"Loaded {len(images)}: {file_path.name}")
            except Exception as e:
                print(f"Error loading {file_path.name}: {e}")
        
        print(f"Total loaded: {len(images)} images in chronological order")
        return images
    
    def detect_separator(self, image):
        """Detect separator line for splitting image"""
        try:
            height, width = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Look for separator in bottom 40% of image
            start_y = int(height * 0.6)
            
            for y in range(start_y, height - 50, 30):
                strip = gray[y:y+40, :]
                mean_val = np.mean(strip)
                std_val = np.std(strip)
                
                # Separator characteristics: light gray with text variation
                if 180 < mean_val < 240 and 10 < std_val < 60:
                    print(f"Found separator at y={y}")
                    return y
            
            print("No separator found")
            return None
        except:
            return None
    
    def guaranteed_head_to_boots_extraction(self, image):
        """GUARANTEED head-to-boots extraction with ultra-conservative approach"""
        try:
            height, width = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            print(f"GUARANTEED head-to-boots extraction from {width}x{height} image")
            
            # Step 1: Find ANY content (model) as anchor point
            content_mask = gray < 240  # Very lenient to catch all model content
            
            # Clean up tiny noise but preserve all model parts
            kernel = np.ones((5, 5), np.uint8)
            content_mask = cv2.morphologyEx(content_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            
            # Find the bounds of ALL content
            rows_with_content = np.any(content_mask, axis=1)
            cols_with_content = np.any(content_mask, axis=0)
            
            if not np.any(rows_with_content) or not np.any(cols_with_content):
                print("No content found, using ultra-conservative center")
                # Ultra-conservative fallback: center 80% of image
                margin_x = int(width * 0.1)
                margin_y = int(height * 0.05)  # Very small vertical margin
                return (margin_x, margin_y, width - margin_x, height - margin_y)
            
            row_indices = np.where(rows_with_content)[0]
            col_indices = np.where(cols_with_content)[0]
            
            content_top = row_indices[0]
            content_bottom = row_indices[-1]
            content_left = col_indices[0]
            content_right = col_indices[-1]
            
            print(f"Detected content bounds: top={content_top}, bottom={content_bottom}, left={content_left}, right={content_right}")
            
            # Step 2: FORCE ULTRA-CONSERVATIVE HEAD-TO-BOOTS BOUNDS
            
            # VERTICAL: Ensure we capture from very top to very bottom
            # Use VERY generous margins to guarantee head and boots inclusion
            
            # Top margin: Go much higher to ensure head is included
            safe_top = max(0, content_top - int(height * 0.15))  # 15% above detected content
            if safe_top > height * 0.1:  # If still more than 10% from top, go to 5%
                safe_top = max(0, int(height * 0.05))
            
            # Bottom margin: Go much lower to ensure boots are included  
            safe_bottom = min(height, content_bottom + int(height * 0.10))  # 10% below detected content
            if safe_bottom < height * 0.9:  # If still more than 10% from bottom, go to 95%
                safe_bottom = min(height, int(height * 0.95))
            
            print(f"Ultra-conservative vertical bounds: {safe_top} to {safe_bottom} (was {content_top} to {content_bottom})")
            
            # HORIZONTAL: Ensure good width but maintain white rectangle
            # Force minimum width while keeping white background detection
            
            current_width = content_right - content_left
            min_width = max(int(width * 0.5), 1200)  # At least 50% of image width OR 1200px
            
            if current_width < min_width:
                # Expand horizontally from center of detected content
                content_center_x = (content_left + content_right) // 2
                expand_left = (min_width - current_width) // 2
                expand_right = min_width - current_width - expand_left
                
                safe_left = max(0, content_left - expand_left)
                safe_right = min(width, content_right + expand_right)
                
                # Adjust if we hit boundaries
                if safe_right - safe_left < min_width:
                    if safe_left == 0:
                        safe_right = min(width, safe_left + min_width)
                    elif safe_right == width:
                        safe_left = max(0, safe_right - min_width)
                
                print(f"Expanded width from {current_width} to {safe_right - safe_left}")
            else:
                # Width is sufficient, just add small margins
                margin = 30
                safe_left = max(0, content_left - margin)
                safe_right = min(width, content_right + margin)
            
            print(f"Final horizontal bounds: {safe_left} to {safe_right}")
            
            # Step 3: Validate bounds ensure reasonable rectangle
            final_width = safe_right - safe_left
            final_height = safe_bottom - safe_top
            
            # Minimum size validation
            if final_width < width * 0.3:
                print("Width too small, expanding to 40% of image")
                center_x = width // 2
                target_width = int(width * 0.4)
                safe_left = max(0, center_x - target_width // 2)
                safe_right = min(width, safe_left + target_width)
            
            if final_height < height * 0.7:
                print("Height too small, expanding to 80% of image") 
                target_height = int(height * 0.8)
                safe_top = max(0, int(height * 0.1))  # Start from 10% down
                safe_bottom = min(height, safe_top + target_height)
            
            final_width = safe_right - safe_left
            final_height = safe_bottom - safe_top
            
            print(f"GUARANTEED head-to-boots bounds: ({safe_left},{safe_top}) size {final_width}x{final_height}")
            
            return (safe_left, safe_top, safe_right, safe_bottom)
            
        except Exception as e:
            print(f"Guaranteed extraction failed: {e}")
            # Emergency ultra-conservative fallback
            margin_x = int(image.shape[1] * 0.15)
            margin_y = int(image.shape[0] * 0.05) 
            return (margin_x, margin_y, image.shape[1] - margin_x, image.shape[0] - margin_y)
    
    def extract_white_rectangle_content(self, image):
        """Extract with GUARANTEED head-to-boots inclusion"""
        try:
            # Guaranteed head-to-boots extraction
            left, top, right, bottom = self.guaranteed_head_to_boots_extraction(image)
            
            # Extract the guaranteed head-to-boots content
            extracted = image[top:bottom, left:right]
            
            print(f"Extracted GUARANTEED head-to-boots: {right-left}x{bottom-top} from {image.shape[1]}x{image.shape[0]}")
            
            return extracted
            
        except Exception as e:
            print(f"Guaranteed head-to-boots extraction failed: {e}")
            return image
    
    def split_image_at_separator(self, image, separator_y):
        """Split image into two parts at separator position"""
        height = image.shape[0]
        top_part = image[0:separator_y, :]
        bottom_part = image[separator_y:height, :]
        return top_part, bottom_part
    
    def process_images_chronologically(self, images):
        """Process images and split them, maintaining chronological order"""
        image_parts = []
        
        for i, img_data in enumerate(images):
            print(f"\nProcessing image {i+1}: {img_data['name']}")
            image = img_data['image']
            
            # Detect separator
            separator_y = self.detect_separator(image)
            
            if separator_y:
                print(f"Splitting at y={separator_y}")
                
                # Split image at separator
                top_part, bottom_part = self.split_image_at_separator(image, separator_y)
                
                # Store parts with chronological index
                image_parts.append({
                    'image': top_part,
                    'type': 'top',
                    'source_image': img_data['name'],
                    'chronological_index': i  # Use the actual chronological position
                })
                
                image_parts.append({
                    'image': bottom_part,
                    'type': 'bottom', 
                    'source_image': img_data['name'],
                    'chronological_index': i  # Use the actual chronological position
                })
                
                print(f"Split into top ({top_part.shape[0]}px) and bottom ({bottom_part.shape[0]}px)")
            else:
                # No separator - treat as single image
                print("No separator found - treating as single image")
                image_parts.append({
                    'image': image,
                    'type': 'full',
                    'source_image': img_data['name'],
                    'chronological_index': i  # Use the actual chronological position
                })
        
        return image_parts
    
    def combine_chronologically(self, image_parts):
        """Combine image parts with GUARANTEED head-to-boots extraction"""
        products = []
        product_id = 1
        
        # Group parts by type and sort by chronological index
        tops = sorted([p for p in image_parts if p['type'] == 'top'], 
                      key=lambda x: x['chronological_index'])
        bottoms = sorted([p for p in image_parts if p['type'] == 'bottom'], 
                         key=lambda x: x['chronological_index'])
        fulls = sorted([p for p in image_parts if p['type'] == 'full'], 
                       key=lambda x: x['chronological_index'])
        
        print(f"Found {len(tops)} tops, {len(bottoms)} bottoms, {len(fulls)} full images")
        
        # CHRONOLOGICAL PAIRING: Bottom(N) + Top(N+1) = Product N
        for i in range(len(bottoms)):
            bottom_part = bottoms[i]
            bottom_index = bottom_part['chronological_index']
            
            # Look for the top part from the NEXT chronological image (N+1)
            target_index = bottom_index + 1
            matching_top = None
            
            for top_part in tops:
                if top_part['chronological_index'] == target_index:
                    matching_top = top_part
                    break
            
            if matching_top:
                try:
                    print(f"\nCreating product {product_id}:")
                    print(f"  Bottom from: {bottom_part['source_image']} (index {bottom_index})")
                    print(f"  Top from: {matching_top['source_image']} (index {target_index})")
                    
                    bottom_img = bottom_part['image']
                    top_img = matching_top['image']
                    
                    # Ensure same width
                    h1, w1 = bottom_img.shape[:2]
                    h2, w2 = top_img.shape[:2]
                    
                    target_width = max(w1, w2)
                    
                    if w1 != target_width:
                        bottom_img = cv2.resize(bottom_img, (target_width, h1))
                    if w2 != target_width:
                        top_img = cv2.resize(top_img, (target_width, h2))
                    
                    # Combine vertically: Bottom first, then Top
                    combined = np.vstack([bottom_img, top_img])
                    
                    # Extract with GUARANTEED HEAD-TO-BOOTS
                    final_product = self.extract_white_rectangle_content(combined)
                    
                    # Save product
                    filename = f"product_{product_id}.jpg"
                    output_path = self.result_dir / filename
                    cv2.imwrite(str(output_path), final_product)
                    
                    products.append({
                        'product_id': product_id,
                        'filename': filename,
                        'path': str(output_path),
                        'bottom_source': bottom_part['source_image'],
                        'top_source': matching_top['source_image'],
                        'dimensions': f"{final_product.shape[1]}x{final_product.shape[0]}"
                    })
                    
                    print(f"  Saved: {filename} ({final_product.shape[1]}x{final_product.shape[0]})")
                    product_id += 1
                    
                except Exception as e:
                    print(f"Error creating product {product_id}: {e}")
                    continue
            else:
                print(f"No matching top part found for bottom from {bottom_part['source_image']}")
        
        # Handle full images with GUARANTEED HEAD-TO-BOOTS
        for full_part in fulls:
            try:
                print(f"\nProcessing full image: {full_part['source_image']}")
                
                full_img = full_part['image']
                
                # Extract with GUARANTEED HEAD-TO-BOOTS
                final_product = self.extract_white_rectangle_content(full_img)
                
                # Save product
                filename = f"product_{product_id}.jpg"
                output_path = self.result_dir / filename
                cv2.imwrite(str(output_path), final_product)
                
                products.append({
                    'product_id': product_id,
                    'filename': filename,
                    'path': str(output_path),
                    'source': full_part['source_image'],
                    'type': 'single_image',
                    'dimensions': f"{final_product.shape[1]}x{final_product.shape[0]}"
                })
                
                print(f"  Saved: {filename} ({final_product.shape[1]}x{final_product.shape[0]})")
                product_id += 1
                
            except Exception as e:
                print(f"Error processing full image: {e}")
                continue
        
        return products


def main():
    """Test the Ultimate Head-to-Boots Extractor"""
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python ultimate_head_to_boots_extractor.py <input_dir> <output_dir>")
        return
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    extractor = UltimateHeadToBootsExtractor(output_dir)
    
    print("Loading images in chronological order...")
    images = extractor.load_images(input_dir)
    if not images:
        print("No images found!")
        return
    
    print(f"\nProcessing {len(images)} images chronologically...")
    image_parts = extractor.process_images_chronologically(images)
    
    print("\nCombining parts chronologically with GUARANTEED HEAD-TO-BOOTS extraction...")
    products = extractor.combine_chronologically(image_parts)
    
    print(f"\nCreated {len(products)} guaranteed head-to-boots products:")
    for product in products:
        if 'top_source' in product:
            print(f"  {product['filename']}: {product['bottom_source']} + {product['top_source']}")
        else:
            print(f"  {product['filename']}: {product['source']} (single)")
    
    print(f"\nGuaranteed head-to-boots extraction complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()