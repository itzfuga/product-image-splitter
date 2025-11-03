#!/usr/bin/env python3
"""
Forced Wide Extractor for Taobao Products
Forces all extractions to be wide enough to capture full models
Ensures consistent results regardless of source rectangle layout
"""

import cv2
import numpy as np
from PIL import Image
import os
from pathlib import Path
import json
import re
from typing import Dict, List, Tuple, Optional


class ForcedWideExtractor:
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
    
    def force_wide_white_rectangle(self, image):
        """Force wide white rectangle extraction - ensures full model capture"""
        try:
            height, width = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            print(f"Forced wide extraction from {width}x{height} image")
            
            # Step 1: Find ANY white content as starting point
            white_mask = gray > 230
            
            # Clean up mask
            kernel = np.ones((10, 10), np.uint8)
            white_mask = cv2.morphologyEx(white_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
            
            # Find content bounds
            rows_with_content = np.any(white_mask, axis=1)
            cols_with_content = np.any(white_mask, axis=0)
            
            if not np.any(rows_with_content) or not np.any(cols_with_content):
                print("No white content found, using center extraction")
                # Fallback: extract center 70% of image
                margin_x = int(width * 0.15)
                margin_y = int(height * 0.1)
                return (margin_x, margin_y, width - margin_x, height - margin_y)
            
            row_indices = np.where(rows_with_content)[0]
            col_indices = np.where(cols_with_content)[0]
            
            content_top = row_indices[0]
            content_bottom = row_indices[-1]
            content_left = col_indices[0]
            content_right = col_indices[-1]
            
            # Step 2: FORCE WIDE EXTRACTION
            # Minimum width should be at least 60% of image width for full model
            min_width = int(width * 0.6)  # Force at least 60% width
            current_width = content_right - content_left
            
            if current_width < min_width:
                print(f"Current width {current_width} too narrow, forcing to {min_width}")
                
                # Expand horizontally from center of detected content
                content_center_x = (content_left + content_right) // 2
                new_left = max(0, content_center_x - min_width // 2)
                new_right = min(width, new_left + min_width)
                
                # Adjust if we hit edge
                if new_right == width:
                    new_left = max(0, width - min_width)
                
                content_left = new_left
                content_right = new_right
                print(f"Forced wide bounds: left={content_left}, right={content_right}")
            
            # Step 3: Ensure minimum height (at least 70% for full body)
            min_height = int(height * 0.7)
            current_height = content_bottom - content_top
            
            if current_height < min_height:
                print(f"Current height {current_height} too short, forcing to {min_height}")
                
                # Expand vertically from center of detected content
                content_center_y = (content_top + content_bottom) // 2
                new_top = max(0, content_center_y - min_height // 2)
                new_bottom = min(height, new_top + min_height)
                
                # Adjust if we hit edge
                if new_bottom == height:
                    new_top = max(0, height - min_height)
                
                content_top = new_top
                content_bottom = new_bottom
                print(f"Forced height bounds: top={content_top}, bottom={content_bottom}")
            
            # Step 4: Add reasonable margins but keep forced dimensions
            margin = 20
            final_left = max(0, content_left - margin)
            final_top = max(0, content_top - margin)
            final_right = min(width, content_right + margin)
            final_bottom = min(height, content_bottom + margin)
            
            final_width = final_right - final_left
            final_height = final_bottom - final_top
            
            print(f"Forced wide rectangle: ({final_left},{final_top}) size {final_width}x{final_height}")
            
            return (final_left, final_top, final_right, final_bottom)
            
        except Exception as e:
            print(f"Forced wide extraction failed: {e}")
            # Emergency fallback: center 70% of image
            margin_x = int(image.shape[1] * 0.15)
            margin_y = int(image.shape[0] * 0.15)
            return (margin_x, margin_y, image.shape[1] - margin_x, image.shape[0] - margin_y)
    
    def extract_white_rectangle_content(self, image):
        """Extract white rectangle content with forced wide dimensions"""
        try:
            # Force wide rectangle extraction
            left, top, right, bottom = self.force_wide_white_rectangle(image)
            
            # Extract the forced wide rectangle content
            extracted = image[top:bottom, left:right]
            
            print(f"Extracted forced wide rectangle: {right-left}x{bottom-top} from {image.shape[1]}x{image.shape[0]}")
            
            return extracted
            
        except Exception as e:
            print(f"Forced wide rectangle extraction failed: {e}")
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
        """Combine image parts in strict chronological order with forced wide extraction"""
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
                    
                    # Extract with FORCED WIDE dimensions
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
        
        # Handle full images (no separators) with forced wide extraction
        for full_part in fulls:
            try:
                print(f"\nProcessing full image: {full_part['source_image']}")
                
                full_img = full_part['image']
                
                # Extract with FORCED WIDE dimensions
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
    """Test the Forced Wide Extractor"""
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python forced_wide_extractor.py <input_dir> <output_dir>")
        return
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    extractor = ForcedWideExtractor(output_dir)
    
    print("Loading images in chronological order...")
    images = extractor.load_images(input_dir)
    if not images:
        print("No images found!")
        return
    
    print(f"\nProcessing {len(images)} images chronologically...")
    image_parts = extractor.process_images_chronologically(images)
    
    print("\nCombining parts chronologically with FORCED WIDE extraction...")
    products = extractor.combine_chronologically(image_parts)
    
    print(f"\nCreated {len(products)} forced wide products:")
    for product in products:
        if 'top_source' in product:
            print(f"  {product['filename']}: {product['bottom_source']} + {product['top_source']}")
        else:
            print(f"  {product['filename']}: {product['source']} (single)")
    
    print(f"\nForced wide extraction complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()