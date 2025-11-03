#!/usr/bin/env python3
"""
Final Fixed Extractor for Taobao Products
Fixes horizontal cropping and product numbering order
Forces consistent wide extraction for ALL products like products 5-9
"""

import cv2
import numpy as np
from PIL import Image
import os
from pathlib import Path
import json
import re
from typing import Dict, List, Tuple, Optional


class FinalFixedExtractor:
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
    
    def force_consistent_wide_extraction(self, image):
        """Force ALL products to use consistent wide extraction like products 5-9"""
        try:
            height, width = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            print(f"Forcing consistent wide extraction from {width}x{height} image")
            
            # Step 1: Find ANY content (model) as anchor point
            content_mask = gray < 240  # Very lenient to catch all model content
            
            # Clean up tiny noise
            kernel = np.ones((5, 5), np.uint8)
            content_mask = cv2.morphologyEx(content_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            
            # Find content bounds
            rows_with_content = np.any(content_mask, axis=1)
            cols_with_content = np.any(content_mask, axis=0)
            
            if not np.any(rows_with_content) or not np.any(cols_with_content):
                print("No content found, using center extraction")
                # Center extraction with wide dimensions
                margin_x = int(width * 0.05)  # 5% margin on each side
                margin_y = int(height * 0.05) 
                return (margin_x, margin_y, width - margin_x, height - margin_y)
            
            row_indices = np.where(rows_with_content)[0]
            col_indices = np.where(cols_with_content)[0]
            
            content_top = row_indices[0]
            content_bottom = row_indices[-1]
            content_left = col_indices[0]
            content_right = col_indices[-1]
            
            print(f"Detected content bounds: top={content_top}, bottom={content_bottom}, left={content_left}, right={content_right}")
            
            # Step 2: FORCE WIDE DIMENSIONS like products 5-9 
            # Products 5-9 have ~1820px width and work perfectly
            # Force ALL products to have similar wide dimensions
            
            # TARGET: Match the successful products 5-9 dimensions
            target_width_ratio = 0.75  # Use 75% of image width (like successful products)
            target_width = int(width * target_width_ratio)
            
            # Center the wide extraction on the detected content
            content_center_x = (content_left + content_right) // 2
            
            # Calculate wide bounds centered on content
            wide_left = max(0, content_center_x - target_width // 2)
            wide_right = min(width, wide_left + target_width)
            
            # Adjust if we hit boundaries
            if wide_right - wide_left < target_width:
                if wide_right == width:
                    wide_left = max(0, width - target_width)
                elif wide_left == 0:
                    wide_right = min(width, target_width)
            
            print(f"Forced wide horizontal bounds: {wide_left} to {wide_right} (width: {wide_right - wide_left})")
            
            # Step 3: CONSERVATIVE VERTICAL BOUNDS (head to boots)
            # Use very conservative vertical bounds to ensure full model
            
            # Top: Start higher to ensure head inclusion
            safe_top = max(0, content_top - int(height * 0.1))  # 10% above content
            if safe_top > height * 0.05:  # But not more than 5% from image top
                safe_top = int(height * 0.05)
            
            # Bottom: Go lower to ensure boots inclusion
            safe_bottom = min(height, content_bottom + int(height * 0.05))  # 5% below content
            if safe_bottom < height * 0.95:  # But at least to 95% of image
                safe_bottom = int(height * 0.95)
            
            print(f"Conservative vertical bounds: {safe_top} to {safe_bottom}")
            
            # Step 4: Remove separator text areas if present
            # Check for separator text in the bounds and exclude those areas
            
            final_top = safe_top
            final_bottom = safe_bottom
            
            # Scan for separator text rows and exclude them
            for y in range(safe_top, safe_bottom, 20):
                if y + 30 < safe_bottom:
                    strip = gray[y:y+30, wide_left:wide_right]
                    if strip.size > 0:
                        mean_val = np.mean(strip)
                        std_val = np.std(strip)
                        
                        # If this looks like separator text
                        if 190 < mean_val < 240 and 10 < std_val < 50:
                            separator_pixels = np.sum((strip > 180) & (strip < 250))
                            if separator_pixels > strip.size * 0.5:  # More than 50% separator
                                print(f"Found separator text at y={y}, excluding region")
                                if y < height // 2:  # Separator in top half
                                    final_top = max(final_top, y + 50)  # Start after separator
                                else:  # Separator in bottom half
                                    final_bottom = min(final_bottom, y - 20)  # End before separator
            
            print(f"FINAL consistent wide bounds: ({wide_left},{final_top}) size {wide_right-wide_left}x{final_bottom-final_top}")
            
            return (wide_left, final_top, wide_right, final_bottom)
            
        except Exception as e:
            print(f"Consistent wide extraction failed: {e}")
            # Emergency fallback with wide dimensions
            margin_x = int(image.shape[1] * 0.1)  # 10% margins
            margin_y = int(image.shape[0] * 0.05)  # 5% margins
            return (margin_x, margin_y, image.shape[1] - margin_x, image.shape[0] - margin_y)
    
    def extract_white_rectangle_content(self, image):
        """Extract with forced consistent wide dimensions"""
        try:
            # Force consistent wide extraction like products 5-9
            left, top, right, bottom = self.force_consistent_wide_extraction(image)
            
            # Extract the consistently wide content
            extracted = image[top:bottom, left:right]
            
            print(f"Extracted CONSISTENT WIDE: {right-left}x{bottom-top} from {image.shape[1]}x{image.shape[0]}")
            
            return extracted
            
        except Exception as e:
            print(f"Consistent wide extraction failed: {e}")
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
    
    def combine_chronologically_fixed_order(self, image_parts):
        """Combine image parts with FIXED PRODUCT NUMBERING and consistent wide extraction"""
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
        
        # FIX PRODUCT NUMBERING: Process full images FIRST in chronological order
        # This ensures images 1.jpg and 2.jpg become products 1 and 2 (not 11 and 12)
        
        print("\n=== PROCESSING FULL IMAGES FIRST (FIXED ORDER) ===")
        for full_part in fulls:
            try:
                print(f"\nCreating product {product_id} from full image: {full_part['source_image']}")
                
                full_img = full_part['image']
                
                # Extract with CONSISTENT WIDE dimensions
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
        
        print(f"\n=== PROCESSING COMBINED IMAGES (CHRONOLOGICAL PAIRING) ===")
        # THEN process combined images: Bottom(N) + Top(N+1) = Product N
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
                    
                    # Extract with CONSISTENT WIDE dimensions
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
        
        return products


def main():
    """Test the Final Fixed Extractor"""
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python final_fixed_extractor.py <input_dir> <output_dir>")
        return
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    extractor = FinalFixedExtractor(output_dir)
    
    print("Loading images in chronological order...")
    images = extractor.load_images(input_dir)
    if not images:
        print("No images found!")
        return
    
    print(f"\nProcessing {len(images)} images chronologically...")
    image_parts = extractor.process_images_chronologically(images)
    
    print("\nCombining parts with FIXED ORDER and CONSISTENT WIDE extraction...")
    products = extractor.combine_chronologically_fixed_order(image_parts)
    
    print(f"\nCreated {len(products)} consistently wide products:")
    for product in products:
        if 'top_source' in product:
            print(f"  {product['filename']}: {product['bottom_source']} + {product['top_source']}")
        else:
            print(f"  {product['filename']}: {product['source']} (single)")
    
    print(f"\nFinal fixed extraction complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()