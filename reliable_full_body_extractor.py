#!/usr/bin/env python3
"""
Reliable Full-Body Taobao Product Extractor
Focus: Extract complete model views without separator text or extreme cropping
"""

import cv2
import numpy as np
from PIL import Image
import os
from pathlib import Path
import json
import re
from typing import Dict, List, Tuple, Optional


class ReliableFullBodyExtractor:
    def __init__(self, result_dir, session_id=None):
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(exist_ok=True)
        self.session_id = session_id
        
    def load_images(self, image_dir):
        """Load all images from directory in natural sort order"""
        image_dir = Path(image_dir)
        images = []
        
        supported_formats = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.avif']
        
        def natural_sort_key(path):
            s = path.name
            return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]
        
        for file_path in sorted(image_dir.glob('*'), key=natural_sort_key):
            if file_path.suffix.lower() in supported_formats:
                try:
                    img = cv2.imread(str(file_path))
                    if img is not None:
                        images.append({
                            'path': str(file_path),
                            'name': file_path.name,
                            'image': img,
                            'index': len(images)
                        })
                        print(f"Loaded: {file_path.name}")
                except Exception as e:
                    print(f"Error loading {file_path.name}: {e}")
        return images
    
    def detect_separators_robust(self, image):
        """More robust separator detection"""
        try:
            height, width = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            separators = []
            
            # Look for text regions that contain "START EXCEED END" or similar separator patterns
            # Check bottom third of image for separators (where they typically appear)
            start_y = int(height * 0.6)  # Start looking from 60% down
            
            for y in range(start_y, height - 50, 20):
                # Check a larger horizontal strip for separator characteristics
                strip_height = 40
                if y + strip_height >= height:
                    continue
                    
                strip = gray[y:y+strip_height, :]
                mean_val = np.mean(strip)
                std_val = np.std(strip)
                
                # Separator characteristics: light gray background with text
                # Typical range for separator areas: 180-240 brightness with text variation
                if 180 < mean_val < 240 and 10 < std_val < 60:
                    # Additional check: look for horizontal text-like patterns
                    horizontal_gradient = np.gradient(strip, axis=1)
                    if np.std(horizontal_gradient) > 5:  # Has text-like horizontal variation
                        separators.append({'y': y, 'confidence': 85})
                        print(f"Found separator at y={y} (mean={mean_val:.1f}, std={std_val:.1f})")
                        break  # Only find one separator per image
            
            return separators
        except Exception as e:
            print(f"Separator detection error: {e}")
            return []
    
    def extract_full_body_product(self, image):
        """Extract full body product with no extreme cropping"""
        try:
            height, width = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            print(f"Extracting full body from {width}x{height} image")
            
            # Step 1: Remove any separator text areas completely
            image = self.remove_separator_regions(image)
            height, width = image.shape[:2]  # Update dimensions after cleaning
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Step 2: Find the central content area (model) with conservative approach
            # Use a more conservative threshold to ensure we capture the full model
            
            # Create content mask - anything that's not pure white background
            content_mask = gray < 240  # Very conservative threshold
            
            # Remove small noise
            kernel = np.ones((5, 5), np.uint8)
            content_mask = cv2.morphologyEx(content_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            content_mask = cv2.morphologyEx(content_mask, cv2.MORPH_OPEN, kernel)
            
            # Find the bounding box of all content
            rows_with_content = np.any(content_mask, axis=1)
            cols_with_content = np.any(content_mask, axis=0)
            
            if not np.any(rows_with_content) or not np.any(cols_with_content):
                print("No content found, returning original")
                return image
            
            row_indices = np.where(rows_with_content)[0]
            col_indices = np.where(cols_with_content)[0]
            
            content_top = row_indices[0]
            content_bottom = row_indices[-1]
            content_left = col_indices[0]
            content_right = col_indices[-1]
            
            # Add VERY generous padding to ensure full model is captured
            # Padding should be at least 10% of image dimensions
            vertical_padding = max(50, int(height * 0.1))
            horizontal_padding = max(50, int(width * 0.1))
            
            # Apply padding but don't go outside image bounds
            final_top = max(0, content_top - vertical_padding)
            final_bottom = min(height, content_bottom + vertical_padding)
            final_left = max(0, content_left - horizontal_padding)
            final_right = min(width, content_right + horizontal_padding)
            
            # Ensure minimum dimensions (at least 60% of original width/height)
            min_width = int(width * 0.6)
            min_height = int(height * 0.6)
            
            if (final_right - final_left) < min_width:
                print("Width too small, expanding...")
                center_x = (final_left + final_right) // 2
                final_left = max(0, center_x - min_width // 2)
                final_right = min(width, final_left + min_width)
            
            if (final_bottom - final_top) < min_height:
                print("Height too small, expanding...")
                center_y = (final_top + final_bottom) // 2
                final_top = max(0, center_y - min_height // 2)
                final_bottom = min(height, final_top + min_height)
            
            # Extract the full body region
            extracted = image[final_top:final_bottom, final_left:final_right]
            
            print(f"Extracted full body: {final_right-final_left}x{final_bottom-final_top} from original {width}x{height}")
            
            return extracted
            
        except Exception as e:
            print(f"Full body extraction failed: {e}")
            return image
    
    def remove_separator_regions(self, image):
        """Aggressively remove separator text regions"""
        try:
            height, width = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Look for separator regions - typically light gray areas with text
            separator_mask = (gray > 170) & (gray < 250)
            
            # Find horizontal strips that look like separators
            rows_to_remove = []
            
            # Check each row for separator characteristics
            for y in range(height):
                row = gray[y, :]
                row_mean = np.mean(row)
                row_std = np.std(row)
                
                # If this row looks like separator text (light gray with variation)
                if 170 < row_mean < 250 and 5 < row_std < 50:
                    # Check if significant portion of row is in separator range
                    separator_pixels = np.sum((row > 170) & (row < 250))
                    if separator_pixels > width * 0.3:  # At least 30% of row
                        rows_to_remove.append(y)
            
            if rows_to_remove:
                print(f"Removing {len(rows_to_remove)} separator rows")
                
                # Group consecutive rows
                groups = []
                current_group = [rows_to_remove[0]]
                
                for i in range(1, len(rows_to_remove)):
                    if rows_to_remove[i] - rows_to_remove[i-1] <= 3:  # Close rows
                        current_group.append(rows_to_remove[i])
                    else:
                        groups.append(current_group)
                        current_group = [rows_to_remove[i]]
                groups.append(current_group)
                
                # Remove groups of separator rows (with some margin)
                mask = np.ones(height, dtype=bool)
                for group in groups:
                    if len(group) > 5:  # Only remove if it's a significant separator region
                        start_remove = max(0, group[0] - 10)
                        end_remove = min(height, group[-1] + 10)
                        mask[start_remove:end_remove] = False
                        print(f"Removing separator region: rows {start_remove}-{end_remove}")
                
                # Apply mask to remove separator rows
                image = image[mask, :]
                
            return image
            
        except Exception as e:
            print(f"Separator removal failed: {e}")
            return image
    
    def split_image_at_separator(self, image, separator_y):
        """Split image into two parts at separator position"""
        height = image.shape[0]
        top_part = image[0:separator_y, :]
        bottom_part = image[separator_y:height, :]
        return top_part, bottom_part
    
    def process_images(self, images):
        """Process images and split them at separators"""
        image_parts = []
        
        for img_data in images:
            print(f"\nProcessing: {img_data['name']}")
            image = img_data['image']
            
            # Detect separators with robust method
            separators = self.detect_separators_robust(image)
            
            if separators:
                separator_y = separators[0]['y']
                print(f"Using separator at y={separator_y}")
                
                # Split image at separator
                top_part, bottom_part = self.split_image_at_separator(image, separator_y)
                
                # Store both parts
                image_parts.append({
                    'image': top_part,
                    'type': 'top',
                    'source_image': img_data['name'],
                    'source_index': img_data['index']
                })
                
                image_parts.append({
                    'image': bottom_part,
                    'type': 'bottom',
                    'source_image': img_data['name'],
                    'source_index': img_data['index']
                })
                
                print(f"Split into top ({top_part.shape[0]}px) and bottom ({bottom_part.shape[0]}px)")
            else:
                # No separator found - treat as single image
                print("No separator found - treating as single image")
                image_parts.append({
                    'image': image,
                    'type': 'full',
                    'source_image': img_data['name'],
                    'source_index': img_data['index']
                })
        
        return image_parts
    
    def combine_image_parts(self, image_parts):
        """Combine image parts into complete product images with full body extraction"""
        products = []
        product_id = 1
        
        # Group parts by type
        tops = [part for part in image_parts if part['type'] == 'top']
        bottoms = [part for part in image_parts if part['type'] == 'bottom']
        fulls = [part for part in image_parts if part['type'] == 'full']
        
        print(f"Found {len(tops)} tops, {len(bottoms)} bottoms, {len(fulls)} full images")
        
        # Combine bottom of image N with top of image N+1
        for i in range(len(bottoms)):
            bottom_part = bottoms[i]
            
            # Look for the corresponding top part from the next image
            target_source_index = bottom_part['source_index'] + 1
            matching_top = None
            
            for top_part in tops:
                if top_part['source_index'] == target_source_index:
                    matching_top = top_part
                    break
            
            if matching_top:
                # Combine bottom + top to create complete product
                try:
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
                    
                    # Combine vertically (bottom first, then top)
                    combined = np.vstack([bottom_img, top_img])
                    
                    # Extract full body product
                    processed = self.extract_full_body_product(combined)
                    
                    # Save product
                    filename = f"product_{product_id}.jpg"
                    output_path = self.result_dir / filename
                    cv2.imwrite(str(output_path), processed)
                    
                    products.append({
                        'product_id': product_id,
                        'filename': filename,
                        'path': str(output_path),
                        'bottom_source': bottom_part['source_image'],
                        'top_source': matching_top['source_image'],
                        'dimensions': f"{processed.shape[1]}x{processed.shape[0]}"
                    })
                    
                    print(f"Created full body product {product_id}: {bottom_part['source_image']} + {matching_top['source_image']}")
                    product_id += 1
                    
                except Exception as e:
                    print(f"Error combining images: {e}")
                    continue
        
        # Handle full images (no separators detected)
        for full_part in fulls:
            try:
                full_img = full_part['image']
                
                # Extract full body product
                processed = self.extract_full_body_product(full_img)
                
                # Save product
                filename = f"product_{product_id}.jpg"
                output_path = self.result_dir / filename
                cv2.imwrite(str(output_path), processed)
                
                products.append({
                    'product_id': product_id,
                    'filename': filename,
                    'path': str(output_path),
                    'source': full_part['source_image'],
                    'type': 'single_image',
                    'dimensions': f"{processed.shape[1]}x{processed.shape[0]}"
                })
                
                print(f"Created full body product {product_id}: {full_part['source_image']} (single)")
                product_id += 1
                
            except Exception as e:
                print(f"Error processing full image: {e}")
                continue
        
        return products


def main():
    """Test the Reliable Full Body Extractor"""
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python reliable_full_body_extractor.py <input_dir> <output_dir>")
        return
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    extractor = ReliableFullBodyExtractor(output_dir)
    
    print("Loading images...")
    images = extractor.load_images(input_dir)
    if not images:
        print("No images found!")
        return
    
    print(f"Processing {len(images)} images...")
    image_parts = extractor.process_images(images)
    
    print("Combining image parts into full body products...")
    products = extractor.combine_image_parts(image_parts)
    
    print(f"\nCreated {len(products)} full body products:")
    for product in products:
        print(f"  {product['filename']}: {product.get('bottom_source', product.get('source', '?'))} + {product.get('top_source', '')}")
    
    print(f"\nFull body processing complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()