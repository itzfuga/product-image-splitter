#!/usr/bin/env python3
"""
Minimal Crop Taobao Product Extractor
Only removes separator text, keeps full model body with minimal cropping
"""

import cv2
import numpy as np
from PIL import Image
import os
from pathlib import Path
import json
import re
from typing import Dict, List, Tuple, Optional


class MinimalCropExtractor:
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
    
    def detect_separators_simple(self, image):
        """Simple separator detection for splitting"""
        try:
            height, width = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            separators = []
            
            # Look for separator in bottom 40% of image
            start_y = int(height * 0.6)
            
            for y in range(start_y, height - 50, 30):
                strip = gray[y:y+40, :]
                mean_val = np.mean(strip)
                std_val = np.std(strip)
                
                # Light gray area with text variation (separator characteristics)
                if 180 < mean_val < 240 and 10 < std_val < 60:
                    separators.append({'y': y, 'confidence': 80})
                    print(f"Found separator at y={y}")
                    break
            
            return separators
        except:
            return []
    
    def remove_only_separator_text(self, image):
        """Remove ONLY separator text areas, keep everything else"""
        try:
            height, width = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            print(f"Checking for separator text in {width}x{height} image")
            
            # Find separator text regions (light gray with text)
            rows_to_remove = []
            
            for y in range(height):
                row = gray[y, :]
                row_mean = np.mean(row)
                row_std = np.std(row)
                
                # Separator text characteristics: light gray (200-240) with text variation
                if 200 < row_mean < 240 and 15 < row_std < 50:
                    # Check if most of the row is separator-colored
                    separator_pixels = np.sum((row > 190) & (row < 250))
                    if separator_pixels > width * 0.6:  # 60% of row is separator
                        rows_to_remove.append(y)
            
            if len(rows_to_remove) > 10:  # Only remove if we find significant separator area
                print(f"Removing {len(rows_to_remove)} separator text rows")
                
                # Group consecutive rows and remove separator blocks
                groups = []
                if rows_to_remove:
                    current_group = [rows_to_remove[0]]
                    for i in range(1, len(rows_to_remove)):
                        if rows_to_remove[i] - rows_to_remove[i-1] <= 5:
                            current_group.append(rows_to_remove[i])
                        else:
                            groups.append(current_group)
                            current_group = [rows_to_remove[i]]
                    groups.append(current_group)
                
                # Remove separator blocks
                mask = np.ones(height, dtype=bool)
                for group in groups:
                    if len(group) > 8:  # Only remove significant separator blocks
                        start = max(0, group[0] - 5)
                        end = min(height, group[-1] + 5)
                        mask[start:end] = False
                        print(f"Removing separator block: rows {start}-{end}")
                
                image = image[mask, :]
                print(f"After separator removal: {image.shape[1]}x{image.shape[0]}")
            else:
                print("No significant separator text found")
                
            return image
            
        except Exception as e:
            print(f"Separator removal failed: {e}")
            return image
    
    def minimal_crop_only_edges(self, image):
        """Only crop pure white edges, keep all model content"""
        try:
            # First remove separator text
            image = self.remove_only_separator_text(image)
            
            height, width = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            print(f"Minimal cropping {width}x{height} image")
            
            # Find content bounds with very lenient threshold (only remove pure white)
            content_mask = gray < 248  # Almost pure white threshold
            
            # Find the bounds of ANY content
            rows_with_content = np.any(content_mask, axis=1)
            cols_with_content = np.any(content_mask, axis=0)
            
            if not np.any(rows_with_content) or not np.any(cols_with_content):
                print("No content found, returning original")
                return image
            
            row_indices = np.where(rows_with_content)[0]
            col_indices = np.where(cols_with_content)[0]
            
            # Get the absolute bounds of all content
            top_content = row_indices[0]
            bottom_content = row_indices[-1]
            left_content = col_indices[0]
            right_content = col_indices[-1]
            
            # Only crop if there's substantial pure white space (more than 5% of dimension)
            crop_top = 0
            crop_bottom = height
            crop_left = 0
            crop_right = width
            
            # Top cropping - only if more than 5% is pure white
            if top_content > height * 0.05:
                crop_top = max(0, top_content - 20)  # Keep 20px margin
                print(f"Cropping {crop_top}px from top")
            
            # Bottom cropping - only if more than 5% is pure white
            if (height - bottom_content) > height * 0.05:
                crop_bottom = min(height, bottom_content + 20)  # Keep 20px margin
                print(f"Cropping {height - crop_bottom}px from bottom")
            
            # Left cropping - only if more than 5% is pure white
            if left_content > width * 0.05:
                crop_left = max(0, left_content - 20)  # Keep 20px margin
                print(f"Cropping {crop_left}px from left")
            
            # Right cropping - only if more than 5% is pure white
            if (width - right_content) > width * 0.05:
                crop_right = min(width, right_content + 20)  # Keep 20px margin
                print(f"Cropping {width - crop_right}px from right")
            
            # Apply minimal cropping
            result = image[crop_top:crop_bottom, crop_left:crop_right]
            
            print(f"Minimal crop result: {result.shape[1]}x{result.shape[0]} (was {width}x{height})")
            
            return result
            
        except Exception as e:
            print(f"Minimal cropping failed: {e}")
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
            
            # Simple separator detection
            separators = self.detect_separators_simple(image)
            
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
        """Combine image parts with minimal cropping approach"""
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
                    
                    # Apply minimal cropping only
                    processed = self.minimal_crop_only_edges(combined)
                    
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
                    
                    print(f"Created minimal crop product {product_id}: {bottom_part['source_image']} + {matching_top['source_image']}")
                    product_id += 1
                    
                except Exception as e:
                    print(f"Error combining images: {e}")
                    continue
        
        # Handle full images (no separators detected)
        for full_part in fulls:
            try:
                full_img = full_part['image']
                
                # Apply minimal cropping only
                processed = self.minimal_crop_only_edges(full_img)
                
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
                
                print(f"Created minimal crop product {product_id}: {full_part['source_image']} (single)")
                product_id += 1
                
            except Exception as e:
                print(f"Error processing full image: {e}")
                continue
        
        return products


def main():
    """Test the Minimal Crop Extractor"""
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python minimal_crop_extractor.py <input_dir> <output_dir>")
        return
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    extractor = MinimalCropExtractor(output_dir)
    
    print("Loading images...")
    images = extractor.load_images(input_dir)
    if not images:
        print("No images found!")
        return
    
    print(f"Processing {len(images)} images...")
    image_parts = extractor.process_images(images)
    
    print("Combining image parts with minimal cropping...")
    products = extractor.combine_image_parts(image_parts)
    
    print(f"\nCreated {len(products)} minimal crop products:")
    for product in products:
        print(f"  {product['filename']}: {product.get('bottom_source', product.get('source', '?'))} + {product.get('top_source', '')}")
    
    print(f"\nMinimal crop processing complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()