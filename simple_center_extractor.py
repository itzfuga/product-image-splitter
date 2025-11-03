#!/usr/bin/env python3
"""
Simple Center-Based Taobao Product Extractor
Focuses on finding the center white rectangle with the model
"""

import cv2
import numpy as np
from PIL import Image
import os
from pathlib import Path
import json
import re
from typing import Dict, List, Tuple, Optional


class SimpleCenterExtractor:
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
    
    def detect_separators(self, image):
        """Simple separator detection"""
        try:
            height, width = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            separators = []
            
            # Look for horizontal bands that are light gray with some variation (text)
            for y in range(100, height - 100, 30):
                strip = gray[y:y+30, :]
                mean_val = np.mean(strip)
                std_val = np.std(strip)
                
                # Separator characteristics: light gray (200-230) with some text variation
                if 200 < mean_val < 230 and 15 < std_val < 40:
                    separators.append({'y': y, 'confidence': 80})
                    print(f"Found separator at y={y}")
                    break  # Only find one separator per image
            
            return separators
        except:
            return []
    
    def extract_center_product_rectangle(self, image):
        """Extract the center white rectangle containing the product - SIMPLE approach"""
        try:
            height, width = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            print(f"Extracting center rectangle from {width}x{height} image")
            
            # Step 1: Find the center region of the image (where products usually are)
            center_x = width // 2
            center_y = height // 2
            
            # Step 2: Look for a large white/light rectangular area near the center
            # Start from center and expand outward to find the product bounds
            
            # Find horizontal bounds (left and right edges of white area)
            left_bound = 0
            right_bound = width
            
            # Scan from center outward to find content bounds
            for x in range(center_x, 50, -10):  # Scan left from center
                col = gray[:, x]
                if np.mean(col) < 180:  # Hit non-white area
                    left_bound = max(0, x - 20)  # Add small margin
                    break
            
            for x in range(center_x, width - 50, 10):  # Scan right from center
                col = gray[:, x]
                if np.mean(col) < 180:  # Hit non-white area
                    right_bound = min(width, x + 20)  # Add small margin
                    break
            
            # Find vertical bounds (top and bottom edges)
            top_bound = 0
            bottom_bound = height
            
            for y in range(50, height - 50, 10):  # Scan from top
                row = gray[y, left_bound:right_bound]
                if np.mean(row) < 200 and np.sum(row < 150) > len(row) * 0.1:  # Has significant content
                    top_bound = max(0, y - 20)
                    break
            
            for y in range(height - 50, 50, -10):  # Scan from bottom
                row = gray[y, left_bound:right_bound]
                if np.mean(row) < 200 and np.sum(row < 150) > len(row) * 0.1:  # Has significant content
                    bottom_bound = min(height, y + 20)
                    break
            
            # Ensure we have a reasonable rectangle
            rect_width = right_bound - left_bound
            rect_height = bottom_bound - top_bound
            
            if rect_width < width * 0.2 or rect_height < height * 0.2:
                print("Rectangle too small, using smart fallback")
                return self.smart_content_extraction(image)
            
            # Extract the rectangle
            extracted = image[top_bound:bottom_bound, left_bound:right_bound]
            
            print(f"Extracted center rectangle: {rect_width}x{rect_height} at ({left_bound},{top_bound})")
            
            # Clean any remaining gray edges
            cleaned = self.ultra_clean_edges(extracted)
            
            return cleaned
            
        except Exception as e:
            print(f"Center extraction failed: {e}")
            return self.smart_content_extraction(image)
    
    def remove_separator_text_completely(self, image):
        """Remove any remaining separator text and gray areas"""
        try:
            height, width = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Look for separator text regions (gray areas with text)
            # Separator areas are typically light gray (190-230) with text
            separator_mask = (gray > 190) & (gray < 230)
            
            # If we find separator regions, remove them
            if np.sum(separator_mask) > width * 20:  # Significant separator area
                print("Removing separator text regions")
                
                # Find rows that contain separator text
                separator_rows = np.any(separator_mask, axis=1)
                separator_row_indices = np.where(separator_rows)[0]
                
                if len(separator_row_indices) > 0:
                    # Remove separator rows (usually at top and bottom)
                    # Remove from top
                    top_separators = separator_row_indices[separator_row_indices < height // 2]
                    if len(top_separators) > 0:
                        remove_from_top = top_separators[-1] + 20  # Remove up to last separator + margin
                        image = image[remove_from_top:, :]
                        gray = gray[remove_from_top:, :]
                        height = image.shape[0]
                    
                    # Remove from bottom
                    bottom_separators = separator_row_indices[separator_row_indices >= height // 2]
                    if len(bottom_separators) > 0:
                        remove_from_bottom = bottom_separators[0] - 20  # Remove from first separator - margin
                        image = image[:remove_from_bottom, :]
                        gray = gray[:remove_from_bottom, :]
            
            return image
            
        except Exception as e:
            print(f"Separator text removal failed: {e}")
            return image
    
    def ultra_clean_edges(self, image):
        """Ultra-aggressive edge cleaning to remove ALL gray pixels"""
        try:
            # First remove separator text
            image = self.remove_separator_text_completely(image)
            
            height, width = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Find content (anything that's clearly not background)
            # Be very strict: only keep very dark content or pure white
            content_mask = (gray < 160) | (gray > 250)
            
            # Find tight bounds
            rows_with_content = np.any(content_mask, axis=1)
            cols_with_content = np.any(content_mask, axis=0)
            
            if not np.any(rows_with_content) or not np.any(cols_with_content):
                return image
            
            row_indices = np.where(rows_with_content)[0]
            col_indices = np.where(cols_with_content)[0]
            
            min_row, max_row = row_indices[0], row_indices[-1]
            min_col, max_col = col_indices[0], col_indices[-1]
            
            # Add minimal padding but ensure white background
            padding = 10
            min_row = max(0, min_row - padding)
            min_col = max(0, min_col - padding)
            max_row = min(height - 1, max_row + padding)
            max_col = min(width - 1, max_col + padding)
            
            # Extract the ultra-clean region
            ultra_clean = image[min_row:max_row+1, min_col:max_col+1]
            
            print(f"Ultra-cleaned: {max_col-min_col+1}x{max_row-min_row+1}")
            return ultra_clean
            
        except Exception as e:
            print(f"Ultra cleaning failed: {e}")
            return image
    
    def smart_content_extraction(self, image):
        """Fallback: smart content extraction with separator removal"""
        try:
            # First remove separator text
            image = self.remove_separator_text_completely(image)
            
            height, width = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Find actual content (model)
            content_mask = gray < 170  # Dark content (stricter)
            
            # Clean up small noise
            kernel = np.ones((5, 5), np.uint8)
            content_mask = cv2.morphologyEx(content_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            
            # Find bounding box of content
            rows = np.any(content_mask, axis=1)
            cols = np.any(content_mask, axis=0)
            
            if np.any(rows) and np.any(cols):
                row_indices = np.where(rows)[0]
                col_indices = np.where(cols)[0]
                
                min_y, max_y = row_indices[0], row_indices[-1]
                min_x, max_x = col_indices[0], col_indices[-1]
                
                # Add white background padding
                padding = 25
                min_x = max(0, min_x - padding)
                min_y = max(0, min_y - padding)
                max_x = min(width, max_x + padding)
                max_y = min(height, max_y + padding)
                
                extracted = image[min_y:max_y, min_x:max_x]
                print(f"Smart extraction: {max_x-min_x}x{max_y-min_y}")
                return extracted
            
            return image
            
        except Exception as e:
            print(f"Smart extraction failed: {e}")
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
            
            # Detect separators
            separators = self.detect_separators(image)
            
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
        """Combine image parts into complete product images with center extraction"""
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
                    
                    # Extract center product rectangle
                    processed = self.extract_center_product_rectangle(combined)
                    
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
                    
                    print(f"Created simple product {product_id}: {bottom_part['source_image']} + {matching_top['source_image']}")
                    product_id += 1
                    
                except Exception as e:
                    print(f"Error combining images: {e}")
                    continue
        
        # Handle full images (no separators detected)
        for full_part in fulls:
            try:
                full_img = full_part['image']
                
                # Extract center product rectangle
                processed = self.extract_center_product_rectangle(full_img)
                
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
                
                print(f"Created simple product {product_id}: {full_part['source_image']} (single)")
                product_id += 1
                
            except Exception as e:
                print(f"Error processing full image: {e}")
                continue
        
        return products


def main():
    """Test the Simple Center Extractor"""
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python simple_center_extractor.py <input_dir> <output_dir>")
        return
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    extractor = SimpleCenterExtractor(output_dir)
    
    print("Loading images...")
    images = extractor.load_images(input_dir)
    if not images:
        print("No images found!")
        return
    
    print(f"Processing {len(images)} images...")
    image_parts = extractor.process_images(images)
    
    print("Combining image parts into simple products...")
    products = extractor.combine_image_parts(image_parts)
    
    print(f"\nCreated {len(products)} simple products:")
    for product in products:
        print(f"  {product['filename']}: {product.get('bottom_source', product.get('source', '?'))} + {product.get('top_source', '')}")
    
    print(f"\nSimple processing complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()