#!/usr/bin/env python3
"""
Robust White Rectangle Detector for Taobao Products
Always finds the largest, most central white rectangle
"""

import cv2
import numpy as np
from PIL import Image
import os
from pathlib import Path
import json
import re
from typing import Dict, List, Tuple, Optional


class RobustWhiteDetector:
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
    
    def find_largest_central_white_rectangle(self, image):
        """Find the largest, most central white rectangle - ROBUST method"""
        try:
            height, width = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            print(f"Robust white rectangle detection in {width}x{height} image")
            
            # Create multiple masks with different thresholds to catch all white rectangles
            white_masks = []
            
            # Threshold 1: Very white areas (245-255)
            white_masks.append(gray > 245)
            
            # Threshold 2: White-ish areas (235-255) 
            white_masks.append(gray > 235)
            
            # Threshold 3: Light areas (225-255)
            white_masks.append(gray > 225)
            
            all_candidates = []
            
            # Find contours for each threshold
            for i, mask in enumerate(white_masks):
                # Clean up mask
                kernel = np.ones((10, 10), np.uint8)
                mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                
                # Find contours
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    area = w * h
                    
                    # Filter for reasonable rectangles
                    min_area = width * height * 0.08  # At least 8% of image
                    max_area = width * height * 0.90  # At most 90% of image
                    
                    if min_area < area < max_area:
                        aspect_ratio = w / h if h > 0 else 0
                        # Accept wider range of aspect ratios
                        if 0.15 < aspect_ratio < 4.0:
                            # Calculate centrality score (how close to center)
                            center_x = x + w // 2
                            center_y = y + h // 2
                            img_center_x = width // 2
                            img_center_y = height // 2
                            
                            # Distance from center (normalized)
                            center_distance = ((center_x - img_center_x) ** 2 + (center_y - img_center_y) ** 2) ** 0.5
                            max_distance = ((width // 2) ** 2 + (height // 2) ** 2) ** 0.5
                            centrality_score = 1.0 - (center_distance / max_distance)
                            
                            # Combined score: area + centrality
                            score = area * 0.7 + centrality_score * area * 0.3
                            
                            all_candidates.append({
                                'x': x, 'y': y, 'w': w, 'h': h,
                                'area': area,
                                'centrality': centrality_score,
                                'score': score,
                                'threshold': i
                            })
            
            if not all_candidates:
                print("No white rectangle candidates found")
                return (0, 0, width, height)
            
            # Sort by score (area + centrality)
            all_candidates.sort(key=lambda c: c['score'], reverse=True)
            
            # Take the best candidate
            best = all_candidates[0]
            
            print(f"Best white rectangle: ({best['x']},{best['y']}) size {best['w']}x{best['h']} (area={best['area']}, centrality={best['centrality']:.2f})")
            
            # Add small margin but ensure we don't go outside bounds
            margin = 5
            final_x = max(0, best['x'] - margin)
            final_y = max(0, best['y'] - margin)
            final_w = min(width - final_x, best['w'] + 2 * margin)
            final_h = min(height - final_y, best['h'] + 2 * margin)
            
            return (final_x, final_y, final_x + final_w, final_y + final_h)
            
        except Exception as e:
            print(f"Robust white rectangle detection failed: {e}")
            return (0, 0, image.shape[1], image.shape[0])
    
    def extract_white_rectangle_content(self, image):
        """Extract ONLY the content inside the largest, most central white rectangle"""
        try:
            # Find white rectangle bounds with robust method
            left, top, right, bottom = self.find_largest_central_white_rectangle(image)
            
            # Extract the white rectangle content
            extracted = image[top:bottom, left:right]
            
            print(f"Extracted robust white rectangle: {right-left}x{bottom-top} from {image.shape[1]}x{image.shape[0]}")
            
            return extracted
            
        except Exception as e:
            print(f"Robust white rectangle extraction failed: {e}")
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
        """Combine image parts in strict chronological order"""
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
                    
                    # Extract ONLY the largest, most central white rectangle
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
        
        # Handle full images (no separators)
        for full_part in fulls:
            try:
                print(f"\nProcessing full image: {full_part['source_image']}")
                
                full_img = full_part['image']
                
                # Extract largest, most central white rectangle
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
    """Test the Robust White Detector"""
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python robust_white_detector.py <input_dir> <output_dir>")
        return
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    extractor = RobustWhiteDetector(output_dir)
    
    print("Loading images in chronological order...")
    images = extractor.load_images(input_dir)
    if not images:
        print("No images found!")
        return
    
    print(f"\nProcessing {len(images)} images chronologically...")
    image_parts = extractor.process_images_chronologically(images)
    
    print("\nCombining parts chronologically and extracting largest central white rectangles...")
    products = extractor.combine_chronologically(image_parts)
    
    print(f"\nCreated {len(products)} robust white rectangle products:")
    for product in products:
        if 'top_source' in product:
            print(f"  {product['filename']}: {product['bottom_source']} + {product['top_source']}")
        else:
            print(f"  {product['filename']}: {product['source']} (single)")
    
    print(f"\nRobust white rectangle extraction complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()