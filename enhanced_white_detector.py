#!/usr/bin/env python3
"""
Enhanced White Rectangle Detector for Taobao Products
Better white rectangle boundary detection using edge detection and contours
"""

import cv2
import numpy as np
from PIL import Image
import os
from pathlib import Path
import json
import re
from typing import Dict, List, Tuple, Optional


class EnhancedWhiteDetector:
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
    
    def find_white_rectangle_enhanced(self, image):
        """Enhanced white rectangle detection using multiple methods"""
        try:
            height, width = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            print(f"Enhanced white rectangle detection in {width}x{height} image")
            
            # Method 1: Edge-based detection
            # Find edges to detect the boundary between gray background and white rectangle
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Find strong vertical and horizontal lines (rectangle boundaries)
            # Horizontal lines
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
            
            # Vertical lines  
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)
            
            # Combine lines
            rectangle_edges = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)
            
            # Method 2: Color-based detection
            # Look for the transition from gray background to white rectangle
            # Gray background is typically 200-220, white rectangle is 240-255
            
            # Create mask for white areas (potential rectangle content)
            white_mask = gray > 235
            
            # Create mask for gray background areas
            gray_mask = (gray > 180) & (gray < 235)
            
            # Method 3: Find the largest rectangular white region
            # Use contour detection on the white mask
            contours, _ = cv2.findContours(white_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            best_rect = None
            largest_area = 0
            
            for contour in contours:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h
                
                # Filter for reasonable rectangle sizes and aspect ratios
                min_area = width * height * 0.15  # At least 15% of image
                max_area = width * height * 0.85  # At most 85% of image
                
                if min_area < area < max_area:
                    aspect_ratio = w / h if h > 0 else 0
                    # Reasonable aspect ratios for product rectangles
                    if 0.2 < aspect_ratio < 2.5:
                        if area > largest_area:
                            largest_area = area
                            best_rect = (x, y, w, h)
            
            if best_rect:
                x, y, w, h = best_rect
                print(f"Found white rectangle via contours: ({x},{y}) size {w}x{h}")
                return (x, y, x + w, y + h)
            
            # Method 4: Fallback - analyze image structure
            # Look for the central white region bounded by gray areas
            
            # Analyze horizontal structure - find top and bottom gray borders
            row_means = np.mean(gray, axis=1)
            
            # Find top border (gray area at top)
            top_border = 0
            for y in range(min(200, height // 4)):  # Check first 25% or 200px
                if row_means[y] < 230:  # Found non-white area
                    top_border = y
                    break
            
            # Find bottom border (gray area at bottom)
            bottom_border = height
            for y in range(height - 1, max(height - 200, height * 3 // 4), -1):  # Check last 25% or 200px
                if row_means[y] < 230:  # Found non-white area
                    bottom_border = y
                    break
            
            # Analyze vertical structure - find left and right gray borders
            col_means = np.mean(gray, axis=0)
            
            # Find left border
            left_border = 0
            for x in range(min(200, width // 4)):  # Check first 25% or 200px
                if col_means[x] < 230:  # Found non-white area
                    left_border = x
                    break
            
            # Find right border
            right_border = width
            for x in range(width - 1, max(width - 200, width * 3 // 4), -1):  # Check last 25% or 200px
                if col_means[x] < 230:  # Found non-white area
                    right_border = x
                    break
            
            # Add some margin to ensure we get the white rectangle content
            margin = 20
            final_left = max(0, left_border + margin)
            final_top = max(0, top_border + margin)
            final_right = min(width, right_border - margin)
            final_bottom = min(height, bottom_border - margin)
            
            # Validate the rectangle
            rect_width = final_right - final_left
            rect_height = final_bottom - final_top
            
            if rect_width > width * 0.2 and rect_height > height * 0.2:
                print(f"Found white rectangle via structure analysis: ({final_left},{final_top}) to ({final_right},{final_bottom})")
                return (final_left, final_top, final_right, final_bottom)
            else:
                print("No valid white rectangle found, using full image")
                return (0, 0, width, height)
            
        except Exception as e:
            print(f"Enhanced white rectangle detection failed: {e}")
            return (0, 0, image.shape[1], image.shape[0])
    
    def extract_white_rectangle_content(self, image):
        """Extract ONLY the content inside the white rectangle"""
        try:
            # Find white rectangle bounds
            left, top, right, bottom = self.find_white_rectangle_enhanced(image)
            
            # Extract the white rectangle content
            extracted = image[top:bottom, left:right]
            
            print(f"Extracted white rectangle: {right-left}x{bottom-top} from {image.shape[1]}x{image.shape[0]}")
            
            return extracted
            
        except Exception as e:
            print(f"White rectangle extraction failed: {e}")
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
                    
                    # Extract ONLY the white rectangle content with enhanced detection
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
                
                # Extract white rectangle content with enhanced detection
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
    """Test the Enhanced White Detector"""
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python enhanced_white_detector.py <input_dir> <output_dir>")
        return
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    extractor = EnhancedWhiteDetector(output_dir)
    
    print("Loading images in chronological order...")
    images = extractor.load_images(input_dir)
    if not images:
        print("No images found!")
        return
    
    print(f"\nProcessing {len(images)} images chronologically...")
    image_parts = extractor.process_images_chronologically(images)
    
    print("\nCombining parts chronologically and extracting white rectangles...")
    products = extractor.combine_chronologically(image_parts)
    
    print(f"\nCreated {len(products)} enhanced white rectangle products:")
    for product in products:
        if 'top_source' in product:
            print(f"  {product['filename']}: {product['bottom_source']} + {product['top_source']}")
        else:
            print(f"  {product['filename']}: {product['source']} (single)")
    
    print(f"\nEnhanced white rectangle extraction complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()