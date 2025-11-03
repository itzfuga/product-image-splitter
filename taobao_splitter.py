#!/usr/bin/env python3
"""
Taobao Product Image Splitter
Specifically designed for Taobao images with "START EXCEED END" separators
"""

import cv2
import numpy as np
import pytesseract
from PIL import Image
import os
from pathlib import Path
import json
import re
from typing import Dict, List, Tuple, Optional


class TaobaoSplitter:
    def __init__(self, result_dir, session_id=None):
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(exist_ok=True)
        self.session_id = session_id
        
        # Processing settings
        self.auto_crop = True
        self.background_color = (255, 255, 255)  # White background
        
    def load_images(self, image_dir):
        """Load all images from directory in natural sort order"""
        image_dir = Path(image_dir)
        images = []
        
        supported_formats = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.avif']
        
        # Natural sort to handle numeric filenames properly (1, 2, 3... not 1, 10, 11...)
        def natural_sort_key(path):
            """Extract numbers from filename for proper sorting"""
            s = path.name
            return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]
        
        for file_path in sorted(image_dir.glob('*'), key=natural_sort_key):
            if file_path.suffix.lower() in supported_formats:
                try:
                    # Try OpenCV first
                    img = cv2.imread(str(file_path))
                    if img is not None:
                        images.append({
                            'path': str(file_path),
                            'name': file_path.name,
                            'image': img,
                            'index': len(images)
                        })
                        print(f"Loaded: {file_path.name}")
                    else:
                        # Fallback to PIL for formats like AVIF
                        pil_img = Image.open(file_path).convert('RGB')
                        cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                        images.append({
                            'path': str(file_path),
                            'name': file_path.name,
                            'image': cv2_img,
                            'index': len(images)
                        })
                        print(f"Loaded (PIL): {file_path.name}")
                        
                except Exception as e:
                    print(f"Error loading {file_path.name}: {e}")
                    
        return images
    
    def detect_separator_text(self, image):
        """Detect 'START EXCEED END' separator text using OCR"""
        try:
            # Convert to PIL if needed
            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = image
            
            # Use OCR to find text
            try:
                data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
            except Exception as ocr_error:
                print(f"OCR failed: {ocr_error}")
                return []
            
            separator_positions = []
            
            # Look for "START EXCEED END" text patterns
            all_text = ' '.join([text for text in data['text'] if text.strip()])
            print(f"OCR detected text: {all_text}")
            
            # Check if this contains separator text
            text_upper = all_text.upper().replace(' ', '').replace('\n', '')
            separator_patterns = [
                r'START.*EXCEED.*END',
                r'STAFF.*START.*EXCEED.*END',
                r'STAFF.*EXCEED.*END',
                r'STARTEXCEEDEND',
                r'STAFFSTARTEXCEEDEND'
            ]
            
            for pattern in separator_patterns:
                if re.search(pattern, text_upper):
                    # Find the Y position of the separator text
                    for i, text in enumerate(data['text']):
                        if text.strip():
                            text_clean = text.upper().replace(' ', '').replace('\n', '')
                            if any(word in text_clean for word in ['START', 'EXCEED', 'END', 'STAFF']):
                                y = data['top'][i]
                                height = data['height'][i]
                                separator_positions.append({
                                    'y': y + height // 2,  # Center of text
                                    'height': height,
                                    'text': text,
                                    'confidence': data['conf'][i]
                                })
                                print(f"Found separator text: '{text}' at y={y}")
                                break
                    break
            
            return separator_positions
            
        except Exception as e:
            print(f"OCR error: {e}")
            return []
    
    def detect_visual_separator(self, image):
        """Detect the visual separator line (gray background with text)"""
        try:
            height, width = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Look for horizontal regions with uniform gray/white background
            separator_positions = []
            
            # Analyze each row for background characteristics
            for y in range(50, height - 50):  # Skip edges
                row = gray[y, :]
                
                # Check if this row is mostly uniform and light colored
                row_mean = np.mean(row)
                row_std = np.std(row)
                
                # Separator characteristics: light colored with low variance
                if row_mean > 200 and row_std < 30:  # Light and uniform
                    # Check surrounding rows too
                    region_start = max(0, y - 25)
                    region_end = min(height, y + 25)
                    region = gray[region_start:region_end, :]
                    
                    region_mean = np.mean(region)
                    region_std = np.std(region)
                    
                    if region_mean > 190 and region_std < 50:
                        separator_positions.append({
                            'y': y,
                            'height': 50,
                            'method': 'visual',
                            'confidence': 80
                        })
                        print(f"Found visual separator at y={y}")
                        break  # Only find one separator per image
            
            return separator_positions
            
        except Exception as e:
            print(f"Visual separator detection error: {e}")
            return []
    
    def split_image_at_separator(self, image, separator_y):
        """Split image into two parts at separator position"""
        height = image.shape[0]
        
        # Split into top and bottom parts
        # Top part: from start to separator
        top_part = image[0:separator_y, :]
        
        # Bottom part: from separator to end
        bottom_part = image[separator_y:height, :]
        
        return top_part, bottom_part
    
    def process_images(self, images):
        """Process images and split them at separators"""
        image_parts = []
        
        for img_data in images:
            print(f"\nProcessing: {img_data['name']}")
            image = img_data['image']
            
            # Try OCR first
            text_separators = self.detect_separator_text(image)
            
            # Try visual detection as backup
            visual_separators = self.detect_visual_separator(image)
            
            # Use the best separator we found
            separator_y = None
            if text_separators:
                separator_y = text_separators[0]['y']
                print(f"Using OCR separator at y={separator_y}")
            elif visual_separators:
                separator_y = visual_separators[0]['y']
                print(f"Using visual separator at y={separator_y}")
            
            if separator_y:
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
    
    def auto_crop_image(self, image):
        """Remove white space around the actual content"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Find all non-white pixels (threshold at 240)
            content_mask = gray < 240
            
            # Find bounds
            content_rows = np.any(content_mask, axis=1)
            content_cols = np.any(content_mask, axis=0)
            
            content_row_indices = np.where(content_rows)[0]
            content_col_indices = np.where(content_cols)[0]
            
            if len(content_row_indices) == 0 or len(content_col_indices) == 0:
                return image  # No content found, return original
            
            min_y = content_row_indices[0]
            max_y = content_row_indices[-1]
            min_x = content_col_indices[0]
            max_x = content_col_indices[-1]
            
            # Add some padding
            padding = 10
            min_x = max(0, min_x - padding)
            min_y = max(0, min_y - padding)
            max_x = min(image.shape[1], max_x + padding)
            max_y = min(image.shape[0], max_y + padding)
            
            # Crop the image
            cropped = image[min_y:max_y, min_x:max_x]
            return cropped
            
        except Exception as e:
            print(f"Auto-crop failed: {e}")
            return image
    
    def combine_image_parts(self, image_parts):
        """Combine image parts into complete product images according to Taobao pattern"""
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
                    
                    # Auto-crop to remove excess whitespace
                    if self.auto_crop:
                        combined = self.auto_crop_image(combined)
                    
                    # Save product
                    filename = f"product_{product_id}.jpg"
                    output_path = self.result_dir / filename
                    cv2.imwrite(str(output_path), combined)
                    
                    products.append({
                        'product_id': product_id,
                        'filename': filename,
                        'path': str(output_path),
                        'bottom_source': bottom_part['source_image'],
                        'top_source': matching_top['source_image'],
                        'dimensions': f"{combined.shape[1]}x{combined.shape[0]}"
                    })
                    
                    print(f"Created product {product_id}: {bottom_part['source_image']} (bottom) + {matching_top['source_image']} (top)")
                    product_id += 1
                    
                except Exception as e:
                    print(f"Error combining images: {e}")
                    continue
        
        # Handle full images (no separators detected)
        for full_part in fulls:
            try:
                full_img = full_part['image']
                
                # Auto-crop to remove excess whitespace
                if self.auto_crop:
                    full_img = self.auto_crop_image(full_img)
                
                # Save product
                filename = f"product_{product_id}.jpg"
                output_path = self.result_dir / filename
                cv2.imwrite(str(output_path), full_img)
                
                products.append({
                    'product_id': product_id,
                    'filename': filename,
                    'path': str(output_path),
                    'source': full_part['source_image'],
                    'type': 'single_image',
                    'dimensions': f"{full_img.shape[1]}x{full_img.shape[0]}"
                })
                
                print(f"Created product {product_id}: {full_part['source_image']} (single image)")
                product_id += 1
                
            except Exception as e:
                print(f"Error processing full image: {e}")
                continue
        
        return products


def main():
    """Test the Taobao splitter"""
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python taobao_splitter.py <input_dir> <output_dir>")
        return
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    splitter = TaobaoSplitter(output_dir)
    
    print("Loading images...")
    images = splitter.load_images(input_dir)
    if not images:
        print("No images found!")
        return
    
    print(f"Processing {len(images)} images...")
    image_parts = splitter.process_images(images)
    
    print("Combining image parts into products...")
    products = splitter.combine_image_parts(image_parts)
    
    print(f"\nCreated {len(products)} products:")
    for product in products:
        print(f"  {product['filename']}: {product.get('bottom_source', product.get('source', '?'))} + {product.get('top_source', '')}")
    
    # Save processing info
    info = {
        'total_images': len(images),
        'total_products': len(products),
        'products': products
    }
    
    info_path = Path(output_dir) / "taobao_processing_info.json"
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"\nProcessing complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()