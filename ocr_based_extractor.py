#!/usr/bin/env python3
"""
OCR-Based Taobao Product Extractor
Uses OCR to find and completely eliminate "START EXCEED END" text
Then extracts full model views with conservative approach
"""

import cv2
import numpy as np
from PIL import Image
import os
from pathlib import Path
import json
import re
from typing import Dict, List, Tuple, Optional

try:
    import pytesseract
    HAS_OCR = True
except ImportError:
    HAS_OCR = False
    print("Warning: pytesseract not available. Will use basic separator detection.")


class OCRBasedExtractor:
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
    
    def find_separator_text_with_ocr(self, image):
        """Use OCR to find 'START EXCEED END' text locations"""
        try:
            if not HAS_OCR:
                return []
            
            height, width = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Convert to PIL Image for OCR
            pil_image = Image.fromarray(gray)
            
            # Get OCR data with bounding boxes
            ocr_data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
            
            separator_regions = []
            
            # Look for separator text patterns
            separator_patterns = [
                'START', 'EXCEED', 'END', 'START EXCEED END',
                'STA', 'EXCEE', 'E D'  # Account for OCR errors
            ]
            
            for i, text in enumerate(ocr_data['text']):
                if text.strip():
                    # Check if this text matches separator patterns
                    text_clean = text.strip().upper()
                    for pattern in separator_patterns:
                        if pattern in text_clean or text_clean in pattern:
                            x = ocr_data['left'][i]
                            y = ocr_data['top'][i]
                            w = ocr_data['width'][i]
                            h = ocr_data['height'][i]
                            
                            separator_regions.append({
                                'x': x, 'y': y, 'width': w, 'height': h,
                                'text': text, 'confidence': ocr_data['conf'][i]
                            })
                            print(f"Found separator text '{text}' at ({x},{y}) size {w}x{h}")
            
            return separator_regions
            
        except Exception as e:
            print(f"OCR separator detection failed: {e}")
            return []
    
    def detect_separators_with_ocr(self, image):
        """Detect separators using OCR and fallback methods"""
        separators = []
        
        # First try OCR approach
        ocr_regions = self.find_separator_text_with_ocr(image)
        
        if ocr_regions:
            # Find the main separator line (usually the largest/most confident)
            best_region = max(ocr_regions, key=lambda r: r.get('confidence', 0))
            separator_y = best_region['y'] + best_region['height'] // 2
            separators.append({'y': separator_y, 'confidence': 90, 'method': 'ocr'})
            print(f"OCR found separator at y={separator_y}")
        else:
            # Fallback to visual detection
            height, width = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Look for separator characteristics in bottom 40% of image
            start_y = int(height * 0.6)
            
            for y in range(start_y, height - 50, 20):
                strip = gray[y:y+30, :]
                mean_val = np.mean(strip)
                std_val = np.std(strip)
                
                # Separator: light gray with text variation
                if 180 < mean_val < 240 and 10 < std_val < 60:
                    separators.append({'y': y, 'confidence': 70, 'method': 'visual'})
                    print(f"Visual detection found separator at y={y}")
                    break
        
        return separators
    
    def completely_remove_separator_text(self, image):
        """Use OCR to find and completely remove separator text"""
        try:
            height, width = image.shape[:2]
            
            # Find all separator text regions
            ocr_regions = self.find_separator_text_with_ocr(image)
            
            if not ocr_regions:
                print("No separator text found via OCR")
                return image
            
            # Create mask to remove all separator text areas
            mask = np.ones((height, width), dtype=bool)
            
            for region in ocr_regions:
                x, y, w, h = region['x'], region['y'], region['width'], region['height']
                
                # Expand the region to ensure complete removal
                margin = 20
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(width, x + w + margin)
                y2 = min(height, y + h + margin)
                
                # Mark this region for removal
                mask[y1:y2, x1:x2] = False
                print(f"Removing separator region: ({x1},{y1}) to ({x2},{y2})")
            
            # Also remove any rows that are mostly separator-colored
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            for y in range(height):
                row = gray[y, :]
                mean_val = np.mean(row)
                # If row is mostly light gray (separator color), remove it
                if 200 < mean_val < 250:
                    separator_pixels = np.sum((row > 190) & (row < 250))
                    if separator_pixels > width * 0.5:  # More than 50% is separator color
                        mask[y, :] = False
            
            # Apply mask to keep only non-separator areas
            valid_rows = np.any(mask, axis=1)
            valid_cols = np.any(mask, axis=0)
            
            if np.any(valid_rows) and np.any(valid_cols):
                cleaned_image = image[valid_rows, :][:, valid_cols]
                print(f"Separator text removal: {width}x{height} -> {cleaned_image.shape[1]}x{cleaned_image.shape[0]}")
                return cleaned_image
            else:
                print("Warning: Separator removal would remove entire image")
                return image
            
        except Exception as e:
            print(f"Separator text removal failed: {e}")
            return image
    
    def extract_conservative_full_model(self, image):
        """Extract full model with very conservative approach - avoid extreme cropping"""
        try:
            # First remove separator text completely
            image = self.completely_remove_separator_text(image)
            
            height, width = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            print(f"Extracting conservative full model from {width}x{height} image")
            
            # Use very conservative content detection to avoid cropping model parts
            # Threshold that captures everything except pure white background
            content_mask = gray < 250  # Very lenient threshold
            
            # Clean up tiny noise but preserve model details
            kernel = np.ones((3, 3), np.uint8)  # Small kernel
            content_mask = cv2.morphologyEx(content_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            
            # Find content bounds
            rows_with_content = np.any(content_mask, axis=1)
            cols_with_content = np.any(content_mask, axis=0)
            
            if not np.any(rows_with_content) or not np.any(cols_with_content):
                print("No content detected, returning original")
                return image
            
            row_indices = np.where(rows_with_content)[0]
            col_indices = np.where(cols_with_content)[0]
            
            # Get content bounds
            top = row_indices[0]
            bottom = row_indices[-1]
            left = col_indices[0]
            right = col_indices[-1]
            
            # Apply MINIMAL cropping with MAXIMUM padding to preserve full model
            # Only crop if there's substantial empty space (more than 15% of dimension)
            
            # Vertical cropping - be very conservative
            if top > height * 0.15:  # Only crop top if more than 15% empty
                crop_top = top - 30  # Leave generous margin
            else:
                crop_top = 0
            
            if (height - bottom) > height * 0.15:  # Only crop bottom if more than 15% empty  
                crop_bottom = bottom + 30  # Leave generous margin
            else:
                crop_bottom = height
            
            # Horizontal cropping - be even more conservative to avoid cutting model
            if left > width * 0.2:  # Only crop left if more than 20% empty
                crop_left = left - 50  # Leave very generous margin
            else:
                crop_left = 0
            
            if (width - right) > width * 0.2:  # Only crop right if more than 20% empty
                crop_right = right + 50  # Leave very generous margin  
            else:
                crop_right = width
            
            # Ensure we don't go outside bounds
            crop_top = max(0, crop_top)
            crop_bottom = min(height, crop_bottom)
            crop_left = max(0, crop_left)
            crop_right = min(width, crop_right)
            
            # Extract with conservative bounds
            extracted = image[crop_top:crop_bottom, crop_left:crop_right]
            
            print(f"Conservative extraction: {crop_right-crop_left}x{crop_bottom-crop_top} (cropped {crop_left},{crop_top},{width-crop_right},{height-crop_bottom})")
            
            return extracted
            
        except Exception as e:
            print(f"Conservative extraction failed: {e}")
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
            
            # Detect separators with OCR
            separators = self.detect_separators_with_ocr(image)
            
            if separators:
                separator_y = separators[0]['y']
                method = separators[0].get('method', 'unknown')
                print(f"Using separator at y={separator_y} (method: {method})")
                
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
        """Combine image parts into complete product images with conservative extraction"""
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
                    
                    # Extract with conservative approach
                    processed = self.extract_conservative_full_model(combined)
                    
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
                    
                    print(f"Created conservative product {product_id}: {bottom_part['source_image']} + {matching_top['source_image']}")
                    product_id += 1
                    
                except Exception as e:
                    print(f"Error combining images: {e}")
                    continue
        
        # Handle full images (no separators detected)
        for full_part in fulls:
            try:
                full_img = full_part['image']
                
                # Extract with conservative approach
                processed = self.extract_conservative_full_model(full_img)
                
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
                
                print(f"Created conservative product {product_id}: {full_part['source_image']} (single)")
                product_id += 1
                
            except Exception as e:
                print(f"Error processing full image: {e}")
                continue
        
        return products


def main():
    """Test the OCR-Based Extractor"""
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python ocr_based_extractor.py <input_dir> <output_dir>")
        return
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    extractor = OCRBasedExtractor(output_dir)
    
    print("Loading images...")
    images = extractor.load_images(input_dir)
    if not images:
        print("No images found!")
        return
    
    print(f"Processing {len(images)} images...")
    image_parts = extractor.process_images(images)
    
    print("Combining image parts into conservative products...")
    products = extractor.combine_image_parts(image_parts)
    
    print(f"\nCreated {len(products)} conservative products:")
    for product in products:
        print(f"  {product['filename']}: {product.get('bottom_source', product.get('source', '?'))} + {product.get('top_source', '')}")
    
    print(f"\nOCR-based processing complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()