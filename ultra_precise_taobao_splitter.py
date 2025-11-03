#!/usr/bin/env python3
"""
Ultra-Precise Taobao Product Image Splitter
Extracts ONLY the pure white rectangle with model - no gray borders, no text
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


class UltraPreciseTaobaoSplitter:
    def __init__(self, result_dir, session_id=None):
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(exist_ok=True)
        self.session_id = session_id
        
        # Processing settings - ultra precise
        self.extract_model_only = True
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
    
    def detect_visual_separator_ultra(self, image):
        """Ultra-precise visual separator detection"""
        try:
            height, width = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            separator_positions = []
            
            # Look for the actual "START EXCEED END" text region
            # This appears as a light gray band with dark text
            
            strip_height = 15
            for y in range(50, height - 100, strip_height):
                if y + strip_height >= height:
                    break
                    
                # Extract horizontal strip
                strip = gray[y:y+strip_height, :]
                strip_mean = np.mean(strip)
                strip_std = np.std(strip)
                
                # Look for characteristics of separator text area:
                # 1. Light gray background (not white, not dark)
                # 2. Some variation (text creates contrast)
                # 3. Horizontal text patterns
                
                if 200 < strip_mean < 230 and 15 < strip_std < 45:
                    # Check larger region for text patterns
                    region_size = 80
                    region_start = max(0, y - region_size//2)
                    region_end = min(height, y + region_size//2)
                    region = gray[region_start:region_end, :]
                    
                    # Look for horizontal text-like structures
                    # Apply horizontal morphological operations to detect text
                    kernel = np.ones((2, 20), np.uint8)  # Horizontal kernel for text
                    processed = cv2.morphologyEx(region, cv2.MORPH_CLOSE, kernel)
                    
                    # Find horizontal edges (typical of text)
                    edges = cv2.Canny(processed, 30, 100)
                    
                    # Count horizontal structures
                    horizontal_kernel = np.ones((1, 30), np.uint8)
                    horizontal_structures = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, horizontal_kernel)
                    horizontal_count = np.sum(horizontal_structures > 0)
                    
                    # If we find significant horizontal structures, this is likely a separator
                    if horizontal_count > width * 0.1:  # At least 10% of width has horizontal features
                        separator_positions.append({
                            'y': y,
                            'height': region_size,
                            'method': 'ultra_visual',
                            'confidence': min(95, 70 + (horizontal_count // (width // 10))),
                            'strip_mean': strip_mean,
                            'horizontal_count': horizontal_count
                        })
                        print(f"Found ultra separator at y={y} (mean={strip_mean:.1f}, h_count={horizontal_count})")
                        
                        # Skip ahead to avoid duplicates
                        y += 100
            
            return separator_positions
            
        except Exception as e:
            print(f"Ultra visual separator detection error: {e}")
            return []
    
    def extract_pure_white_rectangle(self, image):
        """Extract ONLY the pure white rectangle containing the model - ultra precise"""
        try:
            height, width = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            print(f"Ultra-precise extraction from {width}x{height} image")
            
            # Step 1: Find pure white regions (very restrictive threshold)
            # Only consider pixels that are very close to pure white
            white_mask = gray > 245  # Very white pixels only
            
            # Step 2: Find connected white regions
            # Use morphological operations to connect nearby white areas
            kernel = np.ones((10, 10), np.uint8)
            white_mask_cleaned = cv2.morphologyEx(white_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            
            # Step 3: Find contours of white regions
            contours, _ = cv2.findContours(white_mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Step 4: Analyze each white region to find the model area
            best_rect = None
            best_score = 0
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h
                
                # Skip tiny regions
                if w < width * 0.15 or h < height * 0.15:
                    continue
                
                # Skip regions that touch image edges (likely full background)
                edge_margin = 30
                if (x < edge_margin or y < edge_margin or 
                    (x + w) > (width - edge_margin) or (y + h) > (height - edge_margin)):
                    continue
                
                # Extract this white region
                region = image[y:y+h, x:x+w]
                region_gray = gray[y:y+h, x:x+w]
                
                # Analyze content within this white region
                # Look for the model (non-white content)
                content_mask = region_gray < 240  # Non-white content
                content_pixels = np.sum(content_mask)
                content_ratio = content_pixels / (w * h)
                
                # Check background purity (should be very white)
                background_mask = region_gray >= 240
                background_pixels = np.sum(background_mask)
                background_purity = background_pixels / (w * h)
                
                # Analyze color distribution in RGB
                region_rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
                mean_rgb = np.mean(region_rgb, axis=(0, 1))
                
                # Good white rectangles should have:
                # 1. High background purity (lots of white/very light pixels)
                # 2. Some model content (10-50% non-white)
                # 3. Overall light coloring
                # 4. Reasonable aspect ratio
                # 5. Not too close to edges
                
                aspect_ratio = w / h
                size_ratio = area / (width * height)
                lightness_score = 1.0 if np.mean(mean_rgb) > 230 else 0.5
                
                # Scoring system
                content_score = 1.0 if 0.1 < content_ratio < 0.5 else 0.3
                purity_score = 1.0 if background_purity > 0.6 else 0.3
                aspect_score = 1.0 if 0.5 < aspect_ratio < 2.0 else 0.5
                size_score = min(size_ratio * 3, 1.0)  # Prefer larger regions
                
                total_score = content_score * purity_score * aspect_score * size_score * lightness_score
                
                print(f"White region {x},{y} {w}x{h}: content={content_ratio:.2f}, purity={background_purity:.2f}, "
                      f"aspect={aspect_ratio:.2f}, light={np.mean(mean_rgb):.1f}, score={total_score:.3f}")
                
                if total_score > best_score and total_score > 0.1:
                    best_score = total_score
                    best_rect = (x, y, w, h)
            
            if best_rect:
                x, y, w, h = best_rect
                print(f"Selected pure white rectangle: {x},{y} {w}x{h} (score: {best_score:.3f})")
                
                # Extract with minimal padding to stay within white area
                padding = 2
                x = max(0, x + padding)
                y = max(0, y + padding)
                w = max(1, w - 2 * padding)
                h = max(1, h - 2 * padding)
                
                # Final extraction
                extracted = image[y:y+h, x:x+w]
                
                # Additional cleanup: remove any remaining gray edges
                extracted = self.ultra_trim_gray_edges(extracted)
                
                return extracted
            else:
                print("No pure white rectangle found, using fallback")
                return self.fallback_model_extraction(image)
            
        except Exception as e:
            print(f"Pure white rectangle extraction failed: {e}")
            return self.fallback_model_extraction(image)
    
    def ultra_trim_gray_edges(self, image):
        """Remove any remaining gray edges from the extracted image"""
        try:
            height, width = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Find the bounds of the pure content (very strict)
            # Only keep pixels that are either very white (background) or dark enough (model)
            keep_mask = (gray > 240) | (gray < 200)  # Very white OR dark enough to be model
            
            # Find bounding box of content to keep
            rows_to_keep = np.any(keep_mask, axis=1)
            cols_to_keep = np.any(keep_mask, axis=0)
            
            row_indices = np.where(rows_to_keep)[0]
            col_indices = np.where(cols_to_keep)[0]
            
            if len(row_indices) > 0 and len(col_indices) > 0:
                min_row, max_row = row_indices[0], row_indices[-1]
                min_col, max_col = col_indices[0], col_indices[-1]
                
                # Add small padding
                padding = 5
                min_row = max(0, min_row - padding)
                min_col = max(0, min_col - padding)
                max_row = min(height - 1, max_row + padding)
                max_col = min(width - 1, max_col + padding)
                
                trimmed = image[min_row:max_row+1, min_col:max_col+1]
                print(f"Ultra-trimmed to: {max_col-min_col+1}x{max_row-min_row+1}")
                return trimmed
            else:
                return image
                
        except Exception as e:
            print(f"Ultra trim failed: {e}")
            return image
    
    def fallback_model_extraction(self, image):
        """Fallback extraction method"""
        try:
            height, width = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Find content (non-white areas)
            content_mask = gray < 230
            
            # Find bounds
            content_rows = np.any(content_mask, axis=1)
            content_cols = np.any(content_mask, axis=0)
            
            content_row_indices = np.where(content_rows)[0]
            content_col_indices = np.where(content_cols)[0]
            
            if len(content_row_indices) == 0 or len(content_col_indices) == 0:
                return image
            
            min_y = content_row_indices[0]
            max_y = content_row_indices[-1]
            min_x = content_col_indices[0]
            max_x = content_col_indices[-1]
            
            # Add minimal padding
            padding = 10
            min_x = max(0, min_x - padding)
            min_y = max(0, min_y - padding)
            max_x = min(width, max_x + padding)
            max_y = min(height, max_y + padding)
            
            extracted = image[min_y:max_y, min_x:max_x]
            print(f"Fallback extraction: {max_x-min_x}x{max_y-min_y}")
            return extracted
            
        except Exception as e:
            print(f"Fallback extraction failed: {e}")
            return image
    
    def split_image_at_separator(self, image, separator_y):
        """Split image into two parts at separator position"""
        height = image.shape[0]
        
        # Split into top and bottom parts
        top_part = image[0:separator_y, :]
        bottom_part = image[separator_y:height, :]
        
        return top_part, bottom_part
    
    def process_images(self, images):
        """Process images and split them at separators with ultra detection"""
        image_parts = []
        
        for img_data in images:
            print(f"\nProcessing: {img_data['name']}")
            image = img_data['image']
            
            # Try OCR first (may fail if tesseract not installed)
            try:
                # Convert to PIL for OCR
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
                
                text_separators = []
                all_text = ' '.join([text for text in data['text'] if text.strip()])
                text_upper = all_text.upper().replace(' ', '').replace('\n', '')
                
                if 'START' in text_upper and 'EXCEED' in text_upper and 'END' in text_upper:
                    # Find text position
                    for i, text in enumerate(data['text']):
                        if any(word in text.upper() for word in ['START', 'EXCEED', 'END']):
                            y = data['top'][i] + data['height'][i] // 2
                            text_separators.append({'y': y, 'confidence': 90, 'method': 'ocr'})
                            print(f"Found OCR separator at y={y}")
                            break
            except:
                text_separators = []
            
            # Try ultra visual detection
            visual_separators = self.detect_visual_separator_ultra(image)
            
            # Choose best separator
            all_separators = text_separators + visual_separators
            
            if all_separators:
                # Sort by confidence
                all_separators.sort(key=lambda x: x.get('confidence', 0), reverse=True)
                best_separator = all_separators[0]
                separator_y = best_separator['y']
                
                print(f"Using {best_separator.get('method', 'unknown')} separator at y={separator_y}")
                
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
        """Combine image parts into complete product images with ultra-precise extraction"""
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
                    
                    # Ultra-precise model extraction - ONLY white rectangle with model
                    processed = self.extract_pure_white_rectangle(combined)
                    
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
                    
                    print(f"Created ultra-precise product {product_id}: {bottom_part['source_image']} + {matching_top['source_image']}")
                    product_id += 1
                    
                except Exception as e:
                    print(f"Error combining images: {e}")
                    continue
        
        # Handle full images (no separators detected)
        for full_part in fulls:
            try:
                full_img = full_part['image']
                
                # Ultra-precise model extraction
                processed = self.extract_pure_white_rectangle(full_img)
                
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
                
                print(f"Created ultra-precise product {product_id}: {full_part['source_image']} (single)")
                product_id += 1
                
            except Exception as e:
                print(f"Error processing full image: {e}")
                continue
        
        return products


def main():
    """Test the Ultra-Precise Taobao splitter"""
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python ultra_precise_taobao_splitter.py <input_dir> <output_dir>")
        return
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    splitter = UltraPreciseTaobaoSplitter(output_dir)
    
    print("Loading images...")
    images = splitter.load_images(input_dir)
    if not images:
        print("No images found!")
        return
    
    print(f"Processing {len(images)} images...")
    image_parts = splitter.process_images(images)
    
    print("Combining image parts into ultra-precise products...")
    products = splitter.combine_image_parts(image_parts)
    
    print(f"\nCreated {len(products)} ultra-precise products:")
    for product in products:
        print(f"  {product['filename']}: {product.get('bottom_source', product.get('source', '?'))} + {product.get('top_source', '')}")
    
    # Save processing info
    info = {
        'total_images': len(images),
        'total_products': len(products),
        'products': products
    }
    
    info_path = Path(output_dir) / "ultra_precise_processing_info.json"
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"\nUltra-precise processing complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()