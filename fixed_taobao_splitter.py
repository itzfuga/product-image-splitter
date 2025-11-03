#!/usr/bin/env python3
"""
Fixed Taobao Product Image Splitter
Balanced approach - extracts white rectangles with models, removes gray borders and text
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


class FixedTaobaoSplitter:
    def __init__(self, result_dir, session_id=None):
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(exist_ok=True)
        self.session_id = session_id
        
        # Processing settings
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
    
    def detect_visual_separator_enhanced(self, image):
        """Enhanced visual separator detection"""
        try:
            height, width = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            separator_positions = []
            
            # Look for horizontal bands with separator characteristics
            strip_height = 20
            for y in range(50, height - 100, strip_height):
                if y + strip_height >= height:
                    break
                    
                # Extract horizontal strip
                strip = gray[y:y+strip_height, :]
                strip_mean = np.mean(strip)
                strip_std = np.std(strip)
                
                # Separator characteristics: light gray with some variation (text)
                if 190 < strip_mean < 235 and 10 < strip_std < 50:
                    # Check larger region for text patterns
                    region_size = 60
                    region_start = max(0, y - region_size//2)
                    region_end = min(height, y + region_size//2)
                    region = gray[region_start:region_end, :]
                    
                    # Look for horizontal text structures
                    kernel = np.ones((2, 15), np.uint8)
                    processed = cv2.morphologyEx(region, cv2.MORPH_CLOSE, kernel)
                    edges = cv2.Canny(processed, 30, 100)
                    
                    # Count horizontal features
                    horizontal_kernel = np.ones((1, 20), np.uint8)
                    horizontal_structures = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, horizontal_kernel)
                    horizontal_count = np.sum(horizontal_structures > 0)
                    
                    if horizontal_count > width * 0.08:  # 8% of width has horizontal features
                        separator_positions.append({
                            'y': y,
                            'height': region_size,
                            'method': 'visual_enhanced',
                            'confidence': min(90, 60 + (horizontal_count // (width // 20))),
                            'strip_mean': strip_mean
                        })
                        print(f"Found separator at y={y} (mean={strip_mean:.1f})")
                        
                        # Skip ahead to avoid duplicates
                        y += 80
            
            return separator_positions
            
        except Exception as e:
            print(f"Enhanced visual separator detection error: {e}")
            return []
    
    def extract_white_model_rectangle(self, image):
        """Extract the white rectangle containing the model - balanced approach"""
        try:
            height, width = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            print(f"Extracting white model rectangle from {width}x{height} image")
            
            # Method 1: Find light regions that could contain model
            # Use moderate threshold - not too strict
            light_mask = gray > 220  # Light pixels
            
            # Clean up the mask
            kernel = np.ones((8, 8), np.uint8)
            light_mask_cleaned = cv2.morphologyEx(light_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            
            # Find contours of light regions
            contours, _ = cv2.findContours(light_mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                print("No light regions found, using smart content detection")
                return self.smart_content_extraction(image)
            
            # Analyze each region to find the best model area
            best_rect = None
            best_score = 0
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h
                
                # Skip very small regions
                if w < width * 0.1 or h < height * 0.1:
                    continue
                
                # Skip regions that are essentially the full image (with small margins)
                margin = 50
                if (x < margin and y < margin and 
                    (x + w) > (width - margin) and (y + h) > (height - margin)):
                    continue
                
                # Extract and analyze this region
                region = image[y:y+h, x:x+w]
                region_gray = gray[y:y+h, x:x+w]
                
                # Check content characteristics
                # Good model regions have some non-white content but lots of white background
                content_mask = region_gray < 200  # Non-white content
                content_ratio = np.sum(content_mask) / (w * h)
                
                # Check background lightness
                background_mask = region_gray >= 220
                background_ratio = np.sum(background_mask) / (w * h)
                
                # Calculate average color
                mean_color = np.mean(region_gray)
                
                # Scoring criteria (more lenient than ultra-precise)
                # 1. Should have some content (5-60% non-white)
                # 2. Should have light background (40%+ light pixels)
                # 3. Should be reasonably sized
                # 4. Should have good aspect ratio
                # 5. Should be reasonably light overall
                
                content_score = 1.0 if 0.05 < content_ratio < 0.6 else 0.2
                background_score = 1.0 if background_ratio > 0.4 else 0.3
                size_score = min((area / (width * height)) * 2, 1.0)
                aspect_ratio = w / h
                aspect_score = 1.0 if 0.3 < aspect_ratio < 3.0 else 0.5
                lightness_score = 1.0 if mean_color > 200 else 0.7
                
                # Boost score for regions that aren't at image edges
                edge_margin = 30
                edge_penalty = 1.0
                if (x < edge_margin or y < edge_margin or 
                    (x + w) > (width - edge_margin) or (y + h) > (height - edge_margin)):
                    edge_penalty = 0.7
                
                total_score = (content_score * background_score * size_score * 
                              aspect_score * lightness_score * edge_penalty)
                
                print(f"Region {x},{y} {w}x{h}: content={content_ratio:.2f}, bg={background_ratio:.2f}, "
                      f"size={size_score:.2f}, aspect={aspect_ratio:.2f}, light={mean_color:.1f}, score={total_score:.3f}")
                
                if total_score > best_score and total_score > 0.05:
                    best_score = total_score
                    best_rect = (x, y, w, h)
            
            if best_rect:
                x, y, w, h = best_rect
                print(f"Selected white rectangle: {x},{y} {w}x{h} (score: {best_score:.3f})")
                
                # Extract with small padding to ensure we get the full model
                padding = 8
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(width - x, w + 2 * padding)
                h = min(height - y, h + 2 * padding)
                
                extracted = image[y:y+h, x:x+w]
                
                # Final cleanup - remove any gray edges
                cleaned = self.clean_gray_edges(extracted)
                
                return cleaned
            else:
                print("No suitable white rectangle found, using smart content detection")
                return self.smart_content_extraction(image)
            
        except Exception as e:
            print(f"White rectangle extraction failed: {e}")
            return self.smart_content_extraction(image)
    
    def clean_gray_edges(self, image):
        """Remove gray edges from the extracted image"""
        try:
            height, width = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Find content bounds (anything that's not very light)
            content_mask = gray < 210  # Content pixels (not very light)
            
            # Also keep pure white pixels (part of the white background)
            white_mask = gray > 245
            
            # Combine: keep content OR pure white
            keep_mask = content_mask | white_mask
            
            # Find bounding box of what to keep
            rows_to_keep = np.any(keep_mask, axis=1)
            cols_to_keep = np.any(keep_mask, axis=0)
            
            row_indices = np.where(rows_to_keep)[0]
            col_indices = np.where(cols_to_keep)[0]
            
            if len(row_indices) > 0 and len(col_indices) > 0:
                min_row, max_row = row_indices[0], row_indices[-1]
                min_col, max_col = col_indices[0], col_indices[-1]
                
                # Add small padding
                padding = 3
                min_row = max(0, min_row - padding)
                min_col = max(0, min_col - padding)
                max_row = min(height - 1, max_row + padding)
                max_col = min(width - 1, max_col + padding)
                
                cleaned = image[min_row:max_row+1, min_col:max_col+1]
                print(f"Cleaned edges: {max_col-min_col+1}x{max_row-min_row+1}")
                return cleaned
            else:
                return image
                
        except Exception as e:
            print(f"Gray edge cleaning failed: {e}")
            return image
    
    def smart_content_extraction(self, image):
        """Smart content extraction as fallback"""
        try:
            height, width = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Find content that's not pure white or light gray background
            content_mask = (gray < 200) | (gray > 250)  # Content OR pure white
            
            # Clean up noise
            kernel = np.ones((3, 3), np.uint8)
            content_mask = cv2.morphologyEx(content_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            
            # Find largest connected component
            contours, _ = cv2.findContours(content_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Add padding
                padding = 15
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(width - x, w + 2 * padding)
                h = min(height - y, h + 2 * padding)
                
                extracted = image[y:y+h, x:x+w]
                print(f"Smart content extraction: {w}x{h}")
                return extracted
            else:
                # Final fallback - basic content bounds
                content_simple = gray < 220
                rows = np.any(content_simple, axis=1)
                cols = np.any(content_simple, axis=0)
                
                if np.any(rows) and np.any(cols):
                    row_indices = np.where(rows)[0]
                    col_indices = np.where(cols)[0]
                    
                    min_y, max_y = row_indices[0], row_indices[-1]
                    min_x, max_x = col_indices[0], col_indices[-1]
                    
                    padding = 10
                    min_x = max(0, min_x - padding)
                    min_y = max(0, min_y - padding)
                    max_x = min(width, max_x + padding)
                    max_y = min(height, max_y + padding)
                    
                    extracted = image[min_y:max_y, min_x:max_x]
                    print(f"Basic content extraction: {max_x-min_x}x{max_y-min_y}")
                    return extracted
                else:
                    return image
                    
        except Exception as e:
            print(f"Smart content extraction failed: {e}")
            return image
    
    def split_image_at_separator(self, image, separator_y):
        """Split image into two parts at separator position"""
        height = image.shape[0]
        
        # Split into top and bottom parts
        top_part = image[0:separator_y, :]
        bottom_part = image[separator_y:height, :]
        
        return top_part, bottom_part
    
    def process_images(self, images):
        """Process images and split them at separators"""
        image_parts = []
        
        for img_data in images:
            print(f"\nProcessing: {img_data['name']}")
            image = img_data['image']
            
            # Try OCR first (may fail if tesseract not installed)
            try:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
                
                text_separators = []
                all_text = ' '.join([text for text in data['text'] if text.strip()])
                text_upper = all_text.upper().replace(' ', '').replace('\n', '')
                
                if 'START' in text_upper and 'EXCEED' in text_upper and 'END' in text_upper:
                    for i, text in enumerate(data['text']):
                        if any(word in text.upper() for word in ['START', 'EXCEED', 'END']):
                            y = data['top'][i] + data['height'][i] // 2
                            text_separators.append({'y': y, 'confidence': 85, 'method': 'ocr'})
                            print(f"Found OCR separator at y={y}")
                            break
            except:
                text_separators = []
            
            # Try enhanced visual detection
            visual_separators = self.detect_visual_separator_enhanced(image)
            
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
        """Combine image parts into complete product images with balanced extraction"""
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
                    
                    # Extract white model rectangle with balanced approach
                    processed = self.extract_white_model_rectangle(combined)
                    
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
                    
                    print(f"Created fixed product {product_id}: {bottom_part['source_image']} + {matching_top['source_image']}")
                    product_id += 1
                    
                except Exception as e:
                    print(f"Error combining images: {e}")
                    continue
        
        # Handle full images (no separators detected)
        for full_part in fulls:
            try:
                full_img = full_part['image']
                
                # Extract white model rectangle
                processed = self.extract_white_model_rectangle(full_img)
                
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
                
                print(f"Created fixed product {product_id}: {full_part['source_image']} (single)")
                product_id += 1
                
            except Exception as e:
                print(f"Error processing full image: {e}")
                continue
        
        return products


def main():
    """Test the Fixed Taobao splitter"""
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python fixed_taobao_splitter.py <input_dir> <output_dir>")
        return
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    splitter = FixedTaobaoSplitter(output_dir)
    
    print("Loading images...")
    images = splitter.load_images(input_dir)
    if not images:
        print("No images found!")
        return
    
    print(f"Processing {len(images)} images...")
    image_parts = splitter.process_images(images)
    
    print("Combining image parts into fixed products...")
    products = splitter.combine_image_parts(image_parts)
    
    print(f"\nCreated {len(products)} fixed products:")
    for product in products:
        print(f"  {product['filename']}: {product.get('bottom_source', product.get('source', '?'))} + {product.get('top_source', '')}")
    
    # Save processing info
    info = {
        'total_images': len(images),
        'total_products': len(products),
        'products': products
    }
    
    info_path = Path(output_dir) / "fixed_processing_info.json"
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"\nFixed processing complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()