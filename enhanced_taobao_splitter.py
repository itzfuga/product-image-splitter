#!/usr/bin/env python3
"""
Enhanced Taobao Product Image Splitter
- Better separator detection
- Precise model extraction (white square only)
- Fixed sequencing issues
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


class EnhancedTaobaoSplitter:
    def __init__(self, result_dir, session_id=None):
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(exist_ok=True)
        self.session_id = session_id
        
        # Processing settings
        self.auto_crop = True
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
    
    def detect_visual_separator_enhanced(self, image):
        """Enhanced visual separator detection - more accurate"""
        try:
            height, width = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            separator_positions = []
            
            # Look for horizontal bands with specific characteristics
            # Taobao separators are usually light colored with some text
            
            # Analyze image in horizontal strips
            strip_height = 20
            for y in range(50, height - 50, strip_height):
                if y + strip_height >= height:
                    break
                    
                # Extract horizontal strip
                strip = gray[y:y+strip_height, :]
                strip_mean = np.mean(strip)
                strip_std = np.std(strip)
                
                # Check for separator characteristics:
                # 1. Light colored (high mean value)
                # 2. Some texture variation (not completely uniform like pure white space)
                # 3. Horizontal text patterns
                
                # Separator should be lighter than pure content but not pure white
                if 180 < strip_mean < 240 and 10 < strip_std < 60:
                    # Additional check: look for text-like patterns
                    # Separators often have text on them
                    
                    # Check larger region around this position
                    region_start = max(0, y - 30)
                    region_end = min(height, y + strip_height + 30)
                    region = gray[region_start:region_end, :]
                    
                    # Look for horizontal structures (text)
                    kernel = np.ones((3, 15), np.uint8)  # Horizontal kernel for text detection
                    processed = cv2.morphologyEx(region, cv2.MORPH_CLOSE, kernel)
                    
                    # Count horizontal features
                    edges = cv2.Canny(processed, 50, 150)
                    horizontal_lines = cv2.HoughLinesP(edges, 1, np.pi/180, 
                                                     threshold=width//8, 
                                                     minLineLength=width//4, 
                                                     maxLineGap=20)
                    
                    if horizontal_lines is not None and len(horizontal_lines) > 0:
                        separator_positions.append({
                            'y': y + strip_height // 2,
                            'height': strip_height,
                            'method': 'visual_enhanced',
                            'confidence': min(90, 70 + len(horizontal_lines) * 5),
                            'strip_mean': strip_mean,
                            'strip_std': strip_std
                        })
                        print(f"Found enhanced visual separator at y={y + strip_height // 2} (mean={strip_mean:.1f}, std={strip_std:.1f})")
                        
                        # Don't look for more separators too close to this one
                        y += 100
            
            return separator_positions
            
        except Exception as e:
            print(f"Enhanced visual separator detection error: {e}")
            return []
    
    def extract_model_region(self, image):
        """Extract only the model region (white/light colored rectangular area with the actual model)"""
        try:
            height, width = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            print(f"Extracting model region from {width}x{height} image")
            
            # Look for the actual product image rectangle
            # Taobao product images usually have a distinct white/light rectangular area
            
            # First, find all rectangular regions with light backgrounds
            # Use a more restrictive threshold to find very light areas (white/very light gray)
            _, binary = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Look for rectangular regions that contain actual content (the model)
            best_rect = None
            best_score = 0
            
            for contour in contours:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h
                
                # Skip very small regions
                if w < width * 0.2 or h < height * 0.2:
                    continue
                
                # Skip regions that are too close to edges (likely full image)
                edge_margin = 20
                if x < edge_margin and y < edge_margin and (x + w) > (width - edge_margin) and (y + h) > (height - edge_margin):
                    continue
                
                # Extract this region and analyze it
                region = image[y:y+h, x:x+w]
                region_gray = gray[y:y+h, x:x+w]
                
                # Check if this region contains actual content (model)
                # Look for non-white content within this white region
                content_mask = region_gray < 220  # Non-white pixels
                content_ratio = np.sum(content_mask) / (w * h)
                
                # Good model regions should have:
                # 1. Some content (10-60% non-white pixels)
                # 2. Reasonable aspect ratio
                # 3. Not too close to image edges
                # 4. Reasonable size
                
                aspect_ratio = w / h
                size_score = min(area / (width * height), 1.0)  # Normalize size
                content_score = content_ratio if 0.1 < content_ratio < 0.6 else 0
                aspect_score = 1.0 if 0.4 < aspect_ratio < 2.5 else 0.5
                
                # Calculate total score
                score = content_score * aspect_score * size_score
                
                print(f"Region {x},{y} {w}x{h}: content={content_ratio:.2f}, aspect={aspect_ratio:.2f}, size={size_score:.2f}, score={score:.2f}")
                
                if score > best_score and score > 0.05:  # Minimum threshold
                    best_score = score
                    best_rect = (x, y, w, h)
            
            if best_rect:
                x, y, w, h = best_rect
                print(f"Selected model region: {x},{y} {w}x{h} (score: {best_score:.2f})")
                
                # Add minimal padding
                padding = 5
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(width - x, w + 2 * padding)
                h = min(height - y, h + 2 * padding)
                
                # Extract the region
                extracted = image[y:y+h, x:x+w]
                return extracted
            else:
                print("No suitable model region found, using refined content detection")
                return self.extract_refined_content_region(image)
            
        except Exception as e:
            print(f"Model region extraction failed: {e}")
            return self.extract_refined_content_region(image)
    
    def extract_refined_content_region(self, image):
        """Refined content extraction - tries to find the main product region excluding text"""
        try:
            height, width = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Find non-white content with stricter threshold
            content_mask = gray < 230
            
            # Remove small noise/text elements
            kernel = np.ones((5, 5), np.uint8)
            content_mask = cv2.morphologyEx(content_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            content_mask = cv2.morphologyEx(content_mask, cv2.MORPH_OPEN, kernel)
            
            # Find the largest connected component (should be the model)
            contours, _ = cv2.findContours(content_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get the largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Add padding
                padding = 20
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(width - x, w + 2 * padding)
                h = min(height - y, h + 2 * padding)
                
                extracted = image[y:y+h, x:x+w]
                print(f"Extracted refined content region: {w}x{h}")
                return extracted
            else:
                return self.extract_content_region(image)
            
        except Exception as e:
            print(f"Refined content extraction failed: {e}")
            return self.extract_content_region(image)
    
    def extract_content_region(self, image):
        """Fallback: Extract content region (non-white areas)"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Find non-white content
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
            
            # Add padding
            padding = 20
            min_x = max(0, min_x - padding)
            min_y = max(0, min_y - padding)
            max_x = min(image.shape[1], max_x + padding)
            max_y = min(image.shape[0], max_y + padding)
            
            # Extract the region
            extracted = image[min_y:max_y, min_x:max_x]
            print(f"Extracted content region: {max_x-min_x}x{max_y-min_y}")
            return extracted
            
        except Exception as e:
            print(f"Content region extraction failed: {e}")
            return image
    
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
        """Process images and split them at separators with enhanced detection"""
        image_parts = []
        
        for img_data in images:
            print(f"\nProcessing: {img_data['name']}")
            image = img_data['image']
            
            # Try OCR first (may fail if tesseract not installed)
            text_separators = self.detect_separator_text(image)
            
            # Try enhanced visual detection
            visual_separators = self.detect_visual_separator_enhanced(image)
            
            # Combine and choose best separator
            all_separators = []
            
            # Add text separators with high priority
            for sep in text_separators:
                all_separators.append({
                    'y': sep['y'],
                    'confidence': sep.get('confidence', 80) + 20,  # Boost text detection
                    'method': 'text'
                })
            
            # Add visual separators
            for sep in visual_separators:
                all_separators.append({
                    'y': sep['y'],
                    'confidence': sep.get('confidence', 70),
                    'method': 'visual'
                })
            
            # Choose best separator
            if all_separators:
                # Sort by confidence
                all_separators.sort(key=lambda x: x['confidence'], reverse=True)
                best_separator = all_separators[0]
                separator_y = best_separator['y']
                
                print(f"Using {best_separator['method']} separator at y={separator_y} (confidence: {best_separator['confidence']})")
                
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
        """Combine image parts into complete product images with enhanced processing"""
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
                    
                    # Extract model region only
                    if self.extract_model_only:
                        processed = self.extract_model_region(combined)
                    else:
                        processed = self.extract_content_region(combined)
                    
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
                    
                    print(f"Created product {product_id}: {bottom_part['source_image']} (bottom) + {matching_top['source_image']} (top)")
                    product_id += 1
                    
                except Exception as e:
                    print(f"Error combining images: {e}")
                    continue
        
        # Handle full images (no separators detected)
        for full_part in fulls:
            try:
                full_img = full_part['image']
                
                # Extract model region only
                if self.extract_model_only:
                    processed = self.extract_model_region(full_img)
                else:
                    processed = self.extract_content_region(full_img)
                
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
                
                print(f"Created product {product_id}: {full_part['source_image']} (single image)")
                product_id += 1
                
            except Exception as e:
                print(f"Error processing full image: {e}")
                continue
        
        return products


def main():
    """Test the Enhanced Taobao splitter"""
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python enhanced_taobao_splitter.py <input_dir> <output_dir>")
        return
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    splitter = EnhancedTaobaoSplitter(output_dir)
    
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
    
    info_path = Path(output_dir) / "enhanced_taobao_processing_info.json"
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"\nProcessing complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()