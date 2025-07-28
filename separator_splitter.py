#!/usr/bin/env python3
"""
Product Image Separator Splitter
Detects separator lines in e-commerce images and splits them into product segments
"""

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageDraw, ImageFont
import os
from pathlib import Path
import json
import re


class SeparatorSplitter:
    def __init__(self, result_dir):
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(exist_ok=True)
        
        # Configure Tesseract (adjust path if needed)
        # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        
    def load_images(self, image_dir):
        """Load all images from directory"""
        image_dir = Path(image_dir)
        images = []
        
        supported_formats = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.avif']
        
        # Natural sort to handle numeric filenames properly (1, 2, 3... not 1, 10, 11...)
        import re
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
                            'cv2_image': img,
                            'pil_image': None
                        })
                        print(f"Loaded: {file_path.name}")
                    else:
                        # Fallback to PIL for formats like AVIF
                        pil_img = Image.open(file_path).convert('RGB')
                        cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                        images.append({
                            'path': str(file_path),
                            'name': file_path.name,
                            'cv2_image': cv2_img,
                            'pil_image': pil_img
                        })
                        print(f"Loaded (PIL): {file_path.name}")
                        
                except Exception as e:
                    print(f"Error loading {file_path.name}: {e}")
                    
        return images
    
    def detect_text_separators(self, image):
        """Detect separator text using OCR"""
        try:
            # Convert to PIL if needed
            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = image
            
            # Get text with bounding boxes
            try:
                data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
            except Exception as ocr_error:
                print(f"OCR failed (continuing without text detection): {ocr_error}")
                return []
            
            separator_positions = []
            separator_patterns = [
                r'STAFF.*START.*EXCEED.*END',
                r'START.*EXCEED.*END',
                r'STAFF.*EXCEED',
                r'EXCEED.*END',
                # Add more patterns as needed
            ]
            
            for i, text in enumerate(data['text']):
                if text.strip():
                    # Check if text matches separator patterns
                    text_upper = text.upper().replace(' ', '').replace('\n', '')
                    for pattern in separator_patterns:
                        if re.search(pattern, text_upper):
                            y = data['top'][i]
                            height = data['height'][i]
                            separator_positions.append({
                                'y': y,
                                'height': height,
                                'text': text,
                                'confidence': data['conf'][i]
                            })
                            print(f"Found separator text: {text} at y={y}")
            
            return separator_positions
            
        except Exception as e:
            print(f"OCR error: {e}")
            return []
    
    def detect_visual_separators(self, image):
        """Detect separator areas using visual analysis for fashion images"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            
            print(f"Analyzing image: {width}x{height}")
            
            # Focus on whitespace detection for fashion images
            white_separators = self.detect_whitespace_separators(gray)
            
            # Additional method: Look for horizontal transitions from content to background
            separator_positions = []
            
            # Analyze vertical gradient changes
            # Look for areas where there's a sudden change from content to background
            kernel = np.ones((10, width), np.uint8)  # Horizontal kernel
            processed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
            
            # Find horizontal edges that might indicate content boundaries
            edges = cv2.Canny(processed, 100, 200)
            
            # Look for horizontal lines that span significant width
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=width//3, 
                                   minLineLength=width//2, maxLineGap=50)
            
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    # Only consider nearly horizontal lines
                    if abs(y2 - y1) < 10:
                        y_pos = (y1 + y2) // 2
                        separator_positions.append({
                            'y': y_pos,
                            'height': 5,
                            'method': 'visual_line',
                            'confidence': 70
                        })
                        print(f"Found visual line separator at y={y_pos}")
            
            # Combine with whitespace separators
            separator_positions.extend(white_separators)
            
            return separator_positions
            
        except Exception as e:
            print(f"Visual separator detection error: {e}")
            return []
    
    def detect_whitespace_separators(self, gray_image):
        """Detect large whitespace areas that could be separators for fashion images"""
        height, width = gray_image.shape
        separators = []
        
        # Analyze each row for whiteness/background
        row_scores = []
        for y in range(height):
            row = gray_image[y, :]
            
            # Calculate percentage of very light pixels (white/light gray background)
            very_light_pixels = np.sum(row > 240)  # Very light pixels (nearly white)
            light_pixels = np.sum(row > 220)      # Light pixels
            
            # Also check for low variance (uniform background)
            row_variance = np.var(row)
            
            # Background score - higher means more likely to be background
            # Prioritize very white pixels and low variance
            whiteness_score = very_light_pixels / width
            lightness_score = light_pixels / width
            uniformity_score = 1.0 - min(row_variance / 1000, 1.0)  # Lower variance = higher score
            
            # Combined score with emphasis on whiteness and uniformity
            background_score = (whiteness_score * 0.5 + lightness_score * 0.3 + uniformity_score * 0.2)
            row_scores.append(background_score)
        
        # Smooth the scores to reduce noise
        smoothed_scores = []
        window_size = 5
        for i in range(len(row_scores)):
            start = max(0, i - window_size // 2)
            end = min(len(row_scores), i + window_size // 2 + 1)
            smoothed_scores.append(sum(row_scores[start:end]) / (end - start))
        
        # Find ALL continuous background regions
        in_background_region = False
        background_region_start = 0
        min_background_height = 50  # At least 50 pixels high
        background_threshold = 0.8  # 80% background pixels
        
        print(f"Image height: {height}, min_background_height: {min_background_height}")
        
        for y, score in enumerate(smoothed_scores):
            if score > background_threshold:
                if not in_background_region:
                    background_region_start = y
                    in_background_region = True
            else:
                if in_background_region:
                    background_region_height = y - background_region_start
                    if background_region_height >= min_background_height:
                        # Calculate average score for this region
                        region_avg_score = sum(smoothed_scores[background_region_start:y]) / background_region_height
                        
                        separator_y = background_region_start + background_region_height // 2
                        
                        # Add this separator
                        separators.append({
                            'y': separator_y,
                            'height': background_region_height,
                            'method': 'whitespace',
                            'confidence': min(region_avg_score * 100, 95)
                        })
                        
                        print(f"Found white separator at y={separator_y}, height={background_region_height}px")
                    in_background_region = False
        
        # Check if we ended in a background region
        if in_background_region:
            background_region_height = len(smoothed_scores) - background_region_start
            if background_region_height >= min_background_height:
                separator_y = background_region_start + background_region_height // 2
                separators.append({
                    'y': separator_y,
                    'height': background_region_height,
                    'method': 'whitespace',
                    'confidence': 90
                })
                print(f"Found background separator at end y={separator_y}, height={background_region_height}")
        
        return separators
    
    def combine_separator_detections(self, text_separators, visual_separators, image_height):
        """Combine and filter separator detections for fashion images"""
        all_separators = []
        
        # For fashion images, prioritize whitespace detection over text
        # Add visual separators (whitespace has higher priority for fashion images)
        for sep in visual_separators:
            confidence = sep.get('confidence', 80)
            # Boost confidence for whitespace detection
            if sep.get('method') == 'whitespace':
                confidence = min(confidence + 20, 95)
            
            all_separators.append({
                'y': sep['y'],
                'confidence': confidence,
                'method': sep.get('method', 'visual'),
                'height': sep.get('height', 10)
            })
        
        # Add text-based separators (lower priority for fashion images)
        for sep in text_separators:
            all_separators.append({
                'y': sep['y'] + sep['height'] // 2,
                'confidence': sep.get('confidence', 60),  # Lower confidence
                'method': 'text',
                'text': sep.get('text', ''),
                'height': sep.get('height', 10)
            })
        
        # Remove separators too close to edges (more conservative)
        edge_margin = image_height * 0.1  # 10% margin from top/bottom
        filtered_separators = [
            sep for sep in all_separators 
            if edge_margin < sep['y'] < image_height - edge_margin
        ]
        
        # Sort by confidence first, then by y position
        filtered_separators.sort(key=lambda x: (-x['confidence'], x['y']))
        
        # Remove duplicates (separators too close to each other)
        final_separators = []
        min_distance = image_height * 0.15  # 15% of image height minimum distance
        
        for sep in filtered_separators:
            # Check if this separator is too close to any existing one
            too_close = False
            for existing in final_separators:
                if abs(sep['y'] - existing['y']) < min_distance:
                    too_close = True
                    # Keep the one with higher confidence
                    if sep['confidence'] > existing['confidence']:
                        final_separators.remove(existing)
                        too_close = False
                    break
            
            if not too_close:
                final_separators.append(sep)
        
        # Sort final separators by y position
        final_separators.sort(key=lambda x: x['y'])
        
        # For fashion images, we might have multiple separators (e.g., 3 products per image)
        # Don't limit to just 1 separator
        print(f"Found {len(final_separators)} separators in image")
        
        return final_separators
    
    def split_image_at_separators(self, image, separators):
        """Split image into segments at separator positions"""
        height, width = image.shape[:2]
        segments = []
        
        # Create split points (including image boundaries)
        split_points = [0]  # Start of image
        for sep in separators:
            split_points.append(sep['y'])
        split_points.append(height)  # End of image
        
        # Create segments
        for i in range(len(split_points) - 1):
            start_y = split_points[i]
            end_y = split_points[i + 1]
            
            # Skip very small segments
            if end_y - start_y < 50:
                continue
            
            segment = image[start_y:end_y, :]
            segments.append({
                'image': segment,
                'start_y': start_y,
                'end_y': end_y,
                'height': end_y - start_y
            })
        
        return segments
    
    def process_images(self, images):
        """Process all images and split them at separators"""
        all_segments = []
        
        for i, img_data in enumerate(images):
            print(f"\nProcessing image {i+1}/{len(images)}: {img_data['name']}")
            
            image = img_data['cv2_image']
            height, width = image.shape[:2]
            
            # Detect separators
            print("Detecting text separators...")
            text_separators = self.detect_text_separators(image)
            
            print("Detecting visual separators...")
            visual_separators = self.detect_visual_separators(image)
            
            print(f"Found {len(text_separators)} text separators, {len(visual_separators)} visual separators")
            
            # Combine detections
            separators = self.combine_separator_detections(text_separators, visual_separators, height)
            print(f"Final separators: {len(separators)}")
            
            # Split image
            segments = self.split_image_at_separators(image, separators)
            print(f"Created {len(segments)} segments")
            
            # Add metadata to segments
            for j, segment in enumerate(segments):
                segment['source_image'] = img_data['name']
                segment['source_index'] = i
                segment['segment_index'] = j
                all_segments.append(segment)
        
        return all_segments
    
    def create_products_from_segments(self, segments):
        """Create product images by combining segments from different source images"""
        products = []
        product_id = 1
        
        # For Taobao-style images: combine bottom of image N with top of image N+1
        all_segments = sorted(segments, key=lambda x: (x['source_index'], x['segment_index']))
        
        print(f"Total segments: {len(all_segments)}")
        
        # Create products by pairing segments
        for i in range(len(all_segments) - 1):
            current_segment = all_segments[i]
            next_segment = all_segments[i + 1]
            
            # Skip if both segments are from the same image
            if current_segment['source_index'] == next_segment['source_index']:
                continue
                
            print(f"Pairing: {current_segment['source_image']} seg{current_segment['segment_index']} + {next_segment['source_image']} seg{next_segment['segment_index']}")
            
            # Create product by combining current (bottom) with next (top)
            product = self.combine_segments(current_segment, next_segment, product_id)
            if product:
                products.append(product)
                product_id += 1
        
        return products
    
    def create_single_segment_product(self, segment, product_id):
        """Create a product from a single segment"""
        try:
            segment_img = segment['image']
            
            # Save product image
            filename = f"product_{product_id}.png"
            output_path = self.result_dir / filename
            cv2.imwrite(str(output_path), segment_img)
            
            height, width = segment_img.shape[:2]
            
            return {
                'product_id': product_id,
                'filename': filename,
                'path': str(output_path),
                'single_segment': {
                    'source': segment['source_image'],
                    'segment_index': segment['segment_index']
                },
                'dimensions': {
                    'width': width,
                    'height': height
                }
            }
            
        except Exception as e:
            print(f"Error creating single segment product {product_id}: {e}")
            return None
    
    def combine_segments(self, bottom_segment, top_segment, product_id):
        """Combine two segments into a product image"""
        try:
            bottom_img = bottom_segment['image']
            top_img = top_segment['image']
            
            # Ensure both images have the same width
            height1, width1 = bottom_img.shape[:2]
            height2, width2 = top_img.shape[:2]
            
            target_width = max(width1, width2)
            
            # Resize if needed
            if width1 != target_width:
                bottom_img = cv2.resize(bottom_img, (target_width, height1))
            if width2 != target_width:
                top_img = cv2.resize(top_img, (target_width, height2))
            
            # Combine vertically
            combined = np.vstack([bottom_img, top_img])
            
            # Save product image
            filename = f"product_{product_id}.png"
            output_path = self.result_dir / filename
            cv2.imwrite(str(output_path), combined)
            
            return {
                'product_id': product_id,
                'filename': filename,
                'path': str(output_path),
                'bottom_segment': {
                    'source': bottom_segment['source_image'],
                    'segment_index': bottom_segment['segment_index']
                },
                'top_segment': {
                    'source': top_segment['source_image'],
                    'segment_index': top_segment['segment_index']
                },
                'dimensions': {
                    'width': target_width,
                    'height': height1 + height2
                }
            }
            
        except Exception as e:
            print(f"Error combining segments for product {product_id}: {e}")
            return None
    
    def create_debug_visualization(self, images, all_segments):
        """Create visualization showing detected separators and segments"""
        try:
            for i, img_data in enumerate(images):
                image = img_data['cv2_image'].copy()
                height, width = image.shape[:2]
                
                # Get segments for this image
                image_segments = [seg for seg in all_segments if seg['source_index'] == i]
                
                # Draw separator lines
                for j, segment in enumerate(image_segments[:-1]):  # Exclude last segment
                    y = segment['end_y']
                    cv2.line(image, (0, y), (width, y), (0, 255, 0), 3)
                    cv2.putText(image, f"SEP {j+1}", (10, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Draw segment labels
                for j, segment in enumerate(image_segments):
                    center_y = (segment['start_y'] + segment['end_y']) // 2
                    cv2.putText(image, f"SEG {j+1}", (width-100, center_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                
                # Save debug image
                debug_filename = f"debug_{img_data['name']}"
                debug_path = self.result_dir / debug_filename
                cv2.imwrite(str(debug_path), image)
                print(f"Saved debug visualization: {debug_filename}")
                
        except Exception as e:
            print(f"Error creating debug visualization: {e}")


def main():
    """Test the separator splitter"""
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python separator_splitter.py <input_dir> <output_dir>")
        return
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    splitter = SeparatorSplitter(output_dir)
    
    print("Loading images...")
    images = splitter.load_images(input_dir)
    if not images:
        print("No images found!")
        return
    
    print(f"Processing {len(images)} images...")
    segments = splitter.process_images(images)
    
    print("Creating debug visualizations...")
    splitter.create_debug_visualization(images, segments)
    
    print("Creating products...")
    products = splitter.create_products_from_segments(segments)
    
    print(f"\nCreated {len(products)} products:")
    for product in products:
        print(f"  {product['filename']}: {product['bottom_segment']['source']} + {product['top_segment']['source']}")
    
    # Save processing info
    info = {
        'total_images': len(images),
        'total_segments': len(segments),
        'total_products': len(products),
        'products': products
    }
    
    info_path = Path(output_dir) / "processing_info.json"
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"\nProcessing complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()